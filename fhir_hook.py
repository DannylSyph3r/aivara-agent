"""
FHIR context hook — ADK before_model_callback.

Runs before every LLM call. Extracts FHIR credentials from A2A message metadata
and writes them into session state so they are available to tools at call time
without ever appearing in the prompt.

Critical guardrail: if patientId is absent or empty after extraction, this
function returns LlmResponse directly. The LLM never runs. This is code-level
enforcement — an instruction-based guard can be ignored by the model; this cannot.

Patient ID injection: when patientId is present, it is prepended to the system
instruction in the LlmRequest directly. The instruction template cannot use a
{patient_id} placeholder because ADK renders template variables during preprocessing
— before this callback runs — which raises KeyError when patient_id is not yet in
session state. Injecting via llm_request.config.system_instruction is the correct
ADK-documented pattern for dynamic instruction modification at callback time.

Metadata key convention (must match PO_FHIR_EXTENSION_URI in config.py):
    "https://app.promptopinion.ai/schemas/a2a/v1/fhir-context": {
        "fhirUrl":   "https://...",
        "fhirToken": "<bearer-token>",
        "patientId": "<uuid>"
    }

Set LOG_HOOK_RAW_OBJECTS=true in .env to dump raw ADK objects during integration
testing. Set to false before production.
"""
import json
import logging
import os
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from config import FHIR_CONTEXT_KEY
from logging_utils import safe_pretty_json, token_fingerprint

logger = logging.getLogger(__name__)

LOG_HOOK_RAW_OBJECTS: bool = os.getenv("LOG_HOOK_RAW_OBJECTS", "false").lower() == "true"


# ── Private helpers ────────────────────────────────────────────────────────────

def _safe_correlation_ids(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> dict[str, str | None]:
    """Extract correlation IDs from ADK objects for structured log lines."""
    def _first(*values):
        return next((v for v in values if v not in (None, "")), None)

    return {
        "task_id": _first(
            getattr(llm_request, "task_id", None),
            getattr(callback_context, "task_id", None),
        ),
        "context_id": _first(
            getattr(llm_request, "context_id", None),
            getattr(callback_context, "context_id", None),
        ),
        "message_id": _first(
            getattr(llm_request, "message_id", None),
            getattr(callback_context, "message_id", None),
        ),
    }


def _extract_metadata(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> dict:
    """
    Walk all known ADK metadata locations in priority order.

    ADK surfaces A2A metadata differently depending on whether the request
    arrived via message/send or message/stream. Checking all known locations
    ensures we find it regardless of transport path.
    """
    candidates = [
        getattr(callback_context, "metadata", None),
        getattr(llm_request, "metadata", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def _coerce_fhir_data(value) -> dict | None:
    """
    Accept either a dict or a JSON string; return a dict or None.

    A malformed value must not crash the hook — return None and let the
    caller log a warning and proceed to the empty-patient-ID guardrail.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _inject_patient_context(llm_request: LlmRequest, patient_id: str) -> None:
    """
    Prepend patient ID context to the system instruction in the LlmRequest.

    Called after patient_id is confirmed non-empty. Always outputs a plain
    string to system_instruction regardless of the incoming type (str, Content,
    or None), which ADK handles correctly.
    """
    if llm_request.config is None:
        logger.warning("hook_inject_skipped llm_request.config is None")
        return

    patient_header = (
        f"CURRENT PATIENT ID: {patient_id}\n"
        f"Always pass this exact value as the patientId argument to "
        f"every clinical tool that requires it.\n\n"
    )

    si = llm_request.config.system_instruction
    if si is None:
        existing = ""
    elif isinstance(si, str):
        existing = si
    elif hasattr(si, "parts"):
        existing = "".join(
            p.text for p in si.parts if getattr(p, "text", None)
        )
    else:
        existing = str(si)

    llm_request.config.system_instruction = patient_header + existing
    logger.info("hook_patient_context_injected patient_id=%s", patient_id)


# ── Public callback ────────────────────────────────────────────────────────────

def extract_fhir_context(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """
    ADK before_model_callback.

    Extracts FHIR credentials from A2A message metadata into session state.
    All three values (fhir_url, fhir_token, patient_id) are stored even though
    the MCP server currently uses anonymous FHIR access — this keeps state ready
    if authentication requirements change.

    Returns:
        None         — patient ID resolved; allow normal LLM execution.
        LlmResponse  — patient ID missing; short-circuit with user-friendly message.
    """
    correlation = _safe_correlation_ids(callback_context, llm_request)

    if LOG_HOOK_RAW_OBJECTS:
        logger.info(
            "hook_raw_llm_request type=%s attrs=%s",
            type(llm_request).__name__,
            safe_pretty_json(dir(llm_request)),
        )
        logger.info(
            "hook_raw_callback_context task_id=%s context_id=%s metadata=%s state=%s",
            getattr(callback_context, "task_id", None),
            getattr(callback_context, "context_id", None),
            safe_pretty_json(str(getattr(callback_context, "metadata", None))),
            safe_pretty_json(str(getattr(callback_context, "state", None))),
        )

    metadata = _extract_metadata(callback_context, llm_request)
    metadata_keys = list(metadata.keys())

    logger.info(
        "hook_called_enter task_id=%s context_id=%s message_id=%s metadata_keys=%s",
        correlation["task_id"],
        correlation["context_id"],
        correlation["message_id"],
        metadata_keys,
    )

    if not metadata:
        logger.info(
            "hook_called_no_metadata task_id=%s context_id=%s message_id=%s",
            correlation["task_id"],
            correlation["context_id"],
            correlation["message_id"],
        )

    # Locate the FHIR entry by substring match so the full URI variant doesn't matter.
    fhir_data = None
    for key, value in metadata.items():
        if FHIR_CONTEXT_KEY in str(key):
            fhir_data = _coerce_fhir_data(value)
            if fhir_data is None:
                logger.warning(
                    "hook_called_fhir_malformed task_id=%s context_id=%s message_id=%s "
                    "metadata_key=%s value_type=%s",
                    correlation["task_id"],
                    correlation["context_id"],
                    correlation["message_id"],
                    key,
                    type(value).__name__,
                )
            break

    if fhir_data:
        fhir_url   = fhir_data.get("fhirUrl",   "")
        fhir_token = fhir_data.get("fhirToken", "")
        patient_id = fhir_data.get("patientId", "")

        callback_context.state["fhir_url"]   = fhir_url
        callback_context.state["fhir_token"] = fhir_token
        callback_context.state["patient_id"] = patient_id

        logger.info(
            "hook_called_fhir_found task_id=%s context_id=%s message_id=%s "
            "patient_id=%s fhir_url_set=%s fhir_token=%s",
            correlation["task_id"],
            correlation["context_id"],
            correlation["message_id"],
            patient_id or "[EMPTY]",
            bool(fhir_url),
            token_fingerprint(fhir_token),
        )
    else:
        callback_context.state["fhir_url"]   = ""
        callback_context.state["fhir_token"] = ""
        callback_context.state["patient_id"] = ""
        logger.info(
            "hook_called_fhir_not_found task_id=%s context_id=%s message_id=%s "
            "metadata_keys=%s",
            correlation["task_id"],
            correlation["context_id"],
            correlation["message_id"],
            metadata_keys,
        )

    patient_id = callback_context.state.get("patient_id", "")

    if not patient_id:
        logger.warning(
            "hook_patient_id_missing task_id=%s context_id=%s message_id=%s "
            "— skipping LLM to prevent hallucination",
            correlation["task_id"],
            correlation["context_id"],
            correlation["message_id"],
        )
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=(
                    "No patient is currently selected. Please select a patient "
                    "from the Prompt Opinion launchpad before asking clinical questions."
                ))],
            )
        )

    # Patient ID confirmed — inject into system instruction before model call.
    _inject_patient_context(llm_request, patient_id)

    logger.info(
        "hook_patient_id_resolved task_id=%s context_id=%s message_id=%s patient_id=%s",
        correlation["task_id"],
        correlation["context_id"],
        correlation["message_id"],
        patient_id,
    )
    return None