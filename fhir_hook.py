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


def _safe_correlation_ids(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> dict[str, str | None]:
    """Extract invocation_id and agent_name from ADK context — the stable per-turn identifiers."""
    return {
        "invocation_id": getattr(callback_context, "invocation_id", None),
        "agent_name":    getattr(callback_context, "agent_name",    None),
    }


def _extract_metadata(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> dict:
    """Walk all known ADK metadata locations in priority order and return the first populated dict."""
    run_config = getattr(callback_context, "run_config", None)
    custom_metadata = getattr(run_config, "custom_metadata", None) if run_config else None
    a2a_metadata = (
        custom_metadata.get("a2a_metadata")
        if isinstance(custom_metadata, dict)
        else None
    )

    candidates = [
        getattr(callback_context, "metadata", None),
        a2a_metadata,
        getattr(llm_request, "metadata", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def _coerce_fhir_data(value) -> dict | None:
    """Accept a dict or JSON string and return a dict, or None on failure."""
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
    """Prepend the patient ID header to the system instruction in the LlmRequest."""
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


def extract_fhir_context(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """ADK before_model_callback — extracts FHIR credentials into session state and guards against missing patient ID."""
    correlation = _safe_correlation_ids(callback_context, llm_request)

    if LOG_HOOK_RAW_OBJECTS:
        logger.info(
            "hook_raw_llm_request type=%s attrs=%s",
            type(llm_request).__name__,
            safe_pretty_json(dir(llm_request)),
        )
        logger.info(
            "hook_raw_callback_context invocation_id=%s agent=%s metadata=%s state=%s",
            correlation["invocation_id"],
            correlation["agent_name"],
            safe_pretty_json(str(getattr(callback_context, "metadata", None))),
            safe_pretty_json(str(getattr(callback_context, "state", None))),
        )

    metadata = _extract_metadata(callback_context, llm_request)
    metadata_keys = list(metadata.keys())

    logger.info(
        "hook_called_enter invocation_id=%s agent=%s metadata_keys=%s",
        correlation["invocation_id"],
        correlation["agent_name"],
        metadata_keys,
    )

    if not metadata:
        logger.info(
            "hook_called_no_metadata invocation_id=%s agent=%s",
            correlation["invocation_id"],
            correlation["agent_name"],
        )

    fhir_data = None
    for key, value in metadata.items():
        if FHIR_CONTEXT_KEY in str(key):
            fhir_data = _coerce_fhir_data(value)
            if fhir_data is None:
                logger.warning(
                    "hook_called_fhir_malformed invocation_id=%s agent=%s "
                    "metadata_key=%s value_type=%s",
                    correlation["invocation_id"],
                    correlation["agent_name"],
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
            "hook_called_fhir_found invocation_id=%s agent=%s "
            "patient_id=%s fhir_url_set=%s fhir_token=%s",
            correlation["invocation_id"],
            correlation["agent_name"],
            patient_id or "[EMPTY]",
            bool(fhir_url),
            token_fingerprint(fhir_token),
        )
    else:
        callback_context.state["fhir_url"]   = ""
        callback_context.state["fhir_token"] = ""
        callback_context.state["patient_id"] = ""
        logger.info(
            "hook_called_fhir_not_found invocation_id=%s agent=%s metadata_keys=%s",
            correlation["invocation_id"],
            correlation["agent_name"],
            metadata_keys,
        )

    patient_id = callback_context.state.get("patient_id", "")

    if not patient_id:
        logger.warning(
            "hook_patient_id_missing invocation_id=%s agent=%s "
            "— skipping LLM to prevent hallucination",
            correlation["invocation_id"],
            correlation["agent_name"],
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
        "hook_patient_id_resolved invocation_id=%s agent=%s patient_id=%s",
        correlation["invocation_id"],
        correlation["agent_name"],
        patient_id,
    )
    return None