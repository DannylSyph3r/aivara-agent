import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from logging_utils import configure_logging
from config import (
    AIVARA_AGENT_API_KEY,
    AIVARA_AGENT_URL,
    PO_FHIR_EXTENSION_URI,
    validate_config,
)

configure_logging(os.getenv("LOG_LEVEL", "INFO"))
validate_config()

logger = logging.getLogger(__name__)

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentSkill,
)
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from agent import root_agent
from middleware import ApiKeyMiddleware

from typing import Any
from pydantic import Field


class FriendlyErrorMiddleware(BaseHTTPMiddleware):
    """Rewrites Gemini 503 error text in A2A JSON responses with a user-friendly message."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        if request.method != "POST":
            return response

        body = b""
        try:
            async for chunk in response.body_iterator:
                body += chunk

            payload = json.loads(body)
            result = payload.get("result", {})

            for key in ("error", "message", "text"):
                val = str(result.get(key, ""))
                if "503" in val or "UNAVAILABLE" in val or "high demand" in val:
                    result[key] = (
                        "The AI model is temporarily busy. "
                        "Please try again in a few seconds."
                    )
                    payload["result"] = result
                    break

            # Strip stale Content-Length so Starlette sets the correct value.
            headers = {
                k: v for k, v in response.headers.items()
                if k.lower() != "content-length"
            }

            return Response(
                content=json.dumps(payload),
                status_code=response.status_code,
                headers=headers,
                media_type="application/json",
            )

        except Exception:
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
            )


class PoCompatibilityMiddleware(BaseHTTPMiddleware):
    """Bridges PO A2A v1 client incompatibilities with the a2a-sdk 0.3.x stack."""

    _METHOD_ALIASES: dict[str, str] = {
        "SendMessage":          "message/send",
        "SendStreamingMessage": "message/send",
        "GetTask":              "tasks/get",
        "CancelTask":           "tasks/cancel",
        "TaskResubscribe":      "tasks/resubscribe",
    }
    _ROLE_ALIASES: dict[str, str] = {
        "ROLE_USER":  "user",
        "ROLE_AGENT": "agent",
    }
    _STATE_MAP: dict[str, str] = {
        "completed":      "TASK_STATE_COMPLETED",
        "working":        "TASK_STATE_WORKING",
        "submitted":      "TASK_STATE_SUBMITTED",
        "input-required": "TASK_STATE_INPUT_REQUIRED",
        "failed":         "TASK_STATE_FAILED",
        "canceled":       "TASK_STATE_CANCELED",
    }

    async def dispatch(self, request: Request, call_next):
        if request.method != "POST":
            return await call_next(request)

        body_bytes = await request.body()
        body_dirty = False
        parsed: dict = {}
        try:
            parsed = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            parsed = {}

        # Method rewriting
        if isinstance(parsed, dict) and parsed.get("method") in self._METHOD_ALIASES:
            original = parsed["method"]
            parsed["method"] = self._METHOD_ALIASES[original]
            body_dirty = True
            logger.info(
                "jsonrpc_method_rewritten original=%s rewritten=%s",
                original, parsed["method"],
            )

        # Role normalisation
        def _fix_roles(node):
            if isinstance(node, dict):
                if "role" in node and node["role"] in self._ROLE_ALIASES:
                    node["role"] = self._ROLE_ALIASES[node["role"]]
                for v in node.values():
                    _fix_roles(v)
            elif isinstance(node, list):
                for item in node:
                    _fix_roles(item)

        if isinstance(parsed, dict):
            snapshot = json.dumps(parsed, sort_keys=True)
            _fix_roles(parsed)
            if json.dumps(parsed, sort_keys=True) != snapshot:
                body_dirty = True
                logger.info("jsonrpc_roles_normalised")

        if body_dirty:
            body_bytes = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
            request._body = body_bytes

        response = await call_next(request)

        # Response reshaping — only for JSON responses
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return response

        resp_body = b""
        try:
            async for chunk in response.body_iterator:
                resp_body += chunk if isinstance(chunk, bytes) else chunk.encode()

            resp_parsed = json.loads(resp_body)
            result = resp_parsed.get("result") if isinstance(resp_parsed, dict) else None

            if isinstance(result, dict) and result.get("kind") == "task":
                task: dict = {
                    "id":        result.get("id"),
                    "contextId": result.get("contextId"),
                }
                raw_state = result.get("status", {}).get("state", "")
                task["status"] = {
                    "state": self._STATE_MAP.get(raw_state, raw_state.upper())
                }
                clean_artifacts = []
                for artifact in result.get("artifacts", []):
                    clean_parts = [
                        {k: v for k, v in part.items() if k != "kind"}
                        for part in artifact.get("parts", [])
                    ]
                    clean_artifact = {k: v for k, v in artifact.items() if k != "parts"}
                    clean_artifact["parts"] = clean_parts
                    clean_artifacts.append(clean_artifact)
                task["artifacts"] = clean_artifacts
                resp_parsed["result"] = {"task": task}
                logger.info(
                    "response_reshaped task_id=%s state=%s",
                    task.get("id"), task["status"]["state"],
                )

            resp_body = json.dumps(resp_parsed, ensure_ascii=False).encode("utf-8")
        except Exception:
            pass

        headers = dict(response.headers)
        headers["content-length"] = str(len(resp_body))
        return Response(
            content=resp_body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )


class AgentCardV1(AgentCard):
    """AgentCard subclass that adds supportedInterfaces and relaxes securitySchemes typing for A2A v1."""
    supportedInterfaces: list[dict[str, Any]] = Field(default_factory=list)
    securitySchemes: dict[str, Any] | None = None  # override parent's typed field


skills = [
    AgentSkill(
        id="patient-summary",
        name="Patient Summary",
        description=(
            "Retrieves and synthesises patient demographics, conditions, and latest "
            "clinical note into a coherent overview."
        ),
        tags=["demographics", "conditions", "clinical-notes", "fhir"],
        examples=[
            "Give me a summary of this patient",
            "What do I need to know before seeing this patient?",
        ],
    ),
    AgentSkill(
        id="risk-assessment",
        name="Risk Assessment",
        description=(
            "Analyses health burden scores (DALY/QALY), active conditions, observations, "
            "and medications to produce a clinical risk narrative."
        ),
        tags=["risk", "daly", "qaly", "conditions", "observations", "medications", "fhir"],
        examples=[
            "What are the health risks for this patient?",
            "Assess this patient's disease burden",
        ],
    ),
    AgentSkill(
        id="medication-safety",
        name="Medication Safety Review",
        description=(
            "Reviews current medications against known allergies and active conditions "
            "to surface potential safety concerns."
        ),
        tags=["medications", "allergies", "safety", "fhir"],
        examples=[
            "Are there any medication concerns?",
            "What is this patient currently taking?",
        ],
    ),
    AgentSkill(
        id="visit-history",
        name="Visit History Analysis",
        description=(
            "Retrieves encounter timeline, linked clinical notes, and procedures to "
            "provide a longitudinal view of the patient's care history."
        ),
        tags=["encounters", "documents", "procedures", "history", "fhir"],
        examples=[
            "Show me this patient's recent visits",
            "What happened in the last encounter?",
        ],
    ),
]

agent_card = AgentCardV1(
    name="AskAivara",
    description=(
        "Aivara is a clinical intelligence agent that reasons across a patient's "
        "full FHIR health record. It chains clinical data tools to answer complex "
        "questions that require synthesis — patient summaries, risk assessments, "
        "medication safety reviews, and visit history analysis."
    ),
    url=AIVARA_AGENT_URL,
    version="1.0.0",
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain"],
    capabilities=AgentCapabilities(
        streaming=False,
        pushNotifications=False,
        extensions=[
            AgentExtension(
                uri=PO_FHIR_EXTENSION_URI,
                description="FHIR R4 context — patient ID, FHIR URL, and access token.",
                required=True,
            )
        ],
    ),
    supportedInterfaces=[
        {
            "url": AIVARA_AGENT_URL,
            "protocolBinding": "JSONRPC",
            "protocolVersion": "1.0",
        }
    ],
    skills=skills,
    securitySchemes={
        "apiKey": {
            "apiKeySecurityScheme": {
                "name": "X-API-Key",
                "location": "header",
                "description": "API key required to access this agent.",
            }
        }
    },
    security=[{"apiKey": []}],
)

# to_a2a() does not accept api_key — middleware is attached manually below.
a2a_app = to_a2a(
    agent=root_agent,
    agent_card=agent_card,
)
a2a_app.add_middleware(ApiKeyMiddleware)
a2a_app.add_middleware(FriendlyErrorMiddleware)
a2a_app.add_middleware(PoCompatibilityMiddleware)

logger.info(
    "aivara_agent_startup log_level=%s url=%s",
    os.getenv("LOG_LEVEL", "INFO").upper(),
    AIVARA_AGENT_URL,
)