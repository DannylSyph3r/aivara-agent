"""
API key validation, enforcement, and FHIR metadata bridging.

validate_api_key() — standalone helper.
ApiKeyMiddleware   — Starlette middleware attached to the A2A app in app.py.

Two responsibilities:
  1. Keep /.well-known/agent-card.json public; block everything else without
     a valid X-API-Key.
  2. Bridge FHIR metadata from params.message.metadata → params.metadata so
     that ADK's before_model_callback can find it. PO places the FHIR context
     in params.message.metadata; ADK surfaces params.metadata to callbacks.
     Without this bridge the hook never sees the patient ID.
"""
import json
import logging
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

_AGENT_CARD_PATH = "/.well-known/agent-card.json"

# Must match FHIR_CONTEXT_KEY in config.py — substring matched against metadata keys.
_FHIR_CONTEXT_KEY = "fhir-context"


def validate_api_key(api_key: str | None) -> bool:
    """Validate an inbound X-API-Key value against AIVARA_AGENT_API_KEY."""
    expected = os.getenv("AIVARA_AGENT_API_KEY")
    if not expected:
        logger.error("security_misconfigured AIVARA_AGENT_API_KEY not set")
        return False
    return api_key == expected


def _bridge_fhir_metadata(parsed: dict, original_bytes: bytes) -> bytes:
    """
    Copy FHIR context from params.message.metadata → params.metadata.

    PO sends the FHIR context in params.message.metadata. ADK's callback
    system surfaces params.metadata to before_model_callback, not
    params.message.metadata. Without this bridge the hook finds nothing.

    Returns original_bytes unchanged if no bridging is needed or possible.
    """
    if not isinstance(parsed, dict):
        return original_bytes

    params = parsed.get("params")
    if not isinstance(params, dict):
        return original_bytes

    # Already bridged or params.metadata already has FHIR context — skip.
    existing_metadata = params.get("metadata")
    if isinstance(existing_metadata, dict):
        for key in existing_metadata:
            if _FHIR_CONTEXT_KEY in str(key):
                return original_bytes

    # Look for FHIR context in params.message.metadata.
    message = params.get("message")
    if not isinstance(message, dict):
        return original_bytes

    message_metadata = message.get("metadata")
    if not isinstance(message_metadata, dict):
        return original_bytes

    for key, value in message_metadata.items():
        if _FHIR_CONTEXT_KEY in str(key):
            params["metadata"] = {key: value}
            logger.debug("fhir_metadata_bridged key=%s", key)
            return json.dumps(parsed, ensure_ascii=False).encode("utf-8")

    return original_bytes


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """
    Enforce X-API-Key on all endpoints except the agent card, and bridge
    FHIR metadata from params.message.metadata to params.metadata.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path == _AGENT_CARD_PATH:
            return await call_next(request)

        # Read and buffer the body — required for both bridging and key check.
        body_bytes = await request.body()

        # Bridge FHIR metadata before passing downstream.
        if body_bytes:
            try:
                parsed = json.loads(body_bytes)
                bridged = _bridge_fhir_metadata(parsed, body_bytes)
                if bridged is not body_bytes:
                    # Replace Starlette's cached body so downstream sees the
                    # modified payload when it calls request.body().
                    request._body = bridged
            except json.JSONDecodeError:
                pass

        api_key = request.headers.get("X-API-Key")
        if not validate_api_key(api_key):
            logger.warning(
                "security_rejected path=%s method=%s",
                request.url.path,
                request.method,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "detail": "Valid X-API-Key header required.",
                },
            )

        return await call_next(request)