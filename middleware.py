import json
import logging
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

_AGENT_CARD_PATH = "/.well-known/agent-card.json"

# Substring matched against metadata keys — must match FHIR_CONTEXT_KEY in config.py.
_FHIR_CONTEXT_KEY = "fhir-context"


def validate_api_key(api_key: str | None) -> bool:
    """Validate an inbound X-API-Key value against AIVARA_AGENT_API_KEY."""
    expected = os.getenv("AIVARA_AGENT_API_KEY")
    if not expected:
        logger.error("security_misconfigured AIVARA_AGENT_API_KEY not set")
        return False
    return api_key == expected


def _bridge_fhir_metadata(parsed: dict, original_bytes: bytes) -> bytes:
    """Copy FHIR context from params.message.metadata → params.metadata for ADK callback visibility."""
    if not isinstance(parsed, dict):
        return original_bytes

    params = parsed.get("params")
    if not isinstance(params, dict):
        return original_bytes

    # Skip if params.metadata already contains FHIR context.
    existing_metadata = params.get("metadata")
    if isinstance(existing_metadata, dict):
        for key in existing_metadata:
            if _FHIR_CONTEXT_KEY in str(key):
                return original_bytes

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
    """Enforce X-API-Key on all endpoints except the agent card, and bridge FHIR metadata."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path == _AGENT_CARD_PATH:
            return await call_next(request)

        body_bytes = await request.body()

        if body_bytes:
            try:
                parsed = json.loads(body_bytes)
                bridged = _bridge_fhir_metadata(parsed, body_bytes)
                if bridged is not body_bytes:
                    # Replace Starlette's cached body so downstream sees the modified payload.
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