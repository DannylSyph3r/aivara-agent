"""
API key validation and enforcement middleware.

validate_api_key() — standalone helper for testing.
ApiKeyMiddleware   — Starlette middleware attached to the A2A app in app.py.
                     Keeps /.well-known/agent-card.json public;
                     blocks everything else without a valid X-API-Key.
"""
import logging
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

_AGENT_CARD_PATH = "/.well-known/agent-card.json"


def validate_api_key(api_key: str | None) -> bool:
    """Validate an inbound X-API-Key value against AIVARA_AGENT_API_KEY."""
    expected = os.getenv("AIVARA_AGENT_API_KEY")
    if not expected:
        logger.error("security_misconfigured AIVARA_AGENT_API_KEY not set")
        return False
    return api_key == expected


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """
    Enforce X-API-Key on all endpoints except the agent card.

    The agent card must remain public so callers can discover the agent's
    security requirements before they have a key to authenticate with.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path == _AGENT_CARD_PATH:
            return await call_next(request)

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