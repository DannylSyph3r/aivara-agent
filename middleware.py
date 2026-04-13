"""
API key validation helper.

The A2A SDK's to_a2a(api_key=...) enforces X-API-Key on every inbound request
automatically — no Starlette middleware class is needed here. This module
provides an explicit validate_api_key() helper for test contexts and future use.

What never to log:
  - The key value itself
  - Any portion of the FHIR access token
"""
import logging
import os

logger = logging.getLogger(__name__)


def validate_api_key(api_key: str | None) -> bool:
    """
    Validate an inbound X-API-Key value against AIVARA_AGENT_API_KEY.

    Reads directly from os.getenv() to remain decoupled from config.py
    and usable in isolation (e.g. scripts, tests).
    """
    expected = os.getenv("AIVARA_AGENT_API_KEY")
    if not expected:
        logger.error("security_misconfigured AIVARA_AGENT_API_KEY not set")
        return False
    return api_key == expected