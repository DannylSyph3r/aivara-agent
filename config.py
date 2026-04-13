"""
Environment variable loading and application-wide constants.

All modules import their values from here — no module reads os.getenv() directly
except middleware.py, which is intentionally decoupled.

validate_config() is called once at startup in app.py.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# ── Model & MCP ───────────────────────────────────────────────────────────────

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

AIVARA_MCP_URL: str = os.getenv(
    "AIVARA_MCP_URL",
    "https://aivara-tools.up.railway.app/mcp",
)
AIVARA_MCP_API_KEY: str = os.getenv("AIVARA_MCP_API_KEY", "")

# ── Agent identity ─────────────────────────────────────────────────────────────

AIVARA_AGENT_API_KEY: str = os.getenv("AIVARA_AGENT_API_KEY", "")
AIVARA_AGENT_URL: str = os.getenv("AIVARA_AGENT_URL", "http://localhost:8001")

# ── Prompt Opinion ─────────────────────────────────────────────────────────────

PO_FHIR_EXTENSION_URI: str = os.getenv(
    "PO_FHIR_EXTENSION_URI",
    "https://app.promptopinion.ai/schemas/a2a/v1/fhir-context",
)

# Substring matched against A2A metadata keys to locate the FHIR context block.
# PO sends the full URI as the key; we match by substring so the exact URI
# format can vary without requiring a code change.
FHIR_CONTEXT_KEY: str = "fhir-context"

# ── Startup validation ─────────────────────────────────────────────────────────

_REQUIRED_ENV_VARS: list[str] = [
    "GOOGLE_API_KEY",
    "AIVARA_MCP_API_KEY",
    "AIVARA_AGENT_API_KEY",
]


def validate_config() -> None:
    """
    Raise ValueError at startup if any required environment variable is absent.

    Called once in app.py after load_dotenv() — never at import time of this module.
    """
    missing = [key for key in _REQUIRED_ENV_VARS if not os.getenv(key)]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Check your .env file or Railway dashboard."
        )