"""
Environment variable loading and application-wide constants.

All modules import their values from here — no module reads os.getenv() directly
except middleware.py, which is intentionally decoupled.

validate_config() is called once at startup in app.py.

Requires GOOGLE_API_KEY (Google AI Studio) and AIVARA_AGENT_API_KEY.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# ── Model ─────────────────────────────────────────────────────────────────────

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# ── Agent identity ─────────────────────────────────────────────────────────────

AIVARA_AGENT_API_KEY: str = os.getenv("AIVARA_AGENT_API_KEY", "")
AIVARA_AGENT_URL: str = os.getenv("AIVARA_AGENT_URL", "http://localhost:8001")

# ── Prompt Opinion ─────────────────────────────────────────────────────────────

PO_FHIR_EXTENSION_URI: str = os.getenv(
    "PO_FHIR_EXTENSION_URI",
    "https://app.promptopinion.ai/schemas/a2a/v1/fhir-context",
)

# Substring matched against A2A metadata keys to locate the FHIR context block.
FHIR_CONTEXT_KEY: str = "fhir-context"

# ── Startup validation ─────────────────────────────────────────────────────────

def validate_config() -> None:
    """
    Raise ValueError at startup if any required environment variable is absent.

    Called once in app.py after load_dotenv() — never at import time of this module.
    """
    required = ["GOOGLE_API_KEY", "AIVARA_AGENT_API_KEY"]

    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Check your .env file or Railway dashboard."
        )