"""
Environment variable loading and application-wide constants.

All modules import their values from here — no module reads os.getenv() directly
except middleware.py, which is intentionally decoupled.

validate_config() is called once at startup in app.py.

Vertex AI mode:
  Set GOOGLE_GENAI_USE_VERTEXAI=True and provide GOOGLE_CLOUD_PROJECT,
  GOOGLE_CLOUD_LOCATION, and SERVICE_ACCOUNT_JSON (full key file contents).
  The credential file is written to /tmp/google-creds.json at import time
  so it exists before any Google SDK module is imported.

AI Studio mode (default):
  Set GOOGLE_GENAI_USE_VERTEXAI=False (or omit) and provide GOOGLE_API_KEY.
"""
import os

from dotenv import load_dotenv

load_dotenv()


# ── Vertex AI credential setup ─────────────────────────────────────────────────
# Must run before any Google SDK import. Writes SERVICE_ACCOUNT_JSON to a
# temp file and points GOOGLE_APPLICATION_CREDENTIALS at it so the SDK finds
# it automatically. No-ops if SERVICE_ACCOUNT_JSON is not set.

def _setup_vertex_credentials() -> None:
    sa_json = os.getenv("SERVICE_ACCOUNT_JSON")
    if not sa_json:
        return
    creds_path = "/tmp/google-creds.json"
    with open(creds_path, "w") as f:
        f.write(sa_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path


_setup_vertex_credentials()


# ── Backend mode ───────────────────────────────────────────────────────────────

_USE_VERTEX: bool = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"

# ── Model & MCP ───────────────────────────────────────────────────────────────

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

AIVARA_MCP_URL: str = os.getenv(
    "AIVARA_MCP_URL",
    "https://aivara-tools.up.railway.app/mcp",
)
AIVARA_MCP_API_KEY: str = os.getenv("AIVARA_MCP_API_KEY", "")

# ── Vertex AI project ──────────────────────────────────────────────────────────

GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

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

    Vertex AI mode requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.
    AI Studio mode requires GOOGLE_API_KEY.
    Both modes require AIVARA_MCP_API_KEY and AIVARA_AGENT_API_KEY.

    Called once in app.py after load_dotenv() — never at import time of this module.
    """
    if _USE_VERTEX:
        required = [
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "AIVARA_MCP_API_KEY",
            "AIVARA_AGENT_API_KEY",
        ]
    else:
        required = [
            "GOOGLE_API_KEY",
            "AIVARA_MCP_API_KEY",
            "AIVARA_AGENT_API_KEY",
        ]

    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Check your .env file or Railway dashboard."
        )