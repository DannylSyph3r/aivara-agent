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
    import json as _json

    sa_json = os.getenv("SERVICE_ACCOUNT_JSON", "").strip()
    if not sa_json:
        return

    creds_path = "/tmp/google-creds.json"

    # Extract the raw JSON object by finding the outermost { } braces —
    # strips any surrounding quotes Railway adds when storing the value.
    start = sa_json.find("{")
    end = sa_json.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "SERVICE_ACCOUNT_JSON does not contain a JSON object. "
            "Ensure the variable contains the raw minified JSON key file contents."
        )
    sa_json = sa_json[start : end + 1]

    # strict=False allows control characters (e.g. actual newlines) in string
    # values — Railway converts \n escape sequences to real newlines when storing
    # env vars, which standard json.loads() rejects.
    try:
        creds = _json.JSONDecoder(strict=False).decode(sa_json)
    except _json.JSONDecodeError as e:
        raise ValueError(
            f"SERVICE_ACCOUNT_JSON is not valid JSON: {e}."
        ) from e

    with open(creds_path, "w") as f:
        _json.dump(creds, f)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path


_setup_vertex_credentials()


# ── Backend mode ───────────────────────────────────────────────────────────────

_USE_VERTEX: bool = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"

# ── Model ─────────────────────────────────────────────────────────────────────

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

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
    Both modes require AIVARA_AGENT_API_KEY.

    Called once in app.py after load_dotenv() — never at import time of this module.
    """
    if _USE_VERTEX:
        required = [
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "AIVARA_AGENT_API_KEY",
        ]
    else:
        required = [
            "GOOGLE_API_KEY",
            "AIVARA_AGENT_API_KEY",
        ]

    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Check your .env file or Railway dashboard."
        )