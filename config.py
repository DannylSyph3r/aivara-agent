import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

AIVARA_AGENT_API_KEY: str = os.getenv("AIVARA_AGENT_API_KEY", "")
AIVARA_AGENT_URL: str = os.getenv("AIVARA_AGENT_URL", "http://localhost:8001")

PO_FHIR_EXTENSION_URI: str = os.getenv(
    "PO_FHIR_EXTENSION_URI",
    "https://app.promptopinion.ai/schemas/a2a/v1/fhir-context",
)

# Substring matched against A2A metadata keys to locate the FHIR context block.
FHIR_CONTEXT_KEY: str = "fhir-context"


def validate_config() -> None:
    """Raise ValueError at startup if any required environment variable is absent."""
    required = ["GOOGLE_API_KEY", "AIVARA_AGENT_API_KEY"]

    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Check your .env file or Railway dashboard."
        )