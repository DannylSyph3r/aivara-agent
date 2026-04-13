"""
A2A application entry point.

Import order is deliberate:
  1. load_dotenv() — must run before any local module reads os.getenv()
  2. configure_logging() — set up logging before anything emits log lines
  3. validate_config() — fail fast at startup if required keys are missing
  4. Everything else

The agent card is served publicly at GET /.well-known/agent-card.json.
All other endpoints require X-API-Key, enforced by ApiKeyMiddleware.
"""
import logging
import os

from dotenv import load_dotenv

# Step 1 — environment must be loaded before any local import reads os.getenv().
# config.py also calls load_dotenv() defensively; calling it here first ensures
# LOG_LEVEL is available for configure_logging() before config is imported.
load_dotenv()

from logging_utils import configure_logging
from config import (
    AIVARA_AGENT_API_KEY,
    AIVARA_AGENT_URL,
    PO_FHIR_EXTENSION_URI,
    validate_config,
)

# Step 2 — logging before anything else emits lines.
configure_logging(os.getenv("LOG_LEVEL", "INFO"))

# Step 3 — fail fast if the environment is incomplete.
validate_config()

logger = logging.getLogger(__name__)

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentSkill,
    APIKeySecurityScheme,
    In,
    SecurityScheme,
)
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from agent import root_agent
from middleware import ApiKeyMiddleware

# ── Agent skills ───────────────────────────────────────────────────────────────

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

# ── Agent card ─────────────────────────────────────────────────────────────────

agent_card = AgentCard(
    name="Aivara",
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
    skills=skills,
    securitySchemes={
        "apiKey": SecurityScheme(
            root=APIKeySecurityScheme(
                type="apiKey",
                name="X-API-Key",
                in_=In.header,
                description="API key required to access this agent.",
            )
        )
    },
    security=[{"apiKey": []}],
)

# ── A2A application ────────────────────────────────────────────────────────────
# to_a2a() does not accept api_key — ApiKeyMiddleware is attached manually.
# Starlette builds the middleware stack lazily on the first request, so
# add_middleware() called here (at import time) is safe before uvicorn starts.

a2a_app = to_a2a(
    agent=root_agent,
    agent_card=agent_card,
)
a2a_app.add_middleware(ApiKeyMiddleware)

logger.info(
    "aivara_agent_startup log_level=%s url=%s",
    os.getenv("LOG_LEVEL", "INFO").upper(),
    AIVARA_AGENT_URL,
)