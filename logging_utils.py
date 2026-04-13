"""
Logging utilities shared across all modules.

configure_logging() — call once at startup in app.py.
token_fingerprint() — safe summary of a bearer token for log lines; never logs the value.
safe_pretty_json()  — JSON serialisation that never raises.
"""
import hashlib
import json
import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def token_fingerprint(token: str | None) -> str:
    """Return a safe log-friendly fingerprint of a bearer token — never the value itself."""
    if not token:
        return "len=0"
    digest = hashlib.sha256(token.encode()).hexdigest()[:8]
    return f"len={len(token)} sha256={digest}"


def safe_pretty_json(obj) -> str:
    """Serialize obj to indented JSON, falling back to repr on failure."""
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return repr(obj)