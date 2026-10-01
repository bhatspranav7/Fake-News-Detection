"""Environment-driven settings. Everything has a local default."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


APP_NAME = "VeriFact"
VERSION = "1.0.0"

MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "app.db"))

# LLM provider: ollama (local) | groq (cloud, free tier) | openai-compatible | none
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()


def _ollama_url(raw: str) -> str:
    """OLLAMA_HOST is also the *server's* bind address (e.g. `0.0.0.0`,
    `0.0.0.0:11434`), so normalise whatever we find into a client URL."""
    raw = (raw or "").strip().rstrip("/")
    if not raw:
        return "http://127.0.0.1:11434"
    if "://" not in raw:
        raw = "http://" + raw
    from urllib.parse import urlparse, urlunparse

    u = urlparse(raw)
    host = u.hostname or "127.0.0.1"
    if host in {"0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    port = u.port or 11434
    return urlunparse((u.scheme, f"{host}:{port}", "", "", "", ""))


OLLAMA_HOST = _ollama_url(os.getenv("OLLAMA_HOST", ""))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))

# Agents
WEB_SEARCH_ENABLED = _bool("WEB_SEARCH_ENABLED", True)
MAX_CLAIMS = int(os.getenv("MAX_CLAIMS", "3"))
EVIDENCE_PER_CLAIM = int(os.getenv("EVIDENCE_PER_CLAIM", "4"))
FETCH_TIMEOUT = float(os.getenv("FETCH_TIMEOUT", "8"))

# Serving
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
LOAD_TRANSFORMER = _bool("LOAD_TRANSFORMER", True)
LOAD_EMBED = _bool("LOAD_EMBED", True)
ONNX_THREADS = int(os.getenv("ONNX_THREADS", "2"))
API_KEY = os.getenv("VERIFACT_API_KEY", "")  # optional: protects mutating endpoints
