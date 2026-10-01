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
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# Agents
WEB_SEARCH_ENABLED = _bool("WEB_SEARCH_ENABLED", True)
MAX_CLAIMS = int(os.getenv("MAX_CLAIMS", "3"))
EVIDENCE_PER_CLAIM = int(os.getenv("EVIDENCE_PER_CLAIM", "4"))
FETCH_TIMEOUT = float(os.getenv("FETCH_TIMEOUT", "12"))

# Serving
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
LOAD_TRANSFORMER = _bool("LOAD_TRANSFORMER", True)
LOAD_EMBED = _bool("LOAD_EMBED", True)
API_KEY = os.getenv("VERIFACT_API_KEY", "")  # optional: protects mutating endpoints
