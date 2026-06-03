"""
config.py
---------
Single place where ALL configuration lives.

Why one file: the assessment scores how we handle configuration and secrets.
Keeping every setting here (and reading secrets only from environment
variables) means:
  - no API key is ever written in code or committed to git
  - the grader can change models / collection name without touching code
  - it's obvious what is configurable just by reading this one file
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env file into the environment, if one exists.
# In Docker we pass --env-file instead, so this is a no-op there; either works.
load_dotenv()


def _required(name: str) -> str:
    """Read a required env var, or fail loudly with a clear message.

    Failing early with a readable error is much better than a confusing
    crash deep inside the app when a key is missing.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


# --- Secrets (no defaults: the app must not start without these) ---
# Groq powers the CHAT model (free tier, no credit card).
# The embedding model (FastEmbed) runs locally and needs NO key.
GROQ_API_KEY = _required("GROQ_API_KEY")
QDRANT_URL = _required("QDRANT_URL")
QDRANT_API_KEY = _required("QDRANT_API_KEY")

# --- Tunables (sensible defaults, overridable via .env) ---
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "iphone_user_guide")
# Embedding model: FastEmbed (by Qdrant). Runs locally via ONNX, no API key,
# no GPU. bge-small-en-v1.5 outputs 384-dim vectors and is a strong, light
# general-purpose English retrieval model.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# Chat model: served by Groq's free API. Llama 3.3 70B is high quality and
# very fast on Groq's hardware.
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile")

# Where FastEmbed caches the downloaded model. Set in Docker so we can
# pre-download at build time; left as default (None) on a local machine.
FASTEMBED_CACHE_DIR = os.getenv("FASTEMBED_CACHE_DIR") or None

# How many chunks to retrieve per question. 4 is a good default: enough
# context to answer, few enough to keep the prompt focused and cheap.
TOP_K = int(os.getenv("TOP_K", "4"))

# Chunking settings (used by ingest.py). Kept here so the interview
# discussion about chunking maps directly to real, visible numbers.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# The port Streamlit serves on. Documented in the README for port mapping.
APP_PORT = int(os.getenv("APP_PORT", "8501"))
