"""Central configuration for the Mutual Fund FAQs RAG chatbot.

Every tunable constant and path lives here so the ingestion and query paths
cannot drift apart. See Docs/architecture.md and Docs/implementation.md.
"""

import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"
SAMPLES_DIR = PROJECT_ROOT / "samples"

SOURCES_CSV = DATA_DIR / "sources.csv"
INGEST_MANIFEST_CSV = DATA_DIR / "ingest_manifest.csv"
CHUNKS_TXT = DATA_DIR / "chunks.txt"
EMBEDDINGS_PREVIEW_TXT = DATA_DIR / "embeddings_preview.txt"

# --- Secrets (never commit .env) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def mask_secret(value: str) -> str:
    """Return a masked form of a secret, safe to print."""
    if not value:
        return "<not set>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"


# --- Models ---
# Runs locally, needs no API key, produces 384-dim vectors.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
# The embedding model truncates beyond this many tokens. Chunking targets
# must stay under it or chunks are silently cut before they are stored.
MAX_EMBED_TOKENS = 256

# Groq model id. Groq retires ids over time - confirm this against the model
# list in the Groq console before relying on it, and change it only here.
# `llama-3.1-8b-instant` was retired by Groq; the only surviving "llama" ids
# are the prompt-guard classifiers, which are not chat models.
LLM_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 300

# --- Chunking (see architecture.md 2.3) ---
CHUNK_MAX_WORDS = 160
CHUNK_OVERLAP_WORDS = 30

# --- Retrieval ---
TOP_K = 4
MIN_SIMILARITY = 0.15

# Candidates pulled from the vector store before reranking, and how much a
# literal keyword match can promote a chunk. Both exist because dense
# similarity alone ranks fact-bearing chunks out of the top 4: asking for the
# "exit load" of SBI Large Cap Fund returned a returns table, cheque-payment
# text and risk factors, while the chunk that actually states the load
# structure sat at rank 2 for a near-duplicate question. Embeddings match
# topic, not the exact term the user typed. Widening the candidate pool and
# boosting chunks that literally contain the question's content words fixes
# the recall; the floor above still runs on the dense score, because "is this
# the right document at all" is a topical judgement.
TOP_K_FETCH = 25
LEXICAL_WEIGHT = 0.35

# --- Answer contract ---
MAX_ANSWER_SENTENCES = 3
LAST_UPDATED_PREFIX = "Last updated from sources:"
DISCLAIMER = "Facts-only. No investment advice."

# --- Source policy ---
# Enforced at the domain level, not by pinning exact URLs: the campaign pages
# in the brief are landing pages, and figures like exit load often live on a
# factsheet or SID instead. Official URLs may be added; non-official ones
# may not.
ALLOWED_DOMAINS = (
    "sbimf.com",
    "amfiindia.com",
    "sebi.gov.in",
    "sebi.co.in",
)

# --- Guards ---


def is_allowed_url(url: str) -> bool:
    """Return True only for http(s) URLs on an official domain.

    Matching is on the domain suffix, not an exact host, because the official
    pages are served from www.sbimf.com while the allow-list names sbimf.com.
    The leading-dot check keeps lookalike hosts such as "evilsbimf.com" from
    passing.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.netloc or "").lower().split(":")[0]
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


COLLECTION_NAME = "mf_faqs"


# --- Self-check ---

if __name__ == "__main__":
    print("Mutual Fund FAQs RAG Chatbot - configuration")
    print(f"  GROQ_API_KEY     : {mask_secret(GROQ_API_KEY)}")
    print(f"  LLM model        : {LLM_MODEL}")
    print(f"  Embedding model  : {EMBEDDING_MODEL} ({EMBEDDING_DIM}-dim)")
    print(f"  Chunking         : {CHUNK_MAX_WORDS} words / {CHUNK_OVERLAP_WORDS} overlap")
    print(f"  Top-K            : {TOP_K}")
    print(f"  Allowed domains  : {', '.join(ALLOWED_DOMAINS)}")
    print(f"  Chroma path      : {CHROMA_DIR}")
