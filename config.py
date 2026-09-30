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
# A PDF fact table flattened onto one line is window-split, and every window
# after the first would otherwise lose the column names that give its numbers
# meaning. This is how many leading words are re-seated in each such window.
CHUNK_TABLE_LEAD_WORDS = 18

# --- Retrieval ---
# Six, not four: a single figure can need two complementary chunks from the
# same document - the row stating it and the notice explaining what the columns
# mean - and at four the keyed row was crowded out by the notice's own
# introduction, which repeats the question's wording and matches more terms.
TOP_K = 6
# One document in data/sources.csv is not scheme-specific: the base-TER notice
# lists every scheme in a single table, so it is stored with this scheme value.
# A scheme filter is an exact match on that field, which makes such a document
# invisible to every question naming a scheme - yet a notice revising the base
# TER of all the equity schemes is exactly what one scheme's TER question
# needs. The value is repeated verbatim in stored chunk metadata, so it is a
# data contract rather than a literal buried in a query.
GLOBAL_SCHEME = "All schemes"
# Ranking credit for a row of that global table which names the queried scheme,
# in score units on the same scale as the cosine similarity. The keyword scorer
# strips scheme words from the question, because the filter already guarantees
# every scheme-scoped chunk is the right scheme; but a global document is a
# table of rows for every scheme, so the scheme name written in a row is the
# only thing tying it to a question.
#
# Sized above LEXICAL_WEIGHT deliberately. A keyword match is topical evidence
# - the chunk discusses expense ratios - while a table row naming exactly the
# queried scheme is precise evidence that this row is its answer, and precise
# must win. The row abbreviates the question instead of repeating it ("Existing
# Base TER (%)" against a question asking for the total expense ratio), so it
# can never collect keyword credit and must be promoted on this alone.
GLOBAL_ROW_BONUS = 0.40
# Measured, not guessed. Over 20 answerable and 20 off-topic questions, scored
# on the dense similarity because that is what this floor reads:
#   lowest answerable   0.539  "What is the benchmark index for SBI ELSS?"
#   highest off-topic   0.209  "What is the boiling point of water?"
# at 0.15 the floor separated nothing - it sat *below* every off-topic
# question measured, so junk reached the model on all of them. 0.40 keeps 0.139
# of headroom under the weakest real question and 0.191 over the worst
# off-topic one.
#
# This is a backstop, not the defence. Vocabulary in guardrails is the primary
# gate, and it is what makes the gap wide: before it was extended, off-topic
# questions reached 0.469 ("Who won the election in Maharashtra?") against
# 0.539 for the weakest answerable one, a 0.070 margin that no threshold could
# have made safe. Similarity is a blunt instrument, so the floor is set to stop
# obvious misses reaching the LLM while the classifier stops them arriving.
MIN_SIMILARITY = 0.40

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
# Pool must sit well clear of the depth reranking actually uses. The TER row
# sat at rank 25 of a 25-candidate pool, so it was included by luck; one
# unrelated document shifting it to rank 26 would have silently dropped the only
# chunk stating the figure.
TOP_K_FETCH = 60
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
