"""A content-addressed cache of Groq responses.

**Why this exists.** Groq's free tier is capped at 200,000 tokens *per day*, and
the verification suites make 10-30 model calls per run. Two things follow, and
both are painful:

- Re-running a suite to check one behaviour spends tokens re-deriving the other
  twenty-nine answers.
- The cap is daily, not per-second, so a suite that ran successfully this
  morning cannot run this evening. There is no amount of waiting that fixes it
  within a working day, and no backoff that turns a 200,000/200,000 into a
  pass.

So the response is cached, and the key is the request itself. Two calls with a
byte-identical request - same model, temperature, token cap, system prompt and
user message - are the same call, and getting the same answer twice is not
something worth paying for twice.

**Why a stale hit is impossible here.** The key is a SHA-256 over the full
request, not over the question. Changing the system prompt, the retrieved
context, the chunk order, the model id or the temperature all change the key, so
a cache hit can only ever mean "this exact request already produced this exact
text". That matters more than usual in a RAG pipeline: the question is the same
but the *context* is what actually determines the answer, and a cache keyed on
the question alone would happily return an answer derived from a corpus that has
since been re-ingested. This is the opposite failure from the one a cache is
normally blamed for, and it is the one that would matter.

**What is and is not cached.** Only the model's raw completion text, and only
after `guardrails.verify_answer` has not run yet - the post-check is applied
fresh on every call, so a repair is re-derived rather than frozen. Nothing else
in `generator.ask` is cached: retrieval, classification, memory and the
guardrails are all cheap and all local, and caching them would only add ways
for the pipeline to be wrong.

**Turning it off.** `LLM_CACHE=off` in the environment, or `llm_cache.clear()`.
The suites print which mode they ran in, because a suite that silently answered
from disk has not re-verified anything and should not be allowed to look like
it has.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import config

CACHE_DIR = config.DATA_DIR / "llm_cache"

# "off" disables. Anything else (including unset) enables, so the default is
# the useful behaviour and the escape hatch is one env var.
_ENABLED = os.getenv("LLM_CACHE", "on").strip().lower() not in ("off", "0", "false")


def enabled() -> bool:
    """Whether cached responses may be read. Writing follows the same switch."""
    return _ENABLED and CACHE_DIR.exists()


def _key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def request_key(*, model: str, system: str, user: str) -> str:
    """The cache key for one completion request."""
    return _key({
        "v": 1,
        "model": model,
        "system": system,
        "user": user,
        # Kept in the key even though the caller passes them separately, so
        # that changing either without the other cannot collide.
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
    })


def _path(key: str) -> Path:
    # Sharded so no single directory ends up with thousands of entries.
    return CACHE_DIR / key[:2] / f"{key}.json"


def load(key: str) -> str | None:
    """Return the cached completion text for `key`, or None on a miss.

    A corrupt or partial file is treated as a miss rather than raising. A cache
    is an optimisation; letting a bad file break a request would make it a
    liability.
    """
    if not enabled():
        return None
    path = _path(key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    text = payload.get("text")
    return text if isinstance(text, str) and text.strip() else None


def store(key: str, text: str, model: str = "") -> None:
    """Record a completion. Silently does nothing if the cache is unwritable."""
    if not _ENABLED or not text.strip():
        return
    path = _path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"text": text, "model": model}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        # A read-only checkout should still be able to ask questions.
        pass


def describe() -> str:
    """One line for a test header: which mode, and how much is banked."""
    if not _ENABLED:
        return "cache off (LLM_CACHE=off)"
    count = sum(1 for _ in CACHE_DIR.glob("*/*.json")) if CACHE_DIR.exists() else 0
    return f"cache on, {count} response(s) banked"


def clear() -> None:
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
