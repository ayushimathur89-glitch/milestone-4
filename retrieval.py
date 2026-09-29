"""Question retrieval against the Phase 3 vector store.

Two things this file exists to guarantee:

1. **The question is embedded by the same model that embedded the chunks.**
   It imports `ingest.embedder.get_model` rather than loading anything of its
   own. A second model object, or the same model with different normalisation,
   would put questions and chunks in different spaces and return confident
   nonsense without erroring. The only thing this module is allowed to call is
   the shared instance.

2. **A scheme named in the question filters the search.** The `scheme` field
   exists in the chunk metadata for exactly this. "What is the lock-in period
   for SBI ELSS?" should not be answered by a Flexicap chunk that happens to
   rank higher, because lock-in is an ELSS-specific concept with a different
   answer per scheme.

Distances come back as `1 - cosine_similarity` because the collection was
created in cosine space over L2-normalised vectors, so `store.py` documents
that conversion. `config.MIN_SIMILARITY` is a similarity floor, so it is
applied after the conversion, not to the raw distance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import config
from ingest import embedder, store

# Words that carry no retrieval signal: question scaffolding, and the scheme
# name that is already applied as a hard filter. Leaving "fund" or "sbi" in
# would make every chunk in the collection match and the score meaningless.
_STOPWORDS = frozenset("""
a an the is are was were be been being do does did doing i me my we our you
your for of in on at to from by with about as and or if then than that this
these those it its what which where when who whom how why can could should
would will shall may might must tell show give find get please latest
""".split())


def _normalise(text: str) -> str:
    """Lowercase and flatten punctuation so terms match across wordings.

    Hyphens and slashes become spaces, which is what makes the question's
    "capital-gains" match a chunk saying "capital gains", and "KIM/SID" match
    a chunk naming the KIM and the SID separately.
    """
    flat = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    return " " + " ".join(flat.split()) + " "


def content_terms(question: str, scheme: str | None = None) -> tuple[str, ...]:
    """The discriminative words of a question, with padding for matching.

    Single characters are dropped because a stray "a" or "5" matching a chunk
    proves nothing. Terms are returned space-padded so a match is a whole-word
    test rather than a substring one, which stops "load" from matching
    "download" and "sip" from matching "sips".
    """
    scheme_words = {
        word for word in _normalise(scheme).split() if word
    } if scheme else set()
    words = _normalise(question).split()
    kept = [
        w for w in words
        if len(w) > 1 and w not in _STOPWORDS and w not in scheme_words
    ]
    # Longest first so a multi-word term can be tested before its parts.
    return tuple(f" {w} " for w in sorted(set(kept), key=len, reverse=True))


def lexical_score(text: str, terms: tuple[str, ...]) -> float:
    """Fraction of the question's content words present in one chunk.

    Fraction rather than raw count, so a long chunk cannot win by containing
    more of the vocabulary. Terms that appear in nearly every chunk of a
    scheme are the ones that should not move the ranking; requiring the whole
    set to match would let "fund" and "SBI" veto an exact "exit load" hit, so
    the ratio is combined with the dense score instead of replacing it.
    """
    if not terms:
        return 0.0
    haystack = _normalise(text)
    matched = sum(1 for term in terms if term in haystack)
    return matched / len(terms)

# Canonical scheme names must match data/sources.csv exactly, because the
# value is used as a Chroma `where` filter against stored metadata.
SCHEME_ALIASES: dict[str, str] = {
    "SBI Flexicap Fund": "SBI Flexicap Fund",
    "SBI ELSS Tax Saver Fund": "SBI ELSS Tax Saver Fund",
    "SBI Large Cap Fund": "SBI Large Cap Fund",
    "SBI Small Cap Fund": "SBI Small Cap Fund",
    "SBI Balanced Advantage Fund": "SBI Balanced Advantage Fund",
}

# Distinguishing words, not a bag of parts. "small cap" appears in both
# "SBI Small Cap Fund" and the generic phrase "a small cap fund", so matching
# on parts alone filtered a generic question to one scheme and hid the other
# four from the search. Requiring "sbi" alongside the distinguishing words
# means a scheme is only detected when the question actually names it.
#
# Patterns are tried longest-first so "balanced advantage" wins over
# "balanced" and "tax saver" over "elss".
_SCHEME_HINTS: tuple[tuple[str, str], ...] = (
    ("SBI ELSS Tax Saver Fund", r"sbi\s+(?:elss|tax\s+saver)|elss\s+tax\s+saver"),
    ("SBI Balanced Advantage Fund", r"sbi\s+balanced|balanced\s+advantage"),
    ("SBI Flexicap Fund", r"sbi\s+flexicap|flexicap"),
    ("SBI Large Cap Fund", r"sbi\s+large\s+cap"),
    ("SBI Small Cap Fund", r"sbi\s+small\s+cap"),
)

_SCHEME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (canonical, re.compile(pattern, re.I))
    for canonical, pattern in sorted(
        _SCHEME_HINTS, key=lambda item: len(item[1]), reverse=True
    )
)


def detect_scheme(question: str) -> str | None:
    """Return the canonical scheme name if the question names one, else None.

    Deliberately conservative: a wrong filter is worse than no filter, because
    it silently hides the correct answer. "SBI" alone is not enough to guess a
    scheme, and neither is a bare "fund".
    """
    text = question or ""
    for canonical, pattern in _SCHEME_PATTERNS:
        if pattern.search(text):
            return canonical
    return None


@dataclass(frozen=True)
class Retrieved:
    """One retrieved chunk with its score, metadata, and provenance."""

    chunk_id: str
    scheme: str
    doc_type: str
    source_url: str
    page_title: str
    section: str
    fetched_at: str
    text: str
    similarity: float
    # Dense similarity plus the lexical boost. `similarity` stays the raw
    # topical score because the floor and the diagnostics are about whether
    # the chunk is about the right subject; `score` is the ranking key.
    score: float = 0.0
    matched_terms: tuple[str, ...] = ()

    @property
    def allowed_url(self) -> bool:
        """Whether the citation domain passes the official-source policy."""
        return config.is_allowed_url(self.source_url)

    def citation(self) -> str:
        return f"[{self.chunk_id}] {self.scheme} / {self.doc_type}"

    def preview(self, width: int = 200) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= width else flat[: width - 3] + "..."


@dataclass(frozen=True)
class SearchResult:
    query: str
    scheme_filter: str | None
    hits: tuple[Retrieved, ...]
    best_similarity: float
    terms: tuple[str, ...] = ()

    @property
    def has_usable_context(self) -> bool:
        """Whether anything retrieved clears the similarity floor.

        This is the gate that keeps an out-of-corpus question from reaching
        the LLM. A NAV question has no answer in the corpus, but chunks about
        NAV *tables* exist and can score above the floor, so the floor alone
        is a weak signal and `generator` re-checks the answer itself.
        """
        return bool(self.hits) and self.best_similarity >= config.MIN_SIMILARITY

    def above_floor(self) -> tuple[Retrieved, ...]:
        """Hits at or above `config.MIN_SIMILARITY`, best first."""
        return tuple(h for h in self.hits if h.similarity >= config.MIN_SIMILARITY)

    def any_keyword_hit(self) -> bool:
        """Whether any returned chunk literally contains a question term.

        A cheap readout for "did the exact wording survive retrieval", which is
        what separates an answered question from a missed one. The model
        declining while this is False is expected, not a defect.
        """
        return any(h.matched_terms for h in self.hits)


def _open_collection():
    return store.open_collection(store.get_client())


def retrieve(question: str, top_k: int | None = None,
             collection=None) -> SearchResult:
    """Embed the question and return the closest chunks, best match first.

    When the question names a scheme the search is filtered to that scheme.
    If a filtered search returns nothing at all the filter is dropped and the
    search is retried unfiltered, so an alias we failed to recognise degrades
    to a slightly noisier answer rather than to silence.

    A wider pool than `top_k` is fetched and then reranked on a blend of dense
    similarity and literal keyword overlap, because the dense score alone
    pushes the one chunk that states the answer out of the top 4 on questions
    phrased around an exact term like "exit load" or "capital gains".
    """
    top_k = top_k or config.TOP_K
    fetch = max(top_k, config.TOP_K_FETCH)
    collection = collection or _open_collection()
    scheme = detect_scheme(question)
    vector = embedder.embed_query(question).tolist()

    def run(where: dict | None) -> tuple:
        kwargs: dict = {
            "query_embeddings": [vector],
            "n_results": fetch,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return collection.query(**kwargs)

    response = run({"scheme": scheme} if scheme else None)
    used_filter = scheme
    if scheme and not response["ids"][0]:
        response = run(None)
        used_filter = None

    terms = content_terms(question, used_filter)

    hits: list[Retrieved] = []
    for doc, meta, distance in zip(
        response["documents"][0], response["metadatas"][0], response["distances"][0]
    ):
        similarity = 1.0 - float(distance)
        text = doc or ""
        haystack = _normalise(text)
        matched = tuple(term.strip() for term in terms if term in haystack)
        hits.append(Retrieved(
            chunk_id=meta.get("chunk_id", ""),
            scheme=meta.get("scheme", ""),
            doc_type=meta.get("doc_type", ""),
            source_url=meta.get("source_url", ""),
            page_title=meta.get("page_title", ""),
            section=meta.get("section", ""),
            fetched_at=meta.get("fetched_at", ""),
            text=text,
            similarity=similarity,
            score=similarity + config.LEXICAL_WEIGHT * lexical_score(text, terms),
            matched_terms=matched,
        ))

    # Sort on the blended score; dense similarity breaks ties so the ordering
    # stays stable and reproducible between runs.
    hits.sort(key=lambda h: (h.score, h.similarity), reverse=True)
    best = max((h.similarity for h in hits), default=0.0)
    return SearchResult(
        query=question,
        scheme_filter=used_filter,
        hits=tuple(hits[:top_k]),
        best_similarity=best,
        terms=terms,
    )
