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


def _as_schemes(schemes: str | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalise a scheme argument to a tuple, treating a bare string as one."""
    if not schemes:
        return ()
    if isinstance(schemes, str):
        return (schemes,)
    return tuple(schemes)


def content_terms(question: str,
                  scheme: str | tuple[str, ...] | None = None) -> tuple[str, ...]:
    """The discriminative words of a question, with padding for matching.

    Single characters are dropped because a stray "a" or "5" matching a chunk
    proves nothing. Terms are returned space-padded so a match is a whole-word
    test rather than a substring one, which stops "load" from matching
    "download" and "sip" from matching "sips".

    Scheme words are stripped for every scheme named, not just the first: on a
    two-scheme question "flexicap" and "small" are as uninformative as "sbi" is
    on a one-scheme question, and leaving them in would match every chunk of
    both schemes and make the lexical score meaningless.
    """
    scheme_words = {
        word for name in _as_schemes(scheme) for word in _normalise(name).split()
        if word
    }
    words = _normalise(question).split()
    kept = [
        w for w in words
        if len(w) > 1 and w not in _STOPWORDS and w not in scheme_words
    ]
    # Longest first so a multi-word term can be tested before its parts.
    return tuple(f" {w} " for w in sorted(set(kept), key=len, reverse=True))


def mentions_scheme(text: str, scheme: str | tuple[str, ...] | None) -> bool:
    """Whether a chunk's own text names any scheme the question asked about.

    Compares the whole scheme name case-insensitively, which is enough because
    the canonical value and the source's rendering differ only in capitalisation
    and spacing: the TER notice writes "sbi large cap fund direct" where the
    canonical value is "SBI Large Cap Fund". A scheme name alone is a weak
    signal, so callers pair it with a check that the document is on the
    question's subject before acting on it.
    """
    haystack = _normalise(text)
    return any(_normalise(name) in haystack for name in _as_schemes(scheme))


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
    ("SBI Flexicap Fund", r"sbi\s+flexi[\s-]?cap|flexi[\s-]?cap"),
    ("SBI Large Cap Fund", r"sbi\s+large\s+cap"),
    ("SBI Small Cap Fund", r"sbi\s+small\s+cap"),
)

_SCHEME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (canonical, re.compile(pattern, re.I))
    for canonical, pattern in sorted(
        _SCHEME_HINTS, key=lambda item: len(item[1]), reverse=True
    )
)


def detect_schemes(question: str) -> tuple[str, ...]:
    """Return every canonical scheme name the question names, in that order.

    A question may name more than one scheme - "the minimum SIP for SBI
    Flexicap Fund and SBI Small Cap Fund?" - and returning only the first match
    filters the search to that one scheme, which makes the other scheme's
    chunks unreachable. The bot is then handed a one-scheme context for a
    two-scheme question, cannot answer it, and declines while citing a link as
    though it had checked. Both schemes' minimum SIP is in the corpus, so that
    decline is our bug and not a source gap.

    Order is the order the patterns match in, which is longest-pattern-first,
    not the order the names appear in the question. Callers treat the result as
    an unordered set; the order only has to be stable between runs.

    Same conservatism as the single-scheme version: "SBI" alone is not enough,
    and neither is a bare "fund".
    """
    text = question or ""
    return tuple(canonical for canonical, pattern in _SCHEME_PATTERNS
                 if pattern.search(text))


def detect_scheme(question: str) -> str | None:
    """Return the first canonical scheme name in the question, else None.

    Kept for callers that genuinely want one name - `memory.referenced_scheme`
    resolving a pronoun, or the CLI's /filter readout. For retrieval use
    `detect_schemes`, because a question naming two schemes must reach both.
    """
    schemes = detect_schemes(question)
    return schemes[0] if schemes else None


# The same five schemes, matched loosely enough to catch a bare fragment.
# `detect_schemes` requires "sbi" beside the distinguishing words because "small
# cap" also occurs in the generic phrase "a small cap fund", and filtering a
# generic question to one scheme would hide the other four. That reasoning is
# right for a Chroma `where` clause and wrong for telling the user which scheme
# their fragment probably means, so the two are separate tables and only this
# one is permissive.
_SCHEME_FRAGMENTS: tuple[tuple[str, str], ...] = (
    ("SBI ELSS Tax Saver Fund", r"\belss\b|\btax\s+saver\b"),
    ("SBI Balanced Advantage Fund", r"\bbalanced\s+advantage\b"),
    ("SBI Flexicap Fund", r"flexi[\s-]?cap"),
    ("SBI Large Cap Fund", r"\blarge\s+cap\b"),
    ("SBI Small Cap Fund", r"\bsmall\s+cap\b"),
)

_FRAGMENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (canonical, re.compile(pattern, re.I))
    for canonical, pattern in sorted(
        _SCHEME_FRAGMENTS, key=lambda item: len(item[1]), reverse=True
    )
)


def probable_scheme(question: str) -> str | None:
    """The scheme a fragment most likely means, or None.

    **Never use this as a retrieval filter.** It will read "a small cap fund"
    as SBI Small Cap Fund and, worse, as a statement about that one scheme. It
    exists so `guardrails.clarifying_answer` can say "Did you mean SBI Small Cap
    Fund?" and get a follow-up the app can actually answer. Use `detect_schemes`
    to decide what to search.
    """
    text = question or ""
    for canonical, pattern in _FRAGMENT_PATTERNS:
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
    names_scheme: bool = False

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
    # The schemes the search was scoped to, empty when unfiltered. A tuple and
    # not a single name because a question may name two, and `scheme_label`
    # exists for the display: the CLI, the pipeline report and the tests all
    # print this, and a bare tuple would read as a Python repr in a report
    # meant for a person.
    scheme_filter: tuple[str, ...]
    hits: tuple[Retrieved, ...]
    best_similarity: float
    terms: tuple[str, ...] = ()

    @property
    def scheme_label(self) -> str:
        """Human-readable scope: one name, several joined, or 'none'."""
        if not self.scheme_filter:
            return "none"
        if len(self.scheme_filter) == 1:
            return self.scheme_filter[0]
        return " + ".join(self.scheme_filter)

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


def scheme_filter(schemes: str | tuple[str, ...] | None) -> dict | None:
    """Chroma `where` clause for one or more named schemes, or None for none.

    Each scheme is a disjunct, together with `config.GLOBAL_SCHEME`, rather than
    a bare equality. This is the difference between answering "what is the total
    expense ratio of SBI Large Cap Fund?" from the TER notice that revises the
    base TER of every equity scheme, and answering it from the scheme's SID,
    which only states the Regulation 52(6)(c) *ceiling* and never the scheme's
    actual rate. The ceiling is a real number, so the mistake is silent: the
    answer looks confident and is wrong.

    A single scheme becomes a two-branch disjunction; several schemes become one
    branch each, and the global document is added exactly once, because
    repeating it would only make the clause longer. A repeated `{"$or": [{"scheme":
    X}]}` is a different query to Chroma, not a no-op, so the de-duplication is
    load-bearing rather than cosmetic.

    Verified against chromadb 1.5.9, which accepts `$or` at the top level of
    `where`. `$in` on the scalar field works too and reads slightly cleaner, but
    the list form keeps the alternatives visible as documents.
    """
    if not schemes:
        return None
    if isinstance(schemes, str):
        schemes = (schemes,)
    names = list(dict.fromkeys(schemes))
    if config.GLOBAL_SCHEME not in names:
        names.append(config.GLOBAL_SCHEME)
    return {"$or": [{"scheme": name} for name in names]}


def _query(collection, vector: list[float], where: dict | None,
           fetch: int) -> dict:
    """One Chroma query, optionally scoped by a `where` clause."""
    kwargs: dict = {
        "query_embeddings": [vector],
        "n_results": fetch,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


def _merge(responses: list[dict]) -> list[tuple[str, str, float]]:
    """Flatten several query responses into unique (doc, meta, distance).

    De-duplicated by `chunk_id`: a question naming two schemes queries each with
    `config.GLOBAL_SCHEME` included, so the global TER notice comes back in both
    results. Keeping both copies would let one document occupy two of the six
    context slots.
    """
    seen: set[str] = set()
    rows: list[tuple[str, str, float]] = []
    for response in responses:
        for doc, meta, distance in zip(
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            chunk_id = meta.get("chunk_id", "")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rows.append((doc or "", meta, float(distance)))
    return rows


def _scheme_windows(collection, vector: list[float],
                    schemes: tuple[str, ...], fetch: int) -> list[dict]:
    """One query window per scheme, plus one for the global documents.

    A single window spanning the scheme and `config.GLOBAL_SCHEME` is not
    equivalent, and for the same reason the multi-scheme path above is not:
    `fetch` is a fixed budget, and a scheme with hundreds of chunks of its own
    takes all of it. Measured on "How do I download my capital-gains statement
    for SBI ELSS?", the ELSS SID's 567 chunks fill the 60-candidate window
    between them and chunk `all-schemes-2695` - which is the sentence naming
    the Capital Gains Statement and how to get it - is never scored at all. The
    lexical bonus that would have ranked it first is applied *after* the window
    is cut, so a document that literally contains the question's terms can be
    excluded for not being dense enough, which inverts the design: the pool cut
    is a dense-similarity decision and the ranking is deliberately not.

    Giving the global documents their own window costs one extra query and makes
    every non-scheme-specific page reachable from any scheme question, which is
    what `config.GLOBAL_SCHEME` is for. Global chunks still have to win their
    slots on score, so this widens the candidate set and nothing else.
    """
    windows = [_query(collection, vector, scheme_filter(name), fetch)
               for name in schemes]
    windows.append(
        _query(collection, vector, {"scheme": config.GLOBAL_SCHEME}, fetch)
    )
    return windows


def retrieve(question: str, top_k: int | None = None,
             collection=None) -> SearchResult:
    """Embed the question and return the closest chunks, best match first.

    When the question names a scheme the search is filtered to that scheme *and*
    to `config.GLOBAL_SCHEME`, so a document that revises a figure for every
    scheme is reachable from a question about one scheme. If a filtered search
    returns nothing at all the filter is dropped and the search is retried
    unfiltered, so an alias we failed to recognise degrades to a slightly noisier
    answer rather than to silence.

    **A question naming several schemes is queried once per scheme, not once for
    all of them.** One shared pool is not equivalent: `config.TOP_K_FETCH` is a
    fixed 60 chunks, so the scheme with the most text takes the whole window and
    the other scheme's answer never enters the candidate set at all. Measured on
    "the exit load of SBI Flexicap Fund and SBI Small Cap Fund", Small Cap's
    load-structure chunk is present in the Small Cap-only pool and absent from
    the combined one, so the model receives Flexicap's exit load and nothing for
    Small Cap, and then declines half a question it can half-answer. One query
    per scheme gives each scheme its own window, and the merged result is
    reranked as usual so the final order still reflects relevance rather than
    which scheme was queried first.

    A wider pool than `top_k` is fetched and then reranked on a blend of dense
    similarity and literal keyword overlap, because the dense score alone
    pushes the one chunk that states the answer out of the top 4 on questions
    phrased around an exact term like "exit load" or "capital gains".
    """
    top_k = top_k or config.TOP_K
    fetch = max(top_k, config.TOP_K_FETCH)
    collection = collection or _open_collection()
    schemes = detect_schemes(question)
    vector = embedder.embed_query(question).tolist()

    rows: list[tuple[str, str, float]] = []
    if not schemes:
        rows = _merge([_query(collection, vector, None, fetch)])
        used_filter: tuple[str, ...] = ()
    else:
        # One window per named scheme plus one for the global documents, so no
        # single scheme's chunk count can starve the others. See
        # `_scheme_windows` for the measurement that forced this.
        responses = _scheme_windows(collection, vector, schemes, fetch)
        rows = _merge(responses)
        used_filter = schemes
        if not rows:
            rows = _merge([_query(collection, vector, None, fetch)])
            used_filter = ()

    terms = content_terms(question, used_filter)

    hits: list[Retrieved] = []
    for doc, meta, distance in rows:
        similarity = 1.0 - distance
        text = doc or ""
        haystack = _normalise(text)
        matched = tuple(term.strip() for term in terms if term in haystack)
        names_scheme = mentions_scheme(text, used_filter)
        score = similarity + config.LEXICAL_WEIGHT * lexical_score(text, terms)
        # A row in a global document that names the queried scheme is keyed to
        # that scheme: the scheme name is the only column distinguishing it from
        # the other schemes listed beside it. A scheme-scoped chunk is merely
        # about the scheme and says nothing about being the answer to this
        # question, so the credit is reserved for the global row. Without it a
        # row like "SBI Large Cap Fund Direct 0.65 0.66 20.03.2026" loses to
        # notice prose that merely repeats the question's words, because the
        # row's own header abbreviates them ("Base TER") and matches no term.
        #
        # Naming the scheme is necessary but not sufficient: every scheme the
        # notice lists is named in some row, so on its own this promoted a
        # base-TER row to first place for an exit-load question and the answer
        # then cited the TER notice instead of the SID. The document must also
        # be on the question's subject, judged by its own URL and title, so the
        # credit is spent only where the question and the document agree.
        on_subject = any(
            term.strip() and term in _normalise(
                f"{meta.get('source_url', '')} {meta.get('page_title', '')}"
            )
            for term in terms
        )
        if (names_scheme and on_subject
                and meta.get("scheme", "") == config.GLOBAL_SCHEME):
            score += config.GLOBAL_ROW_BONUS
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
            score=score,
            matched_terms=matched,
            names_scheme=names_scheme,
        ))

    # Sort on the blended score; dense similarity breaks ties so the ordering
    # stays stable and reproducible between runs.
    hits.sort(key=lambda h: (h.score, h.similarity), reverse=True)
    best = max((h.similarity for h in hits), default=0.0)
    return SearchResult(
        query=question,
        scheme_filter=used_filter,
        hits=tuple(_cover_schemes(hits, used_filter, top_k)),
        best_similarity=best,
        terms=terms,
    )


def _cover_schemes(hits: list[Retrieved], schemes: tuple[str, ...],
                   top_k: int) -> list[Retrieved]:
    """Give every queried scheme its own budget of chunks.

    Fetching a window per scheme is not the same as *returning* from each, and
    the difference decides whether a two-scheme question gets answered. On "the
    exit load of SBI Flexicap Fund and SBI Small Cap Fund", Small Cap's
    load-structure chunk ranks 7th overall because Flexicap simply has more text
    about exit loads. A fixed top-6 hands the model six chunks and no Small Cap
    figure, and it declines half a question it could half-answer.

    Reserving slots per scheme was tried first and rejected: both schemes were
    already represented in the top 6, so nothing was reserved and the chunk that
    actually states the figure still did not make it. The problem is not that a
    scheme is missing, it is that six chunks cannot hold two schemes' figures.
    So the budget scales with the number of schemes asked about, and each scheme
    keeps a share of it - reserving its best chunks - with the remainder left to
    rank normally.

    The global TER notice never consumes a reserved slot: it states no scheme's
    rate, so on a two-scheme TER question it would displace the figures the user
    asked for.
    """
    if not schemes or len(schemes) < 2:
        return hits[:top_k]

    chosen = list(hits[:top_k])
    seen = {h.chunk_id for h in chosen}
    for name in schemes:
        for hit in hits:
            if hit.scheme == name and hit.chunk_id not in seen:
                chosen.append(hit)
                seen.add(hit.chunk_id)
                break
    return chosen
