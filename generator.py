"""Answer generation via Groq, wrapped in the Phase 4 contract.

The prompt states the answer rules, but the prompt is not the enforcement.
Four layers have to agree before a user sees anything:

1. `memory.resolve_followup` rewrites a follow-up into a standalone question
   before anything else, because `guardrails.classify` and `retrieval.retrieve`
   both read the scheme out of the question, and "what about its fees?" names
   none. Resolving first means the routing decision is made on the question
   that will actually be searched.
2. `guardrails.classify` refuses advisory, performance, PII and off-topic
   questions *before* retrieval, so a refused query never reaches this module
   and never becomes a Groq call. A rewrite cannot make a refused question
   allowed: the rules only substitute a scheme name, so "should I buy it?"
   becomes "should I buy SBI Flexicap Fund?" and is still a recommendation
   request.
3. `guardrails.verify_answer` checks the generated text and repairs it.
4. The "Last updated" line is appended here from the cited chunk's
   `fetched_at`, never taken from the model. The prompt tells the model not to
   state a date, and the post-check catches it if it does anyway.

On the date specifically: the model is given context that already contains
`fetched_at` values, so it *could* copy one. Saying "do not state a date" is
therefore a real instruction and not a formality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import config
import guardrails
import llm_cache
import memory
import retrieval
from ingest import embedder

_CLIENT = None

# Token-budget guard. A 3-sentence answer is small, but the context is not, and
# an overflow would surface as an API error rather than a wrong answer, so the
# limit is stated rather than discovered.
MAX_CONTEXT_CHUNKS = 6
MAX_CONTEXT_CHARS = 9000


def get_client():
    """Return the process-wide Groq client, or None when no key is set."""
    global _CLIENT
    if _CLIENT is None:
        from groq import Groq

        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


MSG_MODEL_UNAVAILABLE = (
    "The assistant model is temporarily unavailable, so no answer could be "
    "generated. This is a provider limit, not a gap in the sources - the "
    "figures are in the official documents, and the same question will answer "
    "once the limit resets."
)


def _describe_api_error(exc: Exception) -> str:
    """A short, honest description of why the model could not be called.

    The distinction that matters downstream is rate limiting versus everything
    else, because a daily token cap is a wall with a timer on it and a 500 is
    not. Both are reported the same way to the user, but only the first is
    worth retrying later rather than debugging.
    """
    text = str(exc)
    if "rate_limit" in text or "429" in text or "TPD" in text:
        return "the free tier's daily token limit is exhausted"
    if "401" in text or "unauthorized" in text.lower():
        return "the API key was rejected"
    return f"the request failed ({type(exc).__name__})"


# Notes from the most recent `complete` call, for `ask` to fold into the
# Answer. Module-level because `complete` sits below `ask` and the alternative
# was threading a fourth return value through every caller.
_UNGROUNDED_NOTES: list[str] = []


def take_ungrounded_notes() -> tuple[str, ...]:
    """Drain the notes from the last `complete` call."""
    notes = tuple(_UNGROUNDED_NOTES)
    _UNGROUNDED_NOTES.clear()
    return notes


def complete(system: str, user: str, context: str = "") -> tuple[str, str, str]:
    """One model call, cached. Returns (text, model_name, error).

    `error` is empty on success. It is returned rather than raised because a
    provider failure is not a bug in this pipeline, and the caller has a
    correct thing to do with it: say the model is unavailable. Raising would
    either crash the app or force every caller to catch, and the CLI, the web
    app and three test suites all call this.

    The cache sits here, at the single place a completion is produced, rather
    than at the call sites, so no path can bypass it and no path can
    double-count it.

    `context` is the source block the answer is meant to come from, and it
    gates the *write*. An answer whose figures are absent from the context is
    still returned, because the post-check and the user are better served by a
    visible wrong answer that a test can catch than by a silent retry - but it
    is not banked. Temperature 0.0 does not make Groq deterministic, so
    without this one unlucky sample in five becomes a permanent wrong answer
    that no later run can correct. See `guardrails.ungrounded_figures`.
    """
    key = llm_cache.request_key(model=config.LLM_MODEL, system=system, user=user)

    cached = llm_cache.load(key)
    if cached is not None:
        return cached, config.LLM_MODEL, ""

    try:
        response = get_client().chat.completions.create(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as exc:  # noqa: BLE001 - any provider failure, one response
        return "", "", _describe_api_error(exc)

    text = (response.choices[0].message.content or "").strip()
    model = getattr(response, "model", config.LLM_MODEL) or config.LLM_MODEL
    if not text:
        return "", model, "the model returned an empty response"

    ungrounded = guardrails.ungrounded_figures(text, context) if context else ()
    if ungrounded:
        # Reported rather than swallowed: the caller surfaces it as a problem
        # on the Answer, so a suite fails on it instead of a human noticing
        # later. The caller's `problems` tuple is the pipeline's only channel
        # for "this happened and it was not fixed".
        _UNGROUNDED_NOTES.append(
            f"model stated {list(ungrounded)}, absent from the retrieved "
            f"context; response not cached"
        )
    else:
        llm_cache.store(key, text, model)
    return text, model, ""


def system_prompt() -> str:
    return (
        "You answer questions about SBI mutual fund schemes using ONLY the "
        "supplied source context.\n\n"
        "Rules, in priority order:\n"
        "1. Use only the context provided. If it does not contain the answer, "
        "say plainly that the information is not in the sources. Never guess, "
        "estimate, or use general knowledge about funds.\n"
        "2. Do not give investment advice, recommendations, or opinions. Do not "
        "tell the user whether to buy, sell, hold, or choose anything. Report "
        "facts only.\n"
        "3. Do not state, mention, or guess any date, and do not write a "
        "'last updated' or 'as of' line. The application appends the correct "
        "date itself.\n"
        "4. At most 3 sentences.\n"
        "5. Do not add a citation link. The application appends the single "
        "source link itself. The one exception: if the question asks WHERE a "
        "document can be found or downloaded, and the context names the "
        "official page, you may state that URL exactly as the context writes "
        "it, and it becomes the citation. Never construct or guess a URL.\n"
        "6. Do not calculate or state returns, growth, or projections.\n"
        "7. Copy figures such as expense ratio, exit load and minimum SIP "
        "exactly as written in the context. Do not round or convert them.\n\n"
        "If the context does not answer the question, reply with exactly: "
        "That information is not in the official sources I have."
    )


def build_context(hits) -> str:
    """Render retrieved chunks as a numbered source block.

    Only chunks that clear `config.MIN_SIMILARITY` are included. A weak match
    handed to the model is worse than no match, because the model will try to
    build an answer from it.
    """
    usable = [h for h in hits if h.similarity >= config.MIN_SIMILARITY]
    if not usable:
        return ""

    kept: list[retrieval.Retrieved] = []
    used = 0
    for hit in usable:
        block = len(hit.text) + 200
        if kept and used + block > MAX_CONTEXT_CHARS:
            break
        kept.append(hit)
        used += block
    return "\n\n".join(
        f"[{i}] scheme={h.scheme} | doc={h.doc_type} | section={h.section or '-'}\n"
        f"source={h.source_url}\n"
        f"fetched_at={h.fetched_at}\n"
        f"text: {h.text}"
        for i, h in enumerate(kept, start=1)
    )


def _pick_citation(hits):
    """Choose the chunk to cite and the source link to show.

    The best-scoring chunk that passes the official-domain policy. Falling
    back through the ranked list rather than returning nothing matters: a
    slightly weaker official citation beats a strong one we are not allowed
    to link.
    """
    for hit in hits:
        if hit.allowed_url:
            return hit
    return None


@dataclass
class Answer:
    question: str
    category: str
    answer: str
    used_llm: bool
    refused: bool
    source_url: str = ""
    fetched_at: str = ""
    chunk_id: str = ""
    # Every scheme the search was scoped to. A tuple because a question can name
    # two, and scoping to only the first is what made a two-scheme question
    # unanswerable; `scheme_label` is the display form.
    scheme_filter: tuple[str, ...] = ()
    hits: tuple = field(default_factory=tuple)
    above_floor: tuple = field(default_factory=tuple)
    problems: tuple[str, ...] = ()
    repairs: tuple[str, ...] = ()
    # The model produced a contract-compliant refusal because the context did
    # not contain the answer. Nothing is wrong with the text; the retrieval
    # fell short. Tracked apart from `refused` so the two are never confused.
    declined: bool = False
    keyword_hit: bool = False
    elapsed: float = 0.0
    model: str = ""
    # The question as typed versus the question that was actually searched.
    # Both are kept because a rewritten follow-up is a guess about what the
    # user meant, and the answer has to be auditable against the real wording.
    resolved_question: str = ""
    inherited_scheme: str | None = None
    rewrite_notes: str = ""
    remembered: bool = False
    # Set when the model could not be called at all - a rate limit, a rejected
    # key, a provider error. Empty means nothing went wrong upstream. This is
    # the flag that keeps an infrastructure failure from being reported as a
    # fact about the corpus, so it is tracked rather than inferred from
    # `used_llm`, which is also False for a correct guardrail refusal.
    llm_error: str = ""

    @property
    def grounded(self) -> bool:
        """Whether the user got an actual answer, not a message about nothing.

        `not refused` alone is not enough: a decline passes the post-check
        cleanly, so without this the pipeline would report a question as
        answered when the model said it had nothing to go on.
        """
        return self.used_llm and not self.refused and not self.declined

    @property
    def unavailable(self) -> bool:
        """Whether the pipeline was healthy and the model simply was not there.

        A fourth outcome alongside answered, declined and refused. Reporting it
        as any of the other three would be a lie about where the failure
        happened, and a suite that could not tell them apart would be able to
        pass with the provider down.
        """
        return bool(self.llm_error)

    @property
    def scheme_label(self) -> str:
        """The retrieval scope as a person reads it, for the CLI and reports."""
        if not self.scheme_filter:
            return "none"
        if len(self.scheme_filter) == 1:
            return self.scheme_filter[0]
        return " + ".join(self.scheme_filter)


def ask(question: str, show_context: bool = False, collection=None,
        conversation: memory.Conversation | None = None) -> Answer:
    """Full pipeline: resolve, classify, retrieve, generate, verify.

    Returns an `Answer` describing what happened, so the CLI can show the
    retrieved chunks and the caller can tell a refusal from a real answer
    without parsing prose.

    `conversation` is optional, and passing None keeps the old single-turn
    behaviour: nothing is remembered and no rewriting happens. When it is
    supplied, the resolved question and the answer are appended to it.
    """
    started = time.perf_counter()

    # Before classification, not after. Routing and retrieval both read the
    # scheme out of the question, so a follow-up has to name one before either
    # of them can be right.
    rewrite = memory.resolve_followup(question, conversation)
    search_question = rewrite.resolved

    def finish(*, pii: bool = False, **fields) -> Answer:
        """Build the Answer and record the exchange, in one place.

        A PII refusal is never recorded. The account number in that question is
        exactly what must not sit in a buffer a later turn can quote back, and
        deciding that at the call site would be one more thing to forget.
        Every other refusal is stored, because "is it taxable?" being
        remembered is what lets the next turn resolve "and its exit load?".

        A model-unavailable turn is recorded for the same reason as a guardrail
        refusal: the user really did ask that question, and "and its exit load?"
        should still resolve after a rate-limited turn. What is not recorded is
        the placeholder text as an *answer*, because the rewriter prefers user
        turns but does fall back to answers, and a buffer full of "the model is
        temporarily unavailable" would quietly poison every later rewrite.
        """
        remembered = False
        if conversation is not None and not pii:
            stored = conversation.add("user", question)
            if fields.get("llm_error"):
                # Unbalanced on purpose: one user turn, no answer. The rewriter
                # reads roles by position, not in pairs, so a lone user turn
                # resolves correctly and there is nothing to misquote.
                remembered = stored
            else:
                stored &= conversation.add(
                    "assistant", fields.get("answer", "")
                )
                remembered = stored
        return Answer(
            question=question,
            resolved_question=search_question,
            inherited_scheme=rewrite.scheme,
            rewrite_notes=rewrite.rewrite_notes,
            remembered=remembered,
            **fields,
        )

    decision = guardrails.classify(search_question)
    if decision.refused:
        # No retrieval and no Groq call: the guardrail is meant to hold
        # structurally, and spec verification step 4 checks it by request log.
        return finish(
            pii=decision.category == "pii",
            category=decision.category,
            answer=decision.message,
            used_llm=False,
            refused=True,
            problems=(decision.reason,) if decision.reason else (),
            elapsed=time.perf_counter() - started,
        )

    # After the classifier, before retrieval. `classify` has already established
    # that this is about mutual funds; this asks the separate question of whether
    # these documents hold anything it could answer. Retrieval cannot answer it
    # either way - a fragment with no question in it retrieves confidently and
    # then declines - so asking first is what turns a dead end into a question.
    if not guardrails.names_supported_topic(search_question):
        scheme = retrieval.probable_scheme(search_question)
        return finish(
            category="underspecified",
            answer=guardrails.clarifying_answer(scheme),
            used_llm=False,
            refused=True,
            problems=(
                "names no topic the corpus publishes"
                + (f"; assumed {scheme} for the reply" if scheme else ""),
            ),
            elapsed=time.perf_counter() - started,
        )

    result = retrieval.retrieve(search_question, collection=collection)
    hits = result.hits
    above = result.above_floor()

    if not result.has_usable_context:
        return finish(
            category=decision.category,
            answer=guardrails.no_context_answer(),
            used_llm=False,
            refused=True,
            scheme_filter=result.scheme_filter,
            hits=hits,
            above_floor=above,
            problems=(
                f"best similarity {result.best_similarity:.3f} is below the "
                f"floor {config.MIN_SIMILARITY}; nothing was sent to the LLM",
            ),
            elapsed=time.perf_counter() - started,
        )

    context = build_context(above)
    citation = _pick_citation(above)
    if citation is None:
        return finish(
            category=decision.category,
            answer=guardrails.no_context_answer(),
            used_llm=False,
            refused=True,
            scheme_filter=result.scheme_filter,
            hits=hits,
            above_floor=above,
            problems=("no retrieved chunk passed the official-domain policy",),
            elapsed=time.perf_counter() - started,
        )

    raw, model, error = complete(
        system_prompt(),
        f"Source context:\n\n{context}\n\nQuestion: {search_question}",
        context=context,
    )
    ungrounded_notes = take_ungrounded_notes()
    if error:
        # Deliberately not a decline. A decline is a claim about the corpus -
        # "the sources do not contain this" - and this is a claim about the
        # provider. Collapsing the two would let a rate limit show up in the
        # pipeline report as evidence about the corpus, which is exactly the
        # kind of miscounting the adjudicated-decline work exists to prevent.
        return finish(
            category=decision.category,
            answer=MSG_MODEL_UNAVAILABLE,
            used_llm=False,
            refused=True,
            declined=False,
            scheme_filter=result.scheme_filter,
            hits=hits,
            above_floor=above,
            problems=(f"model unavailable: {error}",),
            llm_error=error,
            elapsed=time.perf_counter() - started,
        )

    verified = guardrails.verify_answer(
        raw, citation.source_url, citation.fetched_at, context=context
    )
    declined = guardrails.is_decline(verified.answer)

    return finish(
        category=decision.category,
        answer=verified.answer,
        used_llm=True,
        refused=not verified.ok,
        source_url=verified.citation or citation.source_url,
        fetched_at=citation.fetched_at,
        chunk_id=citation.chunk_id,
        scheme_filter=result.scheme_filter,
        hits=hits,
        above_floor=above,
        problems=verified.problems + ungrounded_notes,
        repairs=verified.repairs,
        declined=declined,
        keyword_hit=result.any_keyword_hit(),
        elapsed=time.perf_counter() - started,
        model=model,
    )
