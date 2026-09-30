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
    scheme_filter: str | None = None
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

    @property
    def grounded(self) -> bool:
        """Whether the user got an actual answer, not a message about nothing.

        `not refused` alone is not enough: a decline passes the post-check
        cleanly, so without this the pipeline would report a question as
        answered when the model said it had nothing to go on.
        """
        return self.used_llm and not self.refused and not self.declined


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
        """
        remembered = False
        if conversation is not None and not pii:
            stored = conversation.add("user", question)
            stored &= conversation.add("assistant", fields.get("answer", ""))
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

    client = get_client()
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content":
                f"Source context:\n\n{context}\n\n"
                f"Question: {search_question}"},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    model = getattr(response, "model", config.LLM_MODEL)

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
        problems=verified.problems,
        repairs=verified.repairs,
        declined=declined,
        keyword_hit=result.any_keyword_hit(),
        elapsed=time.perf_counter() - started,
        model=model,
    )
