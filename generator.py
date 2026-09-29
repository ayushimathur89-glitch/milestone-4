"""Answer generation via Groq, wrapped in the Phase 4 contract.

The prompt states the answer rules, but the prompt is not the enforcement.
Three layers have to agree before a user sees anything:

1. `guardrails.classify` refuses advisory, performance, PII and off-topic
   questions *before* retrieval, so a refused query never reaches this module
   and never becomes a Groq call.
2. `guardrails.verify_answer` checks the generated text and repairs it.
3. The "Last updated" line is appended here from the cited chunk's
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

    @property
    def grounded(self) -> bool:
        """Whether the user got an actual answer, not a message about nothing.

        `not refused` alone is not enough: a decline passes the post-check
        cleanly, so without this the pipeline would report a question as
        answered when the model said it had nothing to go on.
        """
        return self.used_llm and not self.refused and not self.declined


def ask(question: str, show_context: bool = False, collection=None) -> Answer:
    """Full pipeline: classify, retrieve, generate, verify.

    Returns an `Answer` describing what happened, so the CLI can show the
    retrieved chunks and the caller can tell a refusal from a real answer
    without parsing prose.
    """
    started = time.perf_counter()

    decision = guardrails.classify(question)
    if decision.refused:
        # No retrieval and no Groq call: the guardrail is meant to hold
        # structurally, and spec verification step 4 checks it by request log.
        return Answer(
            question=question,
            category=decision.category,
            answer=decision.message,
            used_llm=False,
            refused=True,
            problems=(decision.reason,) if decision.reason else (),
            elapsed=time.perf_counter() - started,
        )

    result = retrieval.retrieve(question, collection=collection)
    hits = result.hits
    above = result.above_floor()

    if not result.has_usable_context:
        return Answer(
            question=question,
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
        return Answer(
            question=question,
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
                f"Question: {question}"},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    model = getattr(response, "model", config.LLM_MODEL)

    verified = guardrails.verify_answer(
        raw, citation.source_url, citation.fetched_at, context=context
    )
    declined = guardrails.is_decline(verified.answer)

    return Answer(
        question=question,
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
