"""Phase 5 verification: the 10 PRD questions plus routing and grounding.

    .venv\\Scripts\\python -m tests.test_pipeline

Checks, in the order implementation.md Phase 5 lists them:
  1. citations         exactly one link per answer, on an allowed domain
  2. length and date   body <= 3 sentences, app-supplied date, no model date
  3. routing           refusals answer with no Groq call at all
  4. grounding         every figure in the answer appears in a retrieved chunk
  5. out-of-corpus     a question the corpus cannot answer does not invent one
  6. retrieval quality top-k chunks are logged per question

Grounding is checked against the retrieved text rather than by opening a
browser, because the corpus *is* the fetched source text. It proves the model
copied rather than invented; it cannot prove the live page agrees, which
still needs a human pass.
"""

from __future__ import annotations

import re
import sys
import time

import config
import generator
import guardrails
import retrieval
from ingest import store

PRD_QUESTIONS = [
    "What is the expense ratio of SBI Flexicap Fund?",
    "What is the lock-in period for SBI ELSS Tax Saver Fund?",
    "What is the exit load for SBI Large Cap Fund?",
    "What is the minimum SIP amount for SBI Small Cap Fund?",
    "What is the riskometer and benchmark for SBI Balanced Advantage Fund?",
    "How do I download my capital-gains statement for SBI ELSS?",
    "Where can I download the latest factsheet for SBI Flexicap Fund?",
    "Is there an exit load if I redeem SBI Large Cap Fund units after 1 year?",
    "What is the benchmark index for SBI ELSS?",
    "Where can I find the KIM/SID for SBI Balanced Advantage Fund?",
]

# Must be answered without any Groq call.
REFUSAL_QUESTIONS = [
    ("Should I buy SBI Small Cap Fund?", "advisory"),
    ("Is SBI Large Cap Fund good for me?", "advisory"),
    ("Which of these funds performed best last year?", "performance"),
    ("What is the price of gold today?", "off_topic"),
]

# Corpus genuinely does not contain today's NAV.
OUT_OF_CORPUS = [
    "What is the NAV of SBI Flexicap Fund today?",
    "What will the returns be next year?",
]

# For each question the model declined: does that scheme's part of the corpus
# contain the fact at all? "yes" means retrieval missed it and is our bug;
# "no" means the refusal is honest and the pipeline is behaving.
PROBES = {
    "What is the exit load for SBI Large Cap Fund?": ("SBI Large Cap Fund", "exit load"),
    "Is there an exit load if I redeem SBI Large Cap Fund units after 1 year?":
        ("SBI Large Cap Fund", "exit load"),
    "How do I download my capital-gains statement for SBI ELSS?":
        ("SBI ELSS Tax Saver Fund", "capital gains"),
    "Where can I find the KIM/SID for SBI Balanced Advantage Fund?":
        ("SBI Balanced Advantage Fund", "offer-document-sid-kim"),
    "What is the riskometer and benchmark for SBI Balanced Advantage Fund?":
        ("SBI Balanced Advantage Fund", "riskometer"),
}

# Declines that are a gap in what SBI MF publishes, not a retrieval bug.
#
# The probe heuristic above cannot tell these apart on its own: it counts any
# corpus chunk containing the probe phrase, and six ELSS chunks contain
# "capital gains" - all of them tax treatment ("long-term gains above Rs. 1
# lakh are taxed at 20%"), none of them a download procedure. Term presence is
# not evidence that the corpus can answer the question, so the heuristic called
# this a retrieval miss and blamed our recall for a fact nobody publishes.
#
# The claim is checked rather than trusted: see PROCEDURE_CUES below. If a
# future corpus does state the procedure, the entry becomes a failure and says
# so. Evidence for the gap is in samples/sample_qa.md.
SOURCE_GAPS = {
    "How do I download my capital-gains statement for SBI ELSS?":
        "SBI MF publishes no public page with the procedure (12 candidate "
        "paths 404, no account-statement page in the site nav)",
}

# Words that indicate a chunk actually tells the reader how to do something,
# rather than merely mentioning the subject. Used to corroborate a source gap.
PROCEDURE_CUES = (
    "download", "how to", "where can i", "step ", "steps to", "procedure",
    "log in", "login", "sign in", "register", "log into", "available at",
)

# Filled in by check_declined_questions, read by the summary.
SOURCE_GAP_DECLINES: list[str] = []


def _states_procedure(text: str, probe: str, window: int = 200) -> bool:
    """Whether `text` says how to do something, near the probe phrase.

    Proximity is the whole point. Requiring the cue anywhere in the chunk gave
    a false corroboration: "capital gains" and the bare word "procedure" both
    occur in one 150-word chunk about redemption mechanics, without the chunk
    describing a statement download anywhere. A real procedure puts them side
    by side - "download the capital gains statement" - so the cue has to be
    near the probe to count.
    """
    flat = " ".join((text or "").lower().split())
    probe = probe.lower()
    start = flat.find(probe)
    while start != -1:
        span = flat[max(0, start - window): start + len(probe) + window]
        if any(cue in span for cue in PROCEDURE_CUES):
            return True
        start = flat.find(probe, start + 1)
    return False

# Figures: a number with a percent, currency, or decimal, optionally with
# surrounding units. Used to prove the answer copied rather than invented.
_FIGURE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|rupees?|rs\.?|lakh|crore|bps|"
    r"years?|months?|days?)\b",
    re.I,
)
_DATE_IN_ANSWER = re.compile(
    r"(?i)\b(?:as\s+of|last\s+updated|as\s+on|updated\s+on|"
    r"20\d{2}-\d{2}-\d{2}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|"
    r"oct|nov|dec)[a-z]*\s+20\d{2})\b"
)
_PASS, _FAIL = "ok  ", "FAIL"


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _body(answer: generator.Answer) -> str:
    return answer.answer.split(config.LAST_UPDATED_PREFIX)[0].strip()


def check_citations(answers) -> bool:
    _rule("1. CITATIONS: exactly one link, allowed domain")
    ok = True
    for answer in answers:
        body = _body(answer)
        urls = guardrails.count_urls(body)
        allowed = bool(answer.source_url) and config.is_allowed_url(answer.source_url)
        good = urls == 1 and allowed
        ok = ok and good
        print(f"  {_PASS if good else _FAIL} {urls} link(s), "
              f"domain {'allowed' if allowed else 'NOT ALLOWED'}")
        if answer.source_url:
            print(f"       {answer.source_url[:100]}")
        if not good:
            print(f"       Q: {answer.question}")
    return ok


def check_length_and_date(answers) -> bool:
    _rule("2. LENGTH AND DATE: <=3 sentences, app-supplied date")
    ok = True
    for answer in answers:
        body = _body(answer)
        sentences = guardrails.count_sentences(body)
        has_prefix = config.LAST_UPDATED_PREFIX in answer.answer
        # The model must not state a date of its own. The app's own
        # "Last updated from sources:" line is excluded before scanning.
        model_date = bool(_DATE_IN_ANSWER.search(body))
        good = (
            sentences <= config.MAX_ANSWER_SENTENCES
            and has_prefix
            and answer.fetched_at
            and not model_date
        )
        ok = ok and good
        print(f"  {_PASS if good else _FAIL} {sentences} sentence(s), "
              f"date {answer.fetched_at or 'MISSING'}, "
              f"model stated a date: {model_date}")
        if not good:
            print(f"       Q: {answer.question}")
    return ok


def check_routing(collection) -> bool:
    _rule("3. ROUTING: refusals make no Groq call")
    ok = True
    for question, expected in REFUSAL_QUESTIONS:
        answer = generator.ask(question, collection=collection)
        good = (
            answer.category == expected
            and not answer.used_llm
            and answer.refused
            and not answer.hits
        )
        ok = ok and good
        print(f"  {_PASS if good else _FAIL} {answer.category:<11} "
              f"llm={'yes' if answer.used_llm else 'no ':<3} "
              f"retrieved={len(answer.hits)}  {question[:44]}")
        if not good:
            print(f"       expected {expected}; hits={len(answer.hits)}")
    return ok


def check_grounding(answers) -> bool:
    _rule("4. GROUNDING: every figure in the answer is in a retrieved chunk")
    ok = True
    for answer in answers:
        body = _body(answer)
        figures = [f.group(0).strip().lower() for f in _FIGURE.finditer(body)]
        if not figures:
            print(f"  {_PASS} no numeric claim to verify  {answer.question[:44]}")
            continue
        haystack = " ".join(h.text for h in answer.above_floor).lower()
        haystack = re.sub(r"\s+", " ", haystack)
        missing = []
        for figure in figures:
            number = re.sub(r"[^0-9.,]", "", figure)
            if number and number not in haystack:
                missing.append(figure)
        good = not missing
        ok = ok and good
        print(f"  {_PASS if good else _FAIL} {len(figures)} figure(s) checked  "
              f"{answer.question[:44]}")
        for figure in figures:
            number = re.sub(r"[^0-9.,]", "", figure)
            mark = _PASS if number in haystack else _FAIL
            print(f"       {mark} {figure!r}")
        if missing:
            print(f"       NOT IN CONTEXT: {missing}")
    return ok


def check_out_of_corpus(collection) -> bool:
    _rule("5. OUT-OF-CORPUS: refuses instead of inventing a number")
    ok = True
    for question in OUT_OF_CORPUS:
        answer = generator.ask(question, collection=collection)
        said_no = (
            "not in the" in answer.answer.lower()
            or "could not find" in answer.answer.lower()
        )
        # A live NAV answer is acceptable only if the LLM declined to state a
        # figure. Flag any bare number in a refusal.
        body = _body(answer)
        figures = [f.group(0) for f in _FIGURE.finditer(body)]
        good = said_no and not figures
        ok = ok and good
        print(f"  {_PASS if good else _FAIL} declined={said_no} "
              f"llm={'yes' if answer.used_llm else 'no'}")
        print(f"       {body[:150]}")
        if figures:
            print(f"       FIGURES IN A REFUSAL: {figures}")
    return ok


def check_declined_questions(answers, collection) -> bool:
    """For every declined question, decide whether it was honest or a miss.

    A decline is only correct if the corpus genuinely lacks the fact. If the
    fact is in the corpus but never reached the model, that is a retrieval
    recall bug, and it must fail here rather than be waved through as "the
    model said no".
    """
    _rule("7. DECLINED QUESTIONS: honest refusal, or a retrieval miss?")
    ok = True
    SOURCE_GAP_DECLINES.clear()
    for answer in answers:
        if not answer.declined or answer.question not in PROBES:
            continue
        scheme, probe = PROBES[answer.question]
        got = collection.get(where={"scheme": scheme}, include=["documents"])
        docs = got.get("documents") or []
        normalised = [" ".join((d or "").lower().split()) for d in docs]
        found = sum(1 for d in normalised if probe.lower() in d)
        in_retrieved = any(
            probe.lower() in " ".join(h.text.lower().split())
            for h in answer.above_floor
        )
        if answer.question in SOURCE_GAPS:
            # Corroborate the documented gap instead of believing it.
            corroborating = sum(
                1 for d in docs if _states_procedure(d, probe)
            )
            if corroborating:
                verdict = (f"documented gap CONTRADICTED: {corroborating} "
                           f"chunk(s) do state a procedure")
                good = False
            else:
                verdict = f"source gap, not a retrieval bug: {SOURCE_GAPS[answer.question]}"
                good = True
                SOURCE_GAP_DECLINES.append(answer.question)
        elif found and not in_retrieved:
            verdict, good = "RETRIEVAL MISS (our bug)", False
        elif in_retrieved:
            verdict, good = "in context, model still declined (suspicious)", False
        else:
            verdict, good = f"absent from corpus ({found} chunks), honest", True
        ok = ok and good
        print(f"  {_PASS if good else _FAIL} {scheme}: '{probe}' -> {verdict}")
        print(f"       corpus chunks for this scheme: {len(docs):,}")
    declined = [a for a in answers if a.declined]
    if not declined:
        print("  (no PRD question was declined; nothing to adjudicate)")
    else:
        unexplained = len(declined) - len(SOURCE_GAP_DECLINES)
        print(f"\n  {len(declined)} of {len(PRD_QUESTIONS)} PRD questions "
              f"still decline (spec wants all 10 answered): "
              f"{len(SOURCE_GAP_DECLINES)} documented source gap(s), "
              f"{unexplained} unexplained")
    return ok


def check_retrieval_quality(answers) -> None:
    _rule("6. RETRIEVAL QUALITY: top-k logged (spec step 5, read these)")
    for answer in answers:
        print(f"\n  Q: {answer.question}")
        print(f"     filter: {answer.scheme_filter or 'none'}  "
              f"best: {max((h.similarity for h in answer.hits), default=0):+.3f}")
        for i, hit in enumerate(answer.above_floor[:3], start=1):
            print(f"     [{i}] {hit.similarity:+.3f} {hit.chunk_id} "
                  f"{hit.doc_type}/{hit.section[:34]}")
            print(f"         {hit.preview(150)}")


def main() -> int:
    if not config.GROQ_API_KEY:
        print("GROQ_API_KEY is not set.")
        return 1

    collection = store.open_collection(store.get_client())
    print(f"collection: {collection.count():,} vectors | model {config.LLM_MODEL}")
    print(f"top-{config.TOP_K} | floor {config.MIN_SIMILARITY} | "
          f"cap {config.MAX_ANSWER_SENTENCES} sentences")

    _rule("0. THE 10 PRD QUESTIONS")
    answers: list[generator.Answer] = []
    for i, question in enumerate(PRD_QUESTIONS, start=1):
        started = time.perf_counter()
        answer = generator.ask(question, collection=collection)
        answers.append(answer)
        if answer.grounded:
            status = "answer"
        elif answer.declined:
            status = "DECLINED by the model (context insufficient)"
        else:
            status = f"REFUSED ({answer.category})"
        print(f"\n  [{i:>2}] {question}")
        print(f"       -> {status}  {answer.elapsed:.1f}s  "
              f"llm={'yes' if answer.used_llm else 'no'}")
        print("       " + " ".join(_body(answer).split())[:190])
        if answer.problems:
            print(f"       problems: {answer.problems}")
        if answer.repairs:
            print(f"       repairs : {answer.repairs}")

    results = {
        "citations": check_citations(answers),
        "length and date": check_length_and_date(answers),
        "grounding": check_grounding(answers),
        "routing (no LLM call)": check_routing(collection),
        "out-of-corpus": check_out_of_corpus(collection),
        "declined questions": check_declined_questions(answers, collection),
    }
    check_retrieval_quality(answers)

    _rule("SUMMARY")
    for name, passed in results.items():
        print(f"  {_PASS if passed else _FAIL} {name}")
    answered = sum(1 for a in answers if a.grounded)
    declined = sum(1 for a in answers if a.declined)
    print(f"\n  {answered}/{len(PRD_QUESTIONS)} PRD questions answered, "
          f"{declined} declined, "
          f"{len(PRD_QUESTIONS) - answered - declined} routed away")
    if declined:
        gaps = len(SOURCE_GAP_DECLINES)
        unexplained = declined - gaps
        print(f"\n  Spec line 175 wants all {len(PRD_QUESTIONS)} answered, so "
              f"this run still fails.\n  But the declines are not all the "
              f"same kind of problem:\n"
              f"    {gaps} documented source gap(s) - SBI MF does not publish "
              f"the fact.\n      Correcting behaviour needs a new official "
              f"source, not code.\n"
              f"    {unexplained} unexplained decline(s) - recall shortfall or "
              f"suspicious context,\n      which is our bug and must be fixed.")
        return 1
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
