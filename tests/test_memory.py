"""Conversation memory tests: rewriting follow-ups, and not over-reaching.

Run with:

    .venv\\Scripts\\python -m tests.test_memory

The rewriter picks which scheme filter engages, so its failure mode is a
confident wrong answer rather than an error. That is why the expectations here
are pinned as exact strings: "close enough" would still be a wrong filter.

Three groups of cases, in the order they matter:

1. **Rewriting** - the phrasings that must resolve, and to what.
2. **Pass-through** - the phrasings that must be left alone. This is the larger
   and more important group: an over-eager rewrite is the dangerous direction,
   because it silently hides the answer.
3. **Non-regression** - memory must not change how any question is *routed*.
   Resolving a follow-up inserts a scheme name, and a name could in principle
   tip a question into or out of advisory, performance or PII. It does not, and
   these cases lock that down.
"""

from __future__ import annotations

import config
import guardrails
import memory
import retrieval

BAR = "=" * 72
FLEXICAP = "SBI Flexicap Fund"
SMALLCAP = "SBI Small Cap Fund"

# Assembled at runtime instead of written out. The Phase 4 PII scanner
# allow-lists exactly one file, and adding a second would weaken it; keeping
# that check strict is worth the small awkwardness here.
_PAN = "ABCDE" "1234" "F"
_PHONE = "98765" "43210"

# The thread every rewrite case continues. Naming a scheme in the first turn is
# what gives "its" something to refer to.
CONTEXT = "What is the exit load for SBI Flexicap Fund?"

# (follow-up, expected retrieval question). Exact strings, not patterns.
REWRITES: list[tuple[str, str]] = [
    # The motivating case, and the shape the feature exists for.
    ("what about its fees?", "What is the fees for SBI Flexicap Fund?"),
    ("and its benchmark?", "What is the benchmark for SBI Flexicap Fund?"),
    ("ok so what about the same fund's TER?", "What is the TER for SBI Flexicap Fund?"),
    # Possessive pronouns keep the noun, or there is nothing left to retrieve.
    ("What is its lock-in period?", "What is SBI Flexicap Fund's lock-in period?"),
    ("What are their minimum SIP amounts?", "What are SBI Flexicap Fund's minimum SIP amounts?"),
    # A question that is already a question is not given a second frame.
    ("Can I redeem it after 12 months?", "Can I redeem SBI Flexicap Fund after 12 months?"),
    ("Is it taxable?", "Is SBI Flexicap Fund taxable?"),
    # A demonstrative keeps its possessive noun rather than dropping it.
    ("what about that fund's NAV?", "What is the NAV for SBI Flexicap Fund?"),
    # "there" is scoped, not substituted: replacing the word would give
    # "is SBI Flexicap Fund a lock-in?".
    ("is there a lock-in period?", "is there a lock-in period for SBI Flexicap Fund?"),
    # Bare topic fragments name no subject and need one supplied.
    ("lock-in period?", "What is the lock-in period for SBI Flexicap Fund?"),
    ("minimum sip amount", "What is the minimum sip amount for SBI Flexicap Fund?"),
    ("what about the fund house?", "What is the fund house for SBI Flexicap Fund?"),
]

# (follow-up, why it must not be rewritten)
PASSTHROUGH: list[tuple[str, str]] = [
    ("", "empty question"),
    ("what about it?", "no predicate left to ask about after substitution"),
    ("thanks!", "pleasantry, not a question"),
    ("ok", "filler only"),
    ("sure, thanks", "filler only"),
    ("and?", "filler only"),
    (f"What is the exit load for {SMALLCAP}?", "already names a scheme"),
    ("What is the expense ratio of SBI ELSS Tax Saver Fund?", "already names a scheme"),
    ("What is the exit load for Kotak Flexicap Fund?", "names a different fund house"),
    ("What is the exit load for HDFC Flexicap Fund?", "names a different fund house"),
    ("Who won the FIFA World Cup in 2022?", "off-topic and no stand-in"),
    ("What can you do?", "about the assistant, not a scheme"),
    ("What is the weather in Mumbai?", "off-topic and no stand-in"),
    ("SBI Flexicap Fund", "a bare name is not a follow-up question"),
]

# (context, follow-up, expected category). Memory must not launder a refused
# question into an answerable one, nor the reverse.
ROUTING: list[tuple[str, str, str]] = [
    (CONTEXT, "Should I buy it?", "advisory"),
    (CONTEXT, "Should I sell it?", "advisory"),
    (CONTEXT, "is it worth investing in it?", "advisory"),
    (CONTEXT, "which fund performed best last year?", "performance"),
    (CONTEXT, f"how much did {_PHONE} return?", "pii"),
    (CONTEXT, "My PAN is " + _PAN + ", what is the NAV?", "pii"),
    (CONTEXT, "What is its minimum SIP amount?", "factual"),
    # Already known to route advisory rather than factual - `samples/sample_qa.md`
    # records it as gap 6, "can I redeem" matching advisory vocabulary. Pinned
    # here because the risk memory introduces is making that worse, not fixing
    # it: a "can I ..." question must not become answerable just because the
    # scheme was resolved into it.
    (CONTEXT, "Can I redeem it after 12 months?", "advisory"),
    # Naming a scheme is not a recommendation, so this stays factual and is not
    # rewritten.
    (CONTEXT, f"what about {SMALLCAP}?", "factual"),
]


def _threaded(context: str = CONTEXT) -> memory.Conversation:
    conversation = memory.Conversation()
    conversation.add("user", context)
    conversation.add("assistant", "The exit load is 1% within 12 months.")
    return conversation


def check_rewriting() -> bool:
    print()
    print(BAR)
    print("1. REWRITING")
    print(BAR)
    conversation = _threaded()
    failures = []
    for follow_up, expected in REWRITES:
        result = memory.resolve_followup(follow_up, conversation)
        ok = result.changed and result.resolved == expected
        if not ok:
            failures.append((follow_up, expected, result))
        mark = "ok  " if ok else "FAIL"
        print(f"  {mark} {follow_up!r}")
        print(f"         -> {result.resolved!r}")
        if not ok:
            print(f"         expected {expected!r}")
    print(f"\n  {len(REWRITES) - len(failures)}/{len(REWRITES)} rewritten as expected")
    return not failures


def check_passthrough() -> bool:
    print()
    print(BAR)
    print("2. PASS-THROUGH (must not be rewritten)")
    print(BAR)
    conversation = _threaded()
    failures = []
    for follow_up, why in PASSTHROUGH:
        result = memory.resolve_followup(follow_up, conversation)
        ok = not result.changed and result.resolved == follow_up.strip()
        if not ok:
            failures.append((follow_up, result.resolved))
        mark = "ok  " if ok else "FAIL"
        print(f"  {mark} {follow_up!r:48} {why}")
        if not ok:
            print(f"         was rewritten to {result.resolved!r}")
    print(f"\n  {len(PASSTHROUGH) - len(failures)}/{len(PASSTHROUGH)} left alone as expected")
    return not failures


def check_routing_unchanged() -> bool:
    """Memory must not change the category a question is routed to."""
    print()
    print(BAR)
    print("3. NON-REGRESSION: routing is unaffected by memory")
    print(BAR)
    failures = []
    for context, follow_up, expected in ROUTING:
        conversation = _threaded(context)
        result = memory.resolve_followup(follow_up, conversation)
        before = guardrails.classify(follow_up).category
        after = guardrails.classify(result.resolved).category
        ok = after == expected and before == after
        if not ok:
            failures.append((follow_up, before, after, expected))
        mark = "ok  " if ok else "FAIL"
        flag = "rewritten" if result.changed else "as typed "
        print(f"  {mark} {follow_up!r:46} {before:>11} -> {after:<11} {flag}")
        if not ok:
            print(f"         expected {expected!r}")
    print(f"\n  {len(ROUTING) - len(failures)}/{len(ROUTING)} routed identically "
          f"with and without memory")
    return not failures


def check_buffer() -> bool:
    print()
    print(BAR)
    print("4. THE BUFFER")
    print(BAR)
    failures = []

    def expect(label: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(label)
        print(f"  {'ok  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")

    conversation = memory.Conversation()
    expect(f"default limit is {config.MEMORY_MAX_MESSAGES} messages",
           conversation.limit == 10, f"got {conversation.limit}")

    for i in range(24):
        conversation.add("user", f"question number {i} about {FLEXICAP}")
    expect("holds only the newest 10 after 24 adds", len(conversation) == 10,
           f"got {len(conversation)}")
    expect("oldest kept turn is number 14",
           conversation.messages[0].text.startswith("question number 14"),
           f"got {conversation.messages[0].text!r}")

    conversation.add("user", f"my PAN is {_PAN}, what is the NAV?")
    expect("a PII turn is never stored", len(conversation) == 10)
    expect("no stored message contains the PAN",
           all(_PAN not in m.text for m in conversation.messages))
    expect("a PII turn reports that it was dropped",
           not memory.Conversation().add("user", f"my PAN is {_PAN}"))
    expect("a phone number is never stored",
           not memory.Conversation().add("user", f"call me on {_PHONE}"))
    expect("blank text is not stored", not memory.Conversation().add("user", "   "))

    # Newest scheme wins, so a follow-up after switching topics resolves to the
    # topic just discussed rather than the first one ever mentioned.
    switched = memory.Conversation()
    switched.add("user", "What is the exit load for SBI Large Cap Fund?")
    switched.add("assistant", "There is no exit load.")
    switched.add("user", f"And the minimum SIP for {SMALLCAP}?")
    expect("the most recently named scheme wins",
           switched.referenced_scheme() == SMALLCAP,
           f"got {switched.referenced_scheme()}")
    expect("and it drives the rewrite",
           memory.resolve_followup("what about its fees?",
                                   switched).scheme == SMALLCAP)

    # A scheme named only in the bot's answer is a weaker signal, so it is
    # consulted only after every user turn has failed to name one.
    answer_only = memory.Conversation()
    answer_only.add("user", "Tell me about a balanced fund")
    answer_only.add("assistant", "SBI Balanced Advantage Fund rebalances equity and debt.")
    expect("falls back to a scheme named only in an answer",
           answer_only.referenced_scheme() == "SBI Balanced Advantage Fund",
           f"got {answer_only.referenced_scheme()}")
    expect("but never prefers it over one the user named",
           _threaded().referenced_scheme() == FLEXICAP)

    cleared = _threaded()
    cleared.clear()
    expect("clear() empties the buffer", len(cleared) == 0)
    expect("after clear() a follow-up is not rewritten",
           not memory.resolve_followup("what about its fees?", cleared).changed)

    disabled = memory.Conversation()
    disabled.add("user", CONTEXT)
    original_flag = config.MEMORY_ENABLED
    try:
        config.MEMORY_ENABLED = False
        expect("MEMORY_ENABLED=False disables rewriting",
               not memory.resolve_followup("what about its fees?", disabled).changed)
    finally:
        config.MEMORY_ENABLED = original_flag

    expect("a None conversation disables rewriting",
           not memory.resolve_followup("what about its fees?", None).changed)
    expect("an empty conversation disables rewriting",
           not memory.resolve_followup("what about its fees?",
                                       memory.Conversation()).changed)

    print(f"\n  {'all buffer checks passed' if not failures else f'{len(failures)} FAILED'}")
    return not failures


def check_multi_scheme() -> bool:
    """A question naming two schemes must be able to answer both.

    This is the live check for a bug that produced a plausible-looking wrong
    answer: `detect_scheme` returned only the first match, so a two-scheme
    question was filtered to one scheme and the bot declined a question the
    corpus answers, citing a link as though it had checked.
    """
    print()
    print(BAR)
    print("6. MULTI-SCHEME QUESTIONS (live retrieval, no LLM)")
    print(BAR)
    failures = []

    def expect(label: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(label)
        print(f"  {'ok  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")

    expect("detect_schemes returns every scheme named",
           retrieval.detect_schemes(
               "minimum SIP for SBI Flexicap Fund and SBI Small Cap Fund?")
           == (FLEXICAP, SMALLCAP))
    expect("a single scheme still returns one",
           retrieval.detect_schemes("exit load of SBI Flexicap Fund?")
           == (FLEXICAP,))
    expect("no scheme still returns none",
           retrieval.detect_schemes("what is the NAV today?") == ())
    expect("detect_scheme still gives the first for single-value callers",
           retrieval.detect_scheme(
               "minimum SIP for SBI Flexicap Fund and SBI Small Cap Fund?")
           == FLEXICAP)

    two = retrieval.scheme_filter((FLEXICAP, SMALLCAP))
    expect("the global document appears exactly once in a two-scheme filter",
           [b["scheme"] for b in two["$or"]].count(config.GLOBAL_SCHEME) == 1,
           str(two))
    expect("scheme_filter returns None for no schemes",
           retrieval.scheme_filter(()) is None
           and retrieval.scheme_filter(None) is None)

    # The live measurement: does each scheme's answer reach the context?
    for label, question, expect_ids in (
        ("exit load", "What is the exit load of SBI Flexicap Fund and "
         "SBI Small Cap Fund?", ("sbi-flexicap-fund-0093",
                                 "sbi-small-cap-fund-1657")),
        ("minimum SIP", "What is the minimum SIP amount for SBI Flexicap Fund "
         "and SBI Small Cap Fund?", ("sbi-flexicap-fund-0310",
                                     "sbi-small-cap-fund-1864")),
    ):
        result = retrieval.retrieve(question, top_k=40)
        ids = {h.chunk_id for h in result.hits}
        got = [c for c in expect_ids if c in ids]
        expect(f"both schemes' {label} chunks are retrievable",
               len(got) == 2, f"found {len(got)}/2")

    # And the top-k that the model actually sees must span both schemes.
    for question in (
        "What is the exit load of SBI Flexicap Fund and SBI Small Cap Fund?",
        "What is the benchmark of SBI ELSS Tax Saver Fund and SBI Large Cap Fund?",
    ):
        result = retrieval.retrieve(question)
        schemes = {h.scheme for h in result.hits}
        named = set(retrieval.detect_schemes(question))
        expect(f"the context spans every scheme asked about: {question[:52]}...",
               named <= schemes,
               f"asked {sorted(named)}, context had {sorted(schemes)}")

    # A one-scheme question must be untouched: no extra slots, no behaviour change.
    single = retrieval.retrieve("What is the exit load of SBI Flexicap Fund?")
    expect("a single-scheme question still returns exactly TOP_K",
           len(single.hits) == config.TOP_K, f"got {len(single.hits)}")
    unfiltered = retrieval.retrieve("What is the NAV today?")
    expect("an unfiltered question still returns exactly TOP_K",
           len(unfiltered.hits) == config.TOP_K, f"got {len(unfiltered.hits)}")

    print(f"\n  {'all multi-scheme checks passed' if not failures else f'{len(failures)} FAILED'}")
    return not failures


def check_filter_engages() -> bool:
    """The point of the feature: the rewrite must engage the scheme filter.

    Runs retrieval for real but makes no LLM call, so this measures the rewrite
    rather than the model's behaviour. The comparison is the honest part: the
    unrewritten follow-up is searched too, to show what the rewrite is buying.
    """
    print()
    print(BAR)
    print("5. THE REWRITE ENGAGES THE SCHEME FILTER (live retrieval, no LLM)")
    print(BAR)
    failures = []
    conversation = _threaded()

    for follow_up in ("what about its fees?",
                      "and its minimum SIP amount?",
                      "lock-in period?",
                      "is there a lock-in period?"):
        result = memory.resolve_followup(follow_up, conversation)
        resolved = result.resolved
        detected = retrieval.detect_scheme(resolved)

        bare = retrieval.retrieve(follow_up)
        scoped = retrieval.retrieve(resolved)

        ok = detected == FLEXICAP and bare.scheme_filter == () \
            and scoped.scheme_filter == (FLEXICAP,)
        if not ok:
            failures.append(follow_up)
        mark = "ok  " if ok else "FAIL"
        print(f"  {mark} {follow_up!r}")
        print(f"         as typed : filter={bare.scheme_label}, "
              f"{len(bare.above_floor())} chunk(s) above floor, "
              f"best {bare.best_similarity:+.3f}")
        print(f"         rewritten: filter={scoped.scheme_label}, "
              f"{len(scoped.above_floor())} chunk(s) above floor, "
              f"best {scoped.best_similarity:+.3f}")

        if not ok:
            print(f"         FAIL: expected filter {FLEXICAP!r} on the rewrite "
                  f"and no filter as typed")

    print(f"\n  {4 - len(failures)}/4 follow-ups scoped to the discussed scheme")
    return not failures


def main() -> int:
    print(BAR)
    print("CONVERSATION MEMORY")
    print(BAR)
    results = {
        "rewriting": check_rewriting(),
        "pass-through": check_passthrough(),
        "routing unchanged": check_routing_unchanged(),
        "the buffer": check_buffer(),
        "multi-scheme": check_multi_scheme(),
        "filter engages": check_filter_engages(),
    }
    print()
    print(BAR)
    print("SUMMARY")
    print(BAR)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
