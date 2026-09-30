"""Interactive CLI for testing the Q&A pipeline.

    .venv\\Scripts\\python ask_cli.py
    .venv\\Scripts\\python ask_cli.py --question "What is the exit load?"
    .venv\\Scripts\\python ask_cli.py --no-memory
    .venv\\Scripts\\python ask_cli.py --show-hits 4

Shows the retrieved chunks for every answer, the similarity score, whether the
scheme filter engaged, and whether a Groq call happened at all, so a refusal
can be confirmed as a refusal rather than inferred from its wording.

Keeps the last 10 messages, so a follow-up such as "what about its fees?" is
resolved to the scheme being discussed before retrieval. When that happens the
resolved question is printed above the chunks, because the question that was
actually searched is not the question that was typed.

Commands: type a question and press enter. /chunks N changes how many chunks
are printed. /all prints every retrieved chunk, not just those above the
floor. /filter toggles the scheme filter. /history shows what is remembered.
/reset forgets it. /quit exits.
"""

from __future__ import annotations

import argparse

import config
import generator
import memory
import retrieval
from ingest import store

BAR = "=" * 78
THIN = "-" * 78


def show_hits(answer: generator.Answer, limit: int, show_all: bool) -> None:
    hits = answer.hits if show_all else answer.above_floor
    if not answer.hits:
        print("  (nothing retrieved)")
        return
    if not show_all:
        below = len(answer.hits) - len(answer.above_floor)
        print(f"  {len(answer.above_floor)} of {len(answer.hits)} above the "
              f"floor {config.MIN_SIMILARITY}"
              + (f" ({below} hidden, use /all)" if below else ""))
    for i, hit in enumerate(hits[:limit], start=1):
        mark = "PASS" if hit.similarity >= config.MIN_SIMILARITY else "low "
        domain_ok = "ok " if hit.allowed_url else "!! "
        print(f"\n  [{i}] {hit.chunk_id}  sim {hit.similarity:+.3f} {mark}")
        print(f"      {hit.scheme} / {hit.doc_type} / {hit.section or '-'}")
        print(f"      {hit.source_url}  [domain {domain_ok}]")
        print(f"      fetched_at {hit.fetched_at}")
        if hit.matched_terms:
            print(f"      matched question terms: {', '.join(hit.matched_terms)}")
        print(f"      {hit.preview(220)}")


def run_one(question: str, collection, show_context: bool,
            hit_limit: int, show_all: bool,
            conversation: memory.Conversation | None = None) -> generator.Answer:
    answer = generator.ask(question, collection=collection,
                           conversation=conversation)

    print(BAR)
    print(f"Q: {question}")
    if answer.resolved_question and answer.resolved_question != question:
        # Printed because the filter and the ranking below were driven by this
        # line, not by the question the user typed.
        print(f"   (rewritten for retrieval: {answer.resolved_question})")
        print(f"   {answer.rewrite_notes}")
    print(THIN)
    if answer.scheme_filter:
        print(f"  scheme filter : {answer.scheme_filter}")
    else:
        print("  scheme filter : (none detected)")
    if answer.hits:
        print(f"  retrieved     : {len(answer.hits)} chunk(s), "
              f"best similarity {max(h.similarity for h in answer.hits):+.3f}")

    if show_context or not answer.refused:
        print(f"\n  --- retrieved chunks ---")
        show_hits(answer, hit_limit, show_all)

    if answer.scheme_filter and answer.above_floor:
        terms = ", ".join(
            sorted({t for h in answer.above_floor for t in h.matched_terms})
        )
        print(f"  matched terms : {terms or '(none)'}")

    print(f"\n  --- answer ---")
    for line in answer.answer.splitlines():
        print(f"  {line}")

    if answer.declined:
        # A decline is contract-compliant but is not an answer, and saying so
        # plainly is the point: it tells the reader retrieval fell short
        # rather than the guardrails having stopped something.
        print(f"\n  --- outcome ---")
        print("  DECLINED: the retrieved context did not contain the answer.")
        if answer.keyword_hit:
            print("  The question's wording was found in the retrieved chunks, "
                  "so the\n  model had related material and still declined. "
                  "Worth a look.")
        else:
            print("  None of the question's key terms appear in any retrieved "
                  "chunk.")

    if answer.problems:
        print(f"\n  --- problems ---")
        for problem in answer.problems:
            print(f"  ! {problem}")
    if answer.repairs:
        print(f"\n  --- post-check repairs applied ---")
        for repair in answer.repairs:
            print(f"  ~ {repair}")

    if answer.source_url:
        print(f"\n  citation : {answer.source_url}  (from {answer.chunk_id})")
    if conversation is not None:
        print(f"  memory   : {len(conversation)}/{conversation.limit} messages "
              f"remembered")
    print(f"  LLM call : {'yes' if answer.used_llm else 'NO - answered without the model'}")
    if answer.model:
        print(f"  model    : {answer.model}")
    print(f"  elapsed  : {answer.elapsed:.2f}s")
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the mutual fund FAQ bot.")
    parser.add_argument("--question", "-q", action="append",
                        help="ask a question and exit; repeatable")
    parser.add_argument("--show-hits", type=int, default=config.TOP_K,
                        help="how many retrieved chunks to print (default: TOP_K)")
    parser.add_argument("--no-context", action="store_true",
                        help="hide the retrieved chunk detail")
    parser.add_argument("--all", action="store_true",
                        help="print every retrieved chunk, including sub-threshold ones")
    parser.add_argument("--no-memory", action="store_true",
                        help="do not remember previous questions, and do not "
                             "rewrite follow-ups (each question stands alone)")
    args = parser.parse_args()

    if not config.GROQ_API_KEY:
        print("GROQ_API_KEY is not set. Add it to .env before asking questions.")
        return 1

    conversation = None if args.no_memory else memory.Conversation()

    print(BAR)
    print("SBI Mutual Fund FAQs - facts only, no investment advice")
    print(f"model {config.LLM_MODEL} | top-{config.TOP_K} | "
          f"similarity floor {config.MIN_SIMILARITY}")
    if conversation is None:
        print("Memory off: follow-ups are not rewritten and nothing is "
              "remembered.")
    else:
        print(f"Memory on: last {conversation.limit} messages, so \"what about "
              f"its fees?\"\nresolves to the scheme under discussion. "
              f"/history, /reset.")
    print("Type a question, or /quit to exit. /chunks N, /all, /filter, /help")
    print(BAR)

    collection = store.open_collection(store.get_client())
    show_context = not args.no_context
    hit_limit = args.show_hits
    show_all = args.all

    if args.question:
        for question in args.question:
            run_one(question, collection, show_context, hit_limit, show_all,
                    conversation)
            print()
        return 0

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue

        lowered = question.lower()
        if lowered in ("/quit", "/exit", "/q"):
            break
        if lowered == "/help":
            print("  /chunks N   print N retrieved chunks (default "
                  f"{hit_limit})")
            print("  /all        include sub-threshold chunks too")
            print("  /filter     show the scheme filter decision only")
            if conversation is not None:
                print(f"  /history    the {len(conversation)} message(s) "
                      f"remembered")
                print("  /reset      forget them")
            print("  /quit       exit")
            continue
        if lowered.startswith("/chunks"):
            parts = lowered.split()
            if len(parts) == 2 and parts[1].isdigit():
                hit_limit = int(parts[1])
                print(f"  showing {hit_limit} chunk(s)")
            else:
                print("  usage: /chunks 4")
            continue
        if lowered == "/all":
            show_all = not show_all
            print(f"  show sub-threshold chunks: {show_all}")
            continue
        if lowered == "/history":
            if conversation is None:
                print("  memory is off (--no-memory)")
            else:
                print(conversation.transcript())
            continue
        if lowered == "/reset":
            if conversation is None:
                print("  memory is off (--no-memory)")
            else:
                conversation.clear()
                print("  forgotten; the next question has no context")
            continue
        if lowered == "/filter":
            scheme = retrieval.detect_scheme(question)
            print(f"  detected scheme: {scheme or 'none'}")
            continue

        try:
            run_one(question, collection, show_context, hit_limit, show_all,
                    conversation)
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive
            print(f"\n  ! {type(exc).__name__}: {exc}")

    print("bye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
