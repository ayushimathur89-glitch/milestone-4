"""Ingestion entry point.

Phase 2 runs the load and chunk stages only; embed and store are Phase 3.

    .venv\\Scripts\\python -m ingest.run_ingestion
    .venv\\Scripts\\python -m ingest.run_ingestion --limit 2
"""

from __future__ import annotations

import argparse
from collections import Counter

import config
from ingest import chunker, loader

DEMO_FACTS = {
    "expense ratio": ["expense ratio", "total expense ratio", "recurring expense",
                      "expenses of the scheme", "scheme expenses"],
    "exit load": ["exit load"],
    "lock-in": ["lock-in", "lock in period", "lock-in period", "minimum lock-in"],
    "minimum SIP": ["minimum amount for sip", "minimum sip", "sip amount", "minimum investment"],
    "benchmark": ["benchmark"],
    "riskometer": ["riskometer", "o'meter", "risk-o-meter"],
    "capital gains": ["capital gains", "capital gain"],
}


def report_facts(chunks: list[chunker.Chunk]) -> dict[str, list[str]]:
    """Check every fact the demo depends on is actually in the corpus.

    implementation.md Phase 2 step 4: a fact missing here cannot be answered
    in Phase 5, so it is a blocker rather than a nice-to-have.
    """
    haystack = [(c, c.text.lower()) for c in chunks]
    found: dict[str, list[str]] = {}
    for fact, needles in DEMO_FACTS.items():
        hits = [c.chunk_id for c, low in haystack if any(n in low for n in needles)]
        found[fact] = hits
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ingestion pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="only load the first N sources")
    args = parser.parse_args()

    print("=" * 72)
    print("STAGE 1  LOAD")
    print("=" * 72)
    documents, results = loader.run_load(limit=args.limit)
    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]
    print(f"\n  loaded {len(ok)}/{len(results)}  ->  {config.RAW_DIR}")
    if failed:
        print(f"  {len(failed)} failed:")
        for r in failed:
            print(f"    - {r.url}\n      {r.error}")
    print(f"  manifest -> {config.INGEST_MANIFEST_CSV}")

    if not documents:
        print("\nNothing loaded; stopping before the chunk stage.")
        return 1

    print()
    print("=" * 72)
    print("STAGE 2  CHUNK")
    print("=" * 72)
    chunks = chunker.chunk_all(documents)
    per_doc = Counter(c.doc_type for c in chunks)
    per_scheme = Counter(c.scheme for c in chunks)

    print(f"  chunks: {len(chunks)}")
    for key, value in sorted(per_doc.items()):
        print(f"    {key:<14} {value:>5}")
    print("  by scheme:")
    for key, value in sorted(per_scheme.items()):
        print(f"    {key:<30} {value:>5}")

    words = [c.word_count for c in chunks] or [0]
    print(f"  words/chunk: min {min(words)}  max {max(words)}  mean {sum(words) // len(words)}")
    tables = sum(1 for c in chunks if chunker.is_table_block(c.text))
    print(f"  table chunks (atomic, never split): {tables}")

    problems = chunker.validate(chunks)
    print(f"  contract checks: {'PASS' if not problems else 'FAIL'}")
    for problem in problems[:20]:
        print(f"    - {problem}")

    path = chunker.write_chunks_file(chunks)
    print(f"  chunks file -> {path}  ({path.stat().st_size:,} bytes)")

    print()
    print("=" * 72)
    print("STAGE 2 CHECK  demo fact coverage")
    print("=" * 72)
    facts = report_facts(chunks)
    missing = []
    for fact, hits in facts.items():
        mark = "ok  " if hits else "MISS"
        if not hits:
            missing.append(fact)
        schemes = sorted({c.scheme for c in chunks if c.chunk_id in set(hits[:40])})
        print(f"  {mark} {fact:<16} {len(hits):>4} chunk(s)  schemes: {', '.join(schemes) or '-'}")
    if missing:
        print(f"\n  MISSING: {', '.join(missing)}")
        print("  Add an official factsheet/SID URL to data/sources.csv and re-run.")
        return 1

    print("\nAll demo facts present.")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
