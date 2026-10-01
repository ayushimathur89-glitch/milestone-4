"""Ingestion entry point.

Runs the full pipeline: load, chunk, embed, store.

    .venv\\Scripts\\python -m ingest.run_ingestion
    .venv\\Scripts\\python -m ingest.run_ingestion --limit 2
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import config
from ingest import chunker, embedder, loader, store

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


def sanity_check(collection, chunks, vectors) -> bool:
    """implementation.md Phase 3 step 3: shapes are 384-dim and ranking works.

    A store full of 384-dim vectors proves only that shapes line up. What
    actually matters is that a question about a real topic retrieves its own
    kind of chunk, and ranks it above an unrelated question's result. Cosine
    similarity is the right measure because `embedder` normalises, so the
    dot product is the cosine.
    """
    sample = vectors[0]
    print(f"  chunk vector shape  : {tuple(sample.shape)}")
    if tuple(sample.shape) != (config.EMBEDDING_DIM,):
        print(f"  Expected ({config.EMBEDDING_DIM},); vectors are the wrong width.")
        return False

    relevant = "What is the exit load on the SBI Flexicap Fund?"
    unrelated = "How do I bake sourdough bread at high altitude?"
    q_relevant = embedder.embed_query(relevant)
    q_unrelated = embedder.embed_query(unrelated)
    print(f"  question vector shape: {tuple(q_relevant.shape)}")
    if tuple(q_relevant.shape) != (config.EMBEDDING_DIM,):
        return False

    def top(question: str) -> str:
        result = collection.query(
            query_embeddings=[embedder.embed_query(question).tolist()],
            n_results=1,
            include=["documents", "distances"],
        )
        return result["documents"][0][0], result["distances"][0][0]

    best_doc, best_distance = top(relevant)
    worst_doc, worst_distance = top(unrelated)
    # Cosine space: distance == 1 - cosine similarity.
    best_sim = 1 - best_distance
    worst_sim = 1 - worst_distance
    print(f"  top hit for a real question  : cos {best_sim:+.3f} (dist {best_distance:.3f})")
    print(f"     {best_doc[:96]}")
    print(f"  top hit for an unrelated one  : cos {worst_sim:+.3f} (dist {worst_distance:.3f})")
    print(f"     {worst_doc[:96]}")
    if best_sim <= worst_sim:
        print("  A finance question did not outrank an unrelated one; ranking is broken.")
        return False
    if best_sim < config.MIN_SIMILARITY:
        print(f"  Best cosine {best_sim:.3f} is below config.MIN_SIMILARITY "
              f"{config.MIN_SIMILARITY}; Phase 5 would reject every chunk.")
        return False

    print(f"  ranking ok, and above config.MIN_SIMILARITY ({config.MIN_SIMILARITY})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ingestion pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="only load the first N sources")
    parser.add_argument("--verbose", action="store_true",
                        help="print each source as it loads, and each dropped block")
    args = parser.parse_args()

    print("=" * 72)
    print("STAGE 1  LOAD")
    print("=" * 72)
    documents, results = loader.run_load(limit=args.limit, verbose=args.verbose)
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

    # After every page is in hand, because "this text is on every page" is only
    # knowable once more than one page has been seen.
    dropped = loader.drop_shared_boilerplate(documents, verbose=args.verbose)
    print(f"  site furniture shared across pages, dropped: {dropped}")

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

    print()
    print("=" * 72)
    print("STAGE 3  EMBED")
    print("=" * 72)
    started = time.perf_counter()
    vectors = embedder.embed_texts([c.text for c in chunks], show_progress=False)
    elapsed = time.perf_counter() - started
    dim = embedder.embed_dim()
    print(f"  model       : {config.EMBEDDING_MODEL}")
    print(f"  loaded      : {dim}-dim" if dim == config.EMBEDDING_DIM
          else f"  loaded      : {dim}-dim (config says {config.EMBEDDING_DIM}!)")
    print(f"  vectors     : {len(vectors):,} x {dim} in {elapsed:.1f}s"
          f"  ({len(vectors) / max(elapsed, 1e-9):.1f} chunks/s, CPU)")
    if dim != config.EMBEDDING_DIM:
        print("  Dimension mismatch with config.EMBEDDING_DIM; fix before trusting retrieval.")
        return 1
    if len(vectors) != len(chunks):
        print(f"  Vector count {len(vectors)} != chunk count {len(chunks)}.")
        return 1

    preview = embedder.render_preview(vectors, chunks)
    config.EMBEDDINGS_PREVIEW_TXT.write_text(preview, encoding="utf-8")
    print(f"  preview     -> {config.EMBEDDINGS_PREVIEW_TXT}")

    print()
    print("=" * 72)
    print("STAGE 4  STORE")
    print("=" * 72)
    store.reset_store()
    client = store.get_client()
    collection = store.replace_collection(client)
    stored = store.store_chunks(collection, chunks, vectors)
    size = store.collection_size(collection)
    print(f"  collection  : {config.COLLECTION_NAME}  (cosine space)")
    print(f"  vectors     : {stored:,} written")
    print(f"  count()     : {size:,}")
    print(f"  persisted   -> {config.CHROMA_DIR}"
          f"  ({store.disk_usage() / 1_048_576:.1f} MiB on disk)")
    if size != len(chunks):
        print(f"  Count {size:,} != chunk count {len(chunks):,}; store is inconsistent.")
        return 1

    print()
    print("=" * 72)
    print("STAGE 4 CHECK  embedding sanity")
    print("=" * 72)
    if not sanity_check(collection, chunks, vectors):
        return 1

    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
