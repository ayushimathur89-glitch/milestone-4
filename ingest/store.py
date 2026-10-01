"""ChromaDB persistent vector store.

implementation.md Phase 3: a `PersistentClient` rooted at `data/chroma/`
with one collection, chunk text as the document, the 7 metadata fields as
the metadata, and the embedding stored alongside.

Two decisions worth stating because they bite later:

1. **Cosine space, not the default L2.** `embedder` L2-normalises every
   vector, and `config.MIN_SIMILARITY` is expressed as a cosine similarity.
   With cosine space, Chroma's distance is `1 - cosine_similarity`, so Phase 5
   converts a similarity floor into a distance ceiling once instead of
   scattering the conversion through the query path.

2. **Ingestion replaces the collection.** Re-running the pipeline must not
   append to whatever is already there, or the store grows every run and
   Phase 5 quietly returns stale duplicates. `replace_collection` drops the
   old collection and creates a fresh one, so a run's contents are a pure
   function of that run's sources.

The directory is committed to git. Render's free tier has an ephemeral
filesystem, so the store must be in the repo: `app.py` opens the collection
directly and `open_collection` raises if it is absent, so there is no runtime
rebuild path. See Docs/architecture.md section 4.
"""

from __future__ import annotations

import shutil

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection

import config

_CLIENT = None
_BATCH = 1000


def get_client():
    """Return the process-wide PersistentClient rooted at data/chroma/."""
    global _CLIENT
    if _CLIENT is None:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _CLIENT = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _CLIENT


def reset_store() -> None:
    """Wipe data/chroma/ so the next run rebuilds it from scratch.

    `delete_collection` is not enough. It leaves the collection's HNSW index
    directory on disk and never returns the freed pages in `chroma.sqlite3`,
    so re-running ingestion twice left 24.6 MiB for 2,685 vectors, including a
    whole orphaned index directory for a collection that no longer existed.
    Since ingestion always replaces the only collection and the store is a
    pure function of the sources, wiping the directory is equivalent to
    deleting the collection and leaves no residue to commit.

    `.gitkeep` is restored because the directory is tracked in git.
    """
    global _CLIENT
    _CLIENT = None
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    (config.CHROMA_DIR / ".gitkeep").touch()


def collection_names(client) -> set[str]:
    """Names of existing collections, tolerating both return shapes."""
    names: set[str] = set()
    for entry in client.list_collections():
        names.add(entry if isinstance(entry, str) else entry.name)
    return names


def replace_collection(client) -> Collection:
    """Drop any existing collection and create an empty one in its place.

    This is what makes ingestion idempotent. It is deliberately a replace
    rather than an upsert: chunk ids are stable for a given corpus, but the
    text and vectors behind an id can change when a source PDF is reissued,
    and a partial upsert would mix new vectors with stale ones.
    """
    name = config.COLLECTION_NAME
    if name in collection_names(client):
        client.delete_collection(name)
    return client.create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_dim": config.EMBEDDING_DIM,
        },
    )


def open_collection(client) -> Collection:
    """Open the existing collection for querying. Used by Phase 5."""
    if config.COLLECTION_NAME not in collection_names(client):
        raise RuntimeError(
            f"No collection {config.COLLECTION_NAME!r} in {config.CHROMA_DIR}. "
            "Run: .venv\\Scripts\\python -m ingest.run_ingestion"
        )
    return client.get_collection(config.COLLECTION_NAME)


def store_chunks(collection: Collection, chunks, vectors) -> int:
    """Write chunks and their vectors to the collection. Returns the count."""
    for start in range(0, len(chunks), _BATCH):
        stop = min(start + _BATCH, len(chunks))
        collection.add(
            ids=[c.chunk_id for c in chunks[start:stop]],
            documents=[c.text for c in chunks[start:stop]],
            metadatas=[
                {
                    "chunk_id": c.chunk_id,
                    "scheme": c.scheme,
                    "doc_type": c.doc_type,
                    "source_url": c.source_url,
                    "page_title": c.page_title,
                    "section": c.section,
                    "fetched_at": c.fetched_at,
                }
                for c in chunks[start:stop]
            ],
            embeddings=vectors[start:stop].tolist(),
        )
    return len(chunks)


def collection_size(collection: Collection) -> int:
    return collection.count()


def disk_usage() -> int:
    """Total bytes under data/chroma/, for the commit-size report."""
    if not config.CHROMA_DIR.exists():
        return 0
    return sum(p.stat().st_size for p in config.CHROMA_DIR.rglob("*") if p.is_file())
