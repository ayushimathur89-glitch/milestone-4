"""Shared embedding model.

implementation.md Phase 3 requires a *single shared instance* used by both
this phase and `retrieval.py` in Phase 5. That is not a style preference: if
ingestion and querying ever loaded two different model objects, or the same
model with different settings, chunk vectors and question vectors would not
be comparable and retrieval would silently return nonsense. Everything in
this project that needs a vector goes through `get_model()`.

The model is local and needs no API key (the "no API key" property from the
brief). It is downloaded once (~90 MB) and then served from the local cache.

All vectors are L2-normalised, so cosine similarity reduces to a dot product
and `config.MIN_SIMILARITY` can be compared directly against one. The store
is created in cosine space to match.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import config

_MODEL = None
_DIM: int | None = None


def get_model():
    """Return the process-wide SentenceTransformer, loading it on first use."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        # Pinned to CPU deliberately: this project is deployed to a free-tier
        # host with no GPU, so a CUDA-pinned default would be a latent bug.
        _MODEL = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    return _MODEL


def model_name() -> str:
    return config.EMBEDDING_MODEL


def embed_dim() -> int:
    """Return the vector width, verified against config rather than assumed."""
    global _DIM
    if _DIM is None:
        model = get_model()
        # sentence-transformers 6.x renamed this to get_embedding_dimension.
        getter = getattr(model, "get_embedding_dimension", None) or \
            model.get_sentence_embedding_dimension
        _DIM = int(getter())
    return _DIM


def embed_texts(texts: Sequence[str], batch_size: int = 64,
                show_progress: bool = False) -> np.ndarray:
    """Embed a batch of texts into an (n, dim) L2-normalised float32 array.

    Chunker already guarantees every chunk is within the model's token window,
    so nothing here is silently truncated. `encode` would warn and cut
    anything longer, which is a data-loss failure mode, not a cosmetic one.
    """
    if not texts:
        return np.zeros((0, embed_dim()), dtype=np.float32)
    vectors = get_model().encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(question: str) -> np.ndarray:
    """Embed a single question. Shares the model, so the spaces match."""
    return embed_texts([question])[0]


def render_preview(vectors: np.ndarray, chunks, dims: int = 10,
                   sample: int = 5) -> str:
    """Build a human-readable preview of the first few stored vectors.

    Phase 3 asks for a preview of the first 5 embeddings, first 10
    dimensions each. Raw floats are uninformative to read, so each row is
    paired with the chunk it came from and its L2 norm, which makes the
    normalisation checkable by eye.
    """
    shown = min(sample, len(vectors))
    lines = [
        "Embedding preview",
        "=" * 72,
        f"model         : {config.EMBEDDING_MODEL}",
        f"dimensions    : {vectors.shape[1] if vectors.ndim == 2 else 0}",
        f"shown         : first {shown} vector(s), first {dims} dimension(s) each",
        f"normalised    : yes, L2 unit length, so cosine similarity = dot product",
        "",
        "Signs and magnitudes vary per dimension; what matters is that every",
        "vector has norm 1.0 and that same-dimension columns are comparable",
        "across chunks.",
        "=" * 72,
        "",
    ]
    for i in range(shown):
        vector = vectors[i]
        head = vector[:dims]
        values = "  ".join(f"{v:+.4f}" for v in head)
        lines.append(f"[{i}] {chunks[i].chunk_id}  ({chunks[i].scheme})")
        lines.append(f"    dims 0-{dims - 1}: {values}")
        lines.append(f"    full dim norm    : {float(np.linalg.norm(vector)):.4f}")
        text = " ".join(chunks[i].text.split())
        lines.append(f"    text             : {text[:100]}{'...' if len(text) > 100 else ''}")
        lines.append("")
    return "\n".join(lines)
