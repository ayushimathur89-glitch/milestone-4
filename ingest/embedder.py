"""Shared embedding model.

implementation.md Phase 3 requires a *single shared instance* used by both
this phase and `retrieval.py` in Phase 5. That is not a style preference: if
ingestion and querying ever loaded two different model objects, or the same
model with different settings, chunk vectors and question vectors would not
be comparable and retrieval would silently return nonsense. Everything in
this project that needs a vector goes through `get_model()`.

The model is local and needs no API key (the "no API key" property from the
brief). It is downloaded once (~86 MB) and then served from the local cache.

All vectors are L2-normalised, so cosine similarity reduces to a dot product
and `config.MIN_SIMILARITY` can be compared directly against one. The store
is created in cosine space to match.

**Why ONNX Runtime and not sentence-transformers.** Both load the same weights
(`all-MiniLM-L6-v2`, 384-dim, mean pooling, L2-normalised), but they do not
cost the same memory. Measured resident set on this project's pipeline:

    101 MB   after the store is opened
    632 MB   after the first query          <- the embedding model loads here

The ~530 MB is PyTorch's runtime, not the 86 MB of weights. The app was
deployed to a free-tier host with a 512 MB limit, where that jump is fatal:
the process is OOM-killed the moment somebody asks a question. Symptom was a
page that rendered perfectly and then silently stopped answering, with no
traceback in the service logs because the kernel, not Python, ended the
process. A memory limit is not something to tune around in the app, so the
runtime itself was replaced: onnxruntime loads the same float32 export in
~150 MB, which leaves real headroom under the limit.

`onnx/model.onnx` is the *float32* export, not one of the `model_O4` /
`model_qint8_*` quantisations, and that choice is load-bearing. Quantising
would shrink the download but move every vector far enough to invalidate the
measured thresholds in `config.py`. The float32 export was checked against
torch on this project's own corpus:

    cosine(torch, onnx)  mean 1.00000000, min 0.99999988
    max abs component difference 1.7e-07
    per-question top-1 similarity and count-above-floor: identical to 4 dp

So `MIN_SIMILARITY`, `GLOBAL_ROW_BONUS`, the committed vector store in
`data/chroma/` and the test suite all carry over unchanged. A float32 kernel
difference at the seventh decimal is not a retuning event. If the export is
ever swapped for a quantised one, that stops being true and the thresholds
must be re-measured.

`onnxruntime` and `tokenizers` are not new dependencies for this project:
chromadb already requires both, which is why they are in the build log.

Thread counts are pinned to 1. On a 0.1 CPU instance ONNX Runtime otherwise
sizes its thread pools from the host's core count, which buys no speed and
costs arena memory that this budget cannot spare.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import config

_MODEL = None
_DIM: int | None = None

# ONNX Runtime input names, in the order the export declares them. Named
# rather than positional so a future export that reorders its inputs cannot
# silently feed the wrong tensor.
_INPUT_NAMES = ("input_ids", "attention_mask", "token_type_ids")

# One thread, and no arena. Both exist to save memory, not time: see the
# module docstring. 32 is enough for the largest batch used here.
_BATCH_THREADS = 1


class _OnnxMiniLm:
    """A mean-pooling, L2-normalising sentence encoder over one ONNX export.

    Mirrors just the part of the SentenceTransformer surface this project
    uses. `embedder` is the only consumer, so there is nothing to inherit
    from and no reason to depend on a framework that costs 500 MB.
    """

    def __init__(self, model_name: str, onnx_file: str) -> None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        import onnxruntime as ort

        self._model_name = model_name

        # The tokenizer is fetched from the repo root; the graph is a single
        # file inside it. Truncation is set to the same window the chunker
        # targets, so nothing is silently cut before it is stored or compared.
        self._tokenizer = Tokenizer.from_pretrained(model_name)
        self._tokenizer.enable_truncation(max_length=config.MAX_EMBED_TOKENS)
        self._tokenizer.enable_padding(length=None, pad_id=0,
                                       pad_token="[PAD]")

        path = hf_hub_download(model_name, onnx_file)
        options = ort.SessionOptions()
        options.intra_op_num_threads = _BATCH_THREADS
        options.inter_op_num_threads = _BATCH_THREADS
        options.enable_cpu_mem_arena = False
        self._session = ort.InferenceSession(
            path, sess_options=options, providers=["CPUExecutionProvider"]
        )

        # Some exports name only two inputs. Token type ids are all zeros for
        # a single-segment sentence, so asking for the names the graph actually
        # declared is more robust than assuming three.
        self._inputs = [n for n in _INPUT_NAMES
                        if n in {i.name for i in self._session.get_inputs()}]

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_embedding_dimension(self) -> int:
        """Read the width from the graph's output shape rather than assume 384."""
        shape = self._session.get_outputs()[0].shape
        # Shape may be [batch, seq, dim] or [batch, seq, None] when the export
        # left the width dynamic; config.EMBEDDING_DIM is the fallback.
        dim = shape[-1]
        return int(dim) if isinstance(dim, int) else config.EMBEDDING_DIM

    def encode(self, texts: Sequence[str], batch_size: int = 64,
               show_progress: bool = False) -> np.ndarray:
        """Encode texts to an (n, dim) L2-normalised float32 array."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.get_embedding_dimension()),
                            dtype=np.float32)

        batches = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self._tokenizer.encode_batch(batch)
            feed = {
                "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded],
                                           dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encoded],
                                           dtype=np.int64),
            }
            feed = {k: v for k, v in feed.items() if k in self._inputs}

            hidden = self._session.run(None, feed)[0]   # (b, seq, dim)
            # Mean pooling over real tokens only. Padding contributes nothing
            # because it is masked out here rather than averaged in - which is
            # why the result does not depend on how the batch was padded.
            mask = feed["attention_mask"][:, :, None].astype(np.float32)
            summed = (hidden * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), 1e-9, None)
            pooled = summed / counts

            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            batches.append((pooled / norms).astype(np.float32))
            if show_progress:
                print(f"  embedded {min(start + batch_size, len(texts))}"
                      f"/{len(texts)}", flush=True)

        return np.vstack(batches)


def get_model():
    """Return the process-wide encoder, loading it on first use."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _OnnxMiniLm(config.EMBEDDING_MODEL, config.EMBEDDING_ONNX_FILE)
        measured = _MODEL.get_embedding_dimension()
        if measured != config.EMBEDDING_DIM:
            # The store's vectors and every similarity in config.py are
            # expressed at this width, so a mismatch is a data contract
            # failure, not something to accommodate.
            raise RuntimeError(
                f"{config.EMBEDDING_ONNX_FILE} produces {measured}-dim vectors "
                f"but config.EMBEDDING_DIM is {config.EMBEDDING_DIM}"
            )
    return _MODEL


def model_name() -> str:
    return config.EMBEDDING_MODEL


def embed_dim() -> int:
    """Return the vector width, verified against config rather than assumed."""
    global _DIM
    if _DIM is None:
        _DIM = int(get_model().get_embedding_dimension())
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
        show_progress=show_progress,
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
        f"runtime       : onnxruntime, {config.EMBEDDING_ONNX_FILE} (float32)",
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