"""Stage 2 of ingestion: split extracted text into retrievable chunks.

Implements architecture.md 2.3: split on section heading, then paragraph,
then sentence, at `config.CHUNK_MAX_WORDS` words with
`config.CHUNK_OVERLAP_WORDS` overlap. Table blocks are never split, so an
exit-load slab is never cited without its column meanings.

Every chunk carries the 7 metadata fields required by Phase 3. Chroma accepts
only str/int/float/bool metadata and rejects None, so `""` is used throughout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

import config

HEADING_MAX_CHARS = 110
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\u201c])")
_PIPE_ROW = re.compile(r"\|")
_NUMERIC = re.compile(r"^[^\w]*[\d.,%()/-]+[^\w]*$")


def is_table_block(block: str) -> bool:
    """A flattened fact table must stay atomic - never split, never overlapped.

    pypdf collapses the column gaps that bs4's pipe-joined rows have, so a
    pipe test alone finds nothing in a PDF. Fact tables in these documents are
    mostly numbers, so a block also counts when its rows are three or more
    tokens of which at least two are numeric. The loader shares this rule so
    grouping and splitting can never drift apart.
    """
    if block.count("|") >= 2:
        return True
    rows = [r for r in block.split("\n") if r.strip()]
    if not rows:
        return False
    tabular_rows = 0
    for row in rows:
        tokens = row.split()
        if len(tokens) >= 3 and sum(1 for t in tokens if _NUMERIC.match(t)) >= 2:
            tabular_rows += 1
    return tabular_rows >= 2 and tabular_rows >= len(rows) / 2


@dataclass
class Chunk:
    chunk_id: str
    scheme: str
    doc_type: str
    source_url: str
    page_title: str
    section: str
    fetched_at: str
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)


def _word_count(text: str) -> int:
    return len(text.split())


def looks_like_heading(block: str) -> bool:
    if "\n" in block or is_table_block(block):
        return False
    if _word_count(block) > 16:
        return False
    if block.endswith((".", ":", ";", ",")) and not block.endswith(":"):
        return False
    stripped = block.strip()
    if not stripped:
        return False
    numbered = re.match(r"^(\d+(\.\d+)*\.?|[IVXLC]+\.|Section\s+\d+)\s+\S", stripped)
    if numbered:
        return True
    if stripped.endswith(":"):
        return True
    if stripped.isupper() and len(stripped) > 3:
        return True
    return len(stripped) <= HEADING_MAX_CHARS and not stripped.endswith(".")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_END.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _pack_sentences(sentences: list[str], max_words: int) -> list[str]:
    """Greedy sentence packing for a block that is over the word limit."""
    groups: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if _word_count(sentence) > max_words:
            # No usable sentence boundary inside it, so fall back to windows.
            if current:
                groups.append(" ".join(current))
                current = []
            groups.extend(_window_split(sentence, max_words))
            continue
        if current and _word_count(" ".join(current)) + _word_count(sentence) > max_words:
            groups.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        groups.append(" ".join(current))
    return groups


def _window_split(text: str, max_words: int) -> list[str]:
    """Last-resort split on a hard word boundary.

    SID text is largely unpunctuated runs, so sentence boundaries often do not
    exist. Without this a single unit can exceed the limit and, because the
    embedding window is 256 tokens, be silently truncated in Phase 3.
    """
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _split_table(block: str, max_words: int) -> list[str]:
    """Split an over-long table by rows, repeating the header in every part.

    architecture.md 2.3 requires a table to stay atomic so it is never cited
    without its column meanings. When one table still exceeds the limit the
    atomicity that matters - the header reaching its rows - is preserved by
    carrying the header row into each part. A row too long to share a chunk
    with a header is window-split, since a header plus a truncated row would
    be worse than losing the layout.
    """
    rows = [r for r in block.split("\n") if r.strip()]
    if len(rows) < 2:
        return _window_split(block, max_words)
    header, body = rows[0], rows[1:]
    header_words = _word_count(header)
    if header_words >= max_words:
        return _window_split(block, max_words)

    parts: list[str] = []
    current: list[str] = []
    for row in body:
        if _word_count(row) > max_words - header_words:
            if current:
                parts.append("\n".join([header, *current]))
                current = []
            for piece in _window_split(row, max_words - header_words):
                parts.append("\n".join([header, piece]))
            continue
        if current and header_words + _word_count("\n".join([*current, row])) > max_words:
            parts.append("\n".join([header, *current]))
            current = [row]
        else:
            current.append(row)
    if current:
        parts.append("\n".join([header, *current]))
    return parts


_TOKENIZER: object | None = None


def _tokenizer():
    """Load the embedding model's tokenizer once, lazily.

    The word limit alone cannot guarantee the embedding window. 160 ordinary
    words are about 200 tokens, but a 160-word run of NAV codes, dates and
    branch addresses measured 470 tokens in these SIDs, which would be
    silently truncated at embed time. config.MAX_EMBED_TOKENS exists for
    exactly this, so the ceiling is enforced with the real tokenizer.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
    return _TOKENIZER


def _token_count(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


def _token_split(text: str, max_tokens: int) -> list[str]:
    """Bisect on the real token boundary until every part fits."""
    tokenizer = _tokenizer()
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [text]
    middle = len(ids) // 2
    left = tokenizer.decode(ids[:middle], skip_special_tokens=True).strip()
    right = tokenizer.decode(ids[middle:], skip_special_tokens=True).strip()
    if not left or not right:
        return [text]
    return _token_split(left, max_tokens) + _token_split(right, max_tokens)


def _enforce_limits(text: str, max_words: int) -> list[str]:
    """Split `text` until every piece satisfies both the word and token limits.

    The two constraints interact: 160 single-token words can be 256+ tokens,
    while 256 tokens of ordinary prose can exceed 160 words. Splitting on only
    one of them therefore leaves violations of the other, so the two alternate
    until both hold. The loop is bounded because each pass strictly reduces the
    text, and a fragment that cannot be reduced further is returned as-is
    rather than dropped.
    """
    pieces = [text]
    for _ in range(6):
        violations = [
            p for p in pieces
            if _word_count(p) > max_words or _token_count(p) > config.MAX_EMBED_TOKENS
        ]
        if not violations:
            return pieces
        reduced: list[str] = []
        for piece in pieces:
            if piece not in violations:
                reduced.append(piece)
                continue
            if _word_count(piece) > max_words:
                smaller = (
                    _split_table(piece, max_words)
                    if is_table_block(piece)
                    else _window_split(piece, max_words)
                )
            else:
                smaller = _token_split(piece, config.MAX_EMBED_TOKENS)
            reduced.extend(s for s in smaller if s)
        if reduced == pieces:
            return pieces
        pieces = reduced
    return pieces


def _fit_to_limit(text: str, max_words: int) -> list[str]:
    """Return `text` as a list of pieces that each satisfy both size limits.

    This is the single choke point for the size contract. Upstream packing is
    best-effort, so the guarantee is enforced here: the word limit from
    architecture.md 2.3 and the token ceiling from config.MAX_EMBED_TOKENS,
    which keeps every chunk inside the embedding window so nothing is silently
    truncated at embed time.
    """
    return _enforce_limits(text, max_words)


def _tail_overlap(text: str, overlap_words: int) -> str:
    words = text.split()
    return " ".join(words[-overlap_words:]) if len(words) > overlap_words else ""


def _read_header(raw_text: str) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in raw_text.split("\n"):
        if line.startswith("# ") and ":" in line:
            key, _, value = line[2:].partition(":")
            header[key.strip()] = value.strip()
        elif line.startswith("-" * 10):
            break
    return header


def chunk_document(raw_path, source_url: str, start_index: int) -> list[Chunk]:
    """Chunk one `data/raw/*.txt` file. Chunk ids are stable and ordered."""
    raw_text = raw_path.read_text(encoding="utf-8")
    header = _read_header(raw_text)
    body = raw_text.split("-" * 72 + "\n", 1)[-1]

    scheme = header.get("SCHEME", "")
    doc_type = header.get("DOC_TYPE", "")
    page_title = header.get("PAGE_TITLE", "")
    fetched_at = header.get("FETCHED_AT", "")
    source_url = header.get("SOURCE_URL", source_url)

    max_words = config.CHUNK_MAX_WORDS
    overlap = config.CHUNK_OVERLAP_WORDS

    blocks = [b.strip() for b in body.split("\n\n") if b.strip()]

    # Normalise every block to a unit that already fits the limit, so the
    # packing loop below can never assemble an over-limit chunk.
    units: list[tuple[str, bool]] = []  # (text, is_table)
    for block in blocks:
        if is_table_block(block):
            if _word_count(block) <= max_words:
                units.append((block, True))
            else:
                for part in _split_table(block, max_words):
                    units.append((part, True))
            continue
        if _word_count(block) <= max_words:
            units.append((block, False))
            continue
        for group in _pack_sentences(split_sentences(block), max_words):
            if _word_count(group) <= max_words:
                units.append((group, False))
            else:
                # _pack_sentences already guarantees the limit, so this is a
                # defensive path. Emit every window rather than dropping one.
                for piece in _window_split(group, max_words):
                    units.append((piece, False))

    chunks: list[Chunk] = []
    current: list[str] = []
    carry = ""          # tail of the previous chunk, re-attached for overlap
    heading = ""        # heading awaiting the chunk it belongs to
    section = ""

    def emit(text: str, sec: str) -> None:
        text = re.sub(r"[ \t]+", " ", text).strip()
        if not text:
            return
        for piece in _fit_to_limit(text, max_words):
            index = start_index + len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{_scheme_slug(scheme)}-{index:04d}",
                    scheme=scheme,
                    doc_type=doc_type,
                    source_url=source_url,
                    page_title=page_title,
                    section=sec,
                    fetched_at=fetched_at,
                    text=piece,
                )
            )

    def render(body: str) -> str:
        """Prefix the section heading to the first chunk of that section only.

        The heading is consumed on use so a long section does not pay for it
        in every chunk, and it is dropped entirely when the result would not
        fit, because exceeding the limit is a contract failure.
        """
        nonlocal heading
        if not heading:
            return body
        candidate = f"{heading}\n{body}"
        if _word_count(candidate) <= max_words:
            heading = ""
            return candidate
        return body

    def flush() -> None:
        """Close the current chunk and remember its tail as overlap."""
        nonlocal current, carry
        if not current:
            return
        body = " ".join(current).strip()
        emit(render(body), section)
        carry = _tail_overlap(body, overlap)
        current = []

    for text, is_table in units:
        if is_table:
            # A table is self-contained: never split it, never overlap into it.
            flush()
            if heading and _word_count(f"{heading} {text}") <= max_words:
                emit(f"{heading}\n{text}", section)
            else:
                emit(text, section)
            heading = ""
            carry = ""
            continue

        if looks_like_heading(text):
            flush()
            carry = ""
            # Two headings in a row (a numbered section then its title) keep
            # both, so neither is silently dropped from the corpus.
            if heading and not current and _word_count(f"{heading} {text}") <= max_words:
                text = f"{heading} {text}"
            section = text
            heading = text
            continue

        # A chunk is closed whenever adding this unit would overflow it. The
        # overlap tail is re-seated as real text in the next chunk rather than
        # only being counted, otherwise it is measured but never emitted.
        prospective = f"{heading} {carry} {' '.join(current)} {text}".strip()
        if current and _word_count(prospective) > max_words:
            flush()
            if _word_count(f"{heading} {carry} {text}".strip()) > max_words:
                carry = ""   # unit + heading alone already fills the chunk
            elif carry:
                current = [carry]
        current.append(text)

    # A trailing heading with no body still deserves to be searchable.
    if heading and not current:
        emit(heading, section)
    flush()
    return chunks


def _scheme_slug(scheme: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", scheme.lower())).strip("-")


def chunk_all(documents) -> list[Chunk]:
    """Chunk every loaded document, numbering continuously across the corpus."""
    from pathlib import Path

    chunks: list[Chunk] = []
    for document in documents:
        raw_name = Path(document.source.raw_file).name
        raw_path = config.RAW_DIR / raw_name
        chunks.extend(chunk_document(raw_path, document.source.url, len(chunks) + 1))
    return chunks


def render_chunks_file(chunks: list[Chunk]) -> str:
    """Render the inspectable corpus. This is a graded Phase 2 deliverable."""
    out: list[str] = []
    total_chars = 0
    out.append("=" * 96)
    out.append("CHUNK CORPUS - inspectable without a vector store")
    out.append(f"chunks: {len(chunks)}")
    out.append(
        f"limits: max {config.CHUNK_MAX_WORDS} words / "
        f"{config.CHUNK_OVERLAP_WORDS} overlap, embedding model "
        f"{config.EMBEDDING_MODEL} ({config.EMBEDDING_DIM}-dim)"
    )
    out.append("=" * 96)

    for number, chunk in enumerate(chunks, start=1):
        total_chars += chunk.char_count
        meta = asdict(chunk)
        out.append("")
        out.append("-" * 96)
        out.append(f"CHUNK {number} of {len(chunks)}")
        out.append("-" * 96)
        out.append(f"  chunk_id     : {meta['chunk_id']}")
        out.append(f"  scheme       : {meta['scheme']}")
        out.append(f"  doc_type     : {meta['doc_type']}")
        out.append(f"  source_url   : {meta['source_url']}")
        out.append(f"  page_title   : {meta['page_title']}")
        out.append(f"  section      : {meta['section']}")
        out.append(f"  fetched_at   : {meta['fetched_at']}")
        out.append(f"  word_count   : {chunk.word_count}")
        out.append(f"  char_count   : {chunk.char_count}")
        out.append(f"  total_chars  : {total_chars}")
        out.append("  --- text ---")
        for line in chunk.text.split("\n"):
            out.append(f"  | {line}")
        out.append("")
    return "\n".join(out) + "\n"


def validate(chunks: list[Chunk]) -> list[str]:
    """Check the Phase 2 contract. Returns a list of human-readable problems."""
    problems: list[str] = []
    required = ("chunk_id", "scheme", "doc_type", "source_url", "page_title", "section", "fetched_at")

    for number, chunk in enumerate(chunks, start=1):
        if chunk.word_count > config.CHUNK_MAX_WORDS:
            problems.append(f"chunk {number} ({chunk.chunk_id}) is {chunk.word_count} words")
        for field in required:
            value = getattr(chunk, field)
            if value is None:
                problems.append(f"chunk {number} field {field} is None")
            elif not isinstance(value, str):
                problems.append(f"chunk {number} field {field} is {type(value).__name__}, not str")
        if not config.is_allowed_url(chunk.source_url):
            problems.append(f"chunk {number} source_url off allow-list: {chunk.source_url}")

    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        problems.append("duplicate chunk_id values present")
    return problems


def write_chunks_file(chunks: list[Chunk]) -> "object":
    path = config.CHUNKS_TXT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_chunks_file(chunks), encoding="utf-8")
    return path
