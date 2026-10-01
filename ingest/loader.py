"""Stage 1 of ingestion: fetch each allowed source URL and save clean text.

Reads `data/sources.csv` (hand-curated, never rewritten by this module) and
writes one readable `.txt` per source into `data/raw/`, plus a run report in
`data/ingest_manifest.csv`.

Official sources only. `config.is_allowed_url` is the gate: a URL whose domain
is not in `config.ALLOWED_DOMAINS` is rejected before any request is made, so
a typo in the CSV cannot turn this into a general-purpose crawler.
"""

from __future__ import annotations

import csv
import io
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

import config
from ingest.chunker import is_table_block

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
TIMEOUT = 60
RETRIES = 3
BACKOFF = 2.0
MIN_TEXT_CHARS = 400

# Chrome/nav/footer scaffolding. Matched against tag names and class/id
# attributes, because SBI MF marks chrome with utility classes rather than
# semantic tags, and a bare <footer> check would miss most of it.
_CHROME_TAGS = ("script", "style", "noscript", "template", "svg", "iframe", "form", "button")
_CHROME_CLASS = re.compile(
    r"(cookie|consent|gdpr|navbar|nav-|menu|header|footer|sidebar|breadcrumb|"
    r"chat|chatbot|search|social|newsletter|subscribe|promo|popup|modal|"
    r"skip-|screen-reader|sr-only|accessib|back-to-top|scroll)",
    re.I,
)
# Lines that are pure navigation furniture once whitespace is collapsed.
_JUNK_LINE = re.compile(
    r"^(invest now|know more|click here|read more|view all|apply now|"
    r"disclaimer|mutual fund investments are subject to market risks.*|"
    r"join us|for distributor|english|hindi|skip to|contact us|"
    r"\+?\d[\d\s\-()]{7,}|www\.[\w.]+|copyright.*|Â©.*)$",
    re.I,
)
# Whole blocks that are page furniture rather than content. The campaign pages
# hydrate parts of themselves with JavaScript, so a "Loading..." placeholder
# and a lead-capture form survive every structural filter and would otherwise
# be embedded and later retrieved as though they were facts about a fund.
_PLACEHOLDER_BLOCK = re.compile(
    r"^(loading\.{0,3}|please wait\.{0,3}|"
    r"let'?s connect\b.*|fill in the (below )?form.*|"
    r"get a call (back )?from (one of )?our experts?.*|"
    r"submit|search|menu|close|open|next|previous|back to top|"
    r"skip to (main )?content)\s*$",
    re.I,
)
# Calls to action welded onto the end of real copy, e.g. a heading block
# reading "... MIX OF INVESTMENTS Invest Now". Stripping the CTA keeps the
# sentence instead of discarding the whole block with it.
_TRAILING_CTA = re.compile(
    r"\s+(invest now|apply now|view all|read more|click here|disclaimers?|"
    r"know more( about (our|this|the)[^.]*)?)\s*$",
    re.I,
)


@dataclass
class LoadResult:
    """One row of `data/ingest_manifest.csv`."""

    url: str
    scheme: str
    doc_type: str
    kind: str = ""
    http_status: int = 0
    bytes_downloaded: int = 0
    text_chars: int = 0
    pages: int = 0
    fetch_seconds: float = 0.0
    fetched_at: str = ""
    raw_file: str = ""
    status: str = "error"
    error: str = ""


@dataclass
class LoadedDocument:
    """A successfully extracted document, ready for the chunker."""

    source: LoadResult
    title: str
    blocks: list[str] = field(default_factory=list)


def _decode_html(content: bytes) -> str:
    """Decode HTML, repairing cp1252 bytes that leaked into a utf-8 document.

    SBI MF's CMS emits raw Windows-1252 bytes (0x96 for an en-dash, 0x92 for
    an apostrophe) inside documents served as utf-8. A strict decode raises
    and a lenient one turns every such character into U+FFFD, so a range like
    "1st - 100th" silently loses its dash. Decoding the invalid bytes as
    cp1252 recovers the original character.
    """
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        pass
    text = content.decode("utf-8", errors="surrogateescape")
    repaired = []
    for char in text:
        code = ord(char)
        if 0xDC80 <= code <= 0xDCFF:
            repaired.append(bytes([code - 0xDC00]).decode("cp1252", errors="replace"))
        else:
            repaired.append(char)
    return "".join(repaired)


def _is_chrome(tag) -> bool:
    if tag.name in _CHROME_TAGS:
        return True
    # class is a list in bs4, id/role are strings, so coerce before joining.
    parts: list[str] = []
    for attribute in ("class", "id", "role"):
        value = tag.get(attribute)
        if not value:
            continue
        parts.append(" ".join(value) if isinstance(value, (list, tuple)) else str(value))
    ident = " ".join(parts)
    if ident and _CHROME_CLASS.search(ident):
        return True
    if tag.name == "a" and not tag.get_text(strip=True):
        return True
    return False


# Only ever strip script-like tags and the structural page furniture. Used as
# the fallback when the class-based filter turns out to be over-matching.
_ALWAYS_CHROME = ("script", "style", "noscript", "template", "svg", "iframe")


def _is_structural_chrome(tag) -> bool:
    return tag.name in _ALWAYS_CHROME or tag.name in ("header", "footer", "nav")


# Word's Symbol font is mapped into the Unicode private use area, so bullets
# and arrows arrive as U+F0B7 and friends. They are meaningless to an embedding
# model and cannot be encoded to a cp1252 console, so list markers are folded
# to "-" and the rest are dropped.
_SYMBOL_MAP = {
    "\uf0b7": "-", "\uf0a7": "-", "\uf0a8": "-", "\uf0d8": "-", "\uf0fc": "-",
    "\uf0e0": "-", "\uf0a1": "-", "\uf06f": "-", "\uf075": "-", "\uf0b0": "-",
}
_PRIVATE_USE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd]")


def _clean_line(line: str) -> str:
    line = line.replace("\xa0", " ").replace("\u2013", "-").replace("\u2014", "-")
    for symbol, replacement in _SYMBOL_MAP.items():
        line = line.replace(symbol, replacement)
    line = _PRIVATE_USE.sub("", line)
    line = re.sub(r"[ \t\u200b]+", " ", line)
    line = re.sub(r" {2,}", " ", line)
    return line.strip()


def _dedupe_blank_runs(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if not line and (not out or not out[-1]):
            continue
        out.append(line)
    while out and not out[-1]:
        out.pop()
    return out


def _blocks_from_soup(soup, predicate) -> list[str]:
    """Strip `predicate`-matched tags, then flatten what remains into blocks."""
    for tag in soup.find_all(predicate):
        tag.decompose()

    root = None
    # The first <main>/<article> is not reliably the content one: SBI MF has
    # empty <article class="video-wrapper"> shells that would otherwise win and
    # silently drop the whole page. Pick the first candidate with real text.
    for candidate in (soup.find("main"), soup.find("article"), soup.body, soup):
        if candidate is not None and len(candidate.get_text(" ", strip=True)) >= 200:
            root = candidate
            break
    root = root if root is not None else (soup.body or soup)
    blocks: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        text = _TRAILING_CTA.sub("", " ".join(x for x in buffer if x).strip()).strip()
        if text and not _JUNK_LINE.match(text) and not _PLACEHOLDER_BLOCK.match(text):
            blocks.append(text)
        buffer.clear()

    for el in root.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "br", "div", "section"], recursive=True
    ):
        if el.name == "br":
            buffer.append("\n")
            continue
        if el.name == "tr":
            cells = [
                _clean_line(c.get_text(" ", strip=True))
                for c in el.find_all(["th", "td"], recursive=False)
            ]
            if any(cells):
                buffer.append(" | ".join(cells) + "\n")
            continue
        if el.name in ("div", "section") and el.find(["p", "li", "tr", "h2", "h3", "h4"]):
            continue
        if el.name.startswith("h") and len(el.name) == 2:
            flush()
            heading = _clean_line(el.get_text(" ", strip=True))
            if heading and not _JUNK_LINE.match(heading):
                blocks.append(heading)
            continue
        text = _clean_line(el.get_text(" ", strip=True))
        if text:
            buffer.append(text + " ")
    flush()
    return [b for b in _dedupe_blank_runs(blocks) if b]


# Below this share of the original text, the class-based filter is judged to
# have eaten real content rather than page furniture.
_OVERFILTER_RATIO = 0.4


def extract_from_html(content: bytes) -> tuple[str, list[str]]:
    """Return (page_title, blocks) for an HTML page.

    Tables are flattened to pipe-delimited rows so a fee slab keeps its header
    and its data together in one block rather than becoming loose sentences.

    The class-based chrome filter is run first, but SBI MF wraps real content
    in utility classes that the filter can match, which once left a 1,965-char
    page down to 13 characters. If the aggressive pass throws away more than
    60% of the visible text, it is redone with structural tags only and the
    larger result wins.
    """
    html = _decode_html(content)
    probe = BeautifulSoup(html, "html.parser")
    title = _clean_line(probe.title.get_text() if probe.title else "")
    original = len(probe.get_text(" ", strip=True))

    blocks = _blocks_from_soup(BeautifulSoup(html, "html.parser"), _is_chrome)
    kept = sum(len(b) for b in blocks)
    if original and kept < original * _OVERFILTER_RATIO:
        fallback = _blocks_from_soup(BeautifulSoup(html, "html.parser"), _is_structural_chrome)
        if sum(len(b) for b in fallback) > kept:
            blocks = fallback

    return title, blocks


def _is_table_line(line: str) -> bool:
    """A single table row, using the same rule as the chunker.

    Kept as a thin wrapper so grouping in the loader and splitting in the
    chunker can never disagree about what counts as a table.
    """
    return is_table_block(line)


def extract_from_pdf(content: bytes) -> tuple[str, list[str]]:
    """Return (page_title, blocks) for a PDF, using its text layer.

    pypdf emits one string per page with layout whitespace preserved. Runs of
    lines that look tabular are grouped so a header row stays attached to its
    rows; everything else is reflowed into paragraphs.
    """
    reader = PdfReader(io.BytesIO(content))
    raw_blocks: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        for line in text.split("\n"):
            line = _clean_line(line)
            if not line:
                if raw_blocks and raw_blocks[-1] != "":
                    raw_blocks.append("")
                continue
            raw_blocks.append(line)

    # Group consecutive table-looking lines into one atomic block.
    blocks: list[str] = []
    pending: list[str] = []
    for line in raw_blocks:
        if line == "":
            if pending:
                blocks.append("\n".join(pending))
                pending = []
            blocks.append("")
            continue
        if _is_table_line(line):
            pending.append(line)
        else:
            if pending:
                blocks.append("\n".join(pending))
                pending = []
            blocks.append(line)
    if pending:
        blocks.append("\n".join(pending))

    title = ""
    for b in blocks:
        if b and not b.startswith(" ") and len(b) < 120 and b.upper() == b and re.search(r"[A-Z]", b):
            title = b.strip()
            break
    if not title:
        for b in blocks:
            if b.strip():
                title = b.strip().split("\n")[0][:120]
                break

    out: list[str] = []
    para: list[str] = []
    for b in blocks:
        if b == "":
            if para:
                text = re.sub(r"\s+", " ", " ".join(para)).strip()
                if text and not _JUNK_LINE.match(text):
                    out.append(text)
                para = []
            continue
        if _is_table_line(b) or "\n" in b:
            if para:
                text = re.sub(r"\s+", " ", " ".join(para)).strip()
                if text:
                    out.append(text)
                para = []
            if b.strip():
                out.append(b.strip())
            continue
        para.append(b.strip())
    if para:
        text = re.sub(r"\s+", " ", " ".join(para)).strip()
        if text:
            out.append(text)

    return title, [b for b in out if b.strip()]


def _slug(value: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.lower())).strip("-")


def fetch(url: str) -> tuple[requests.Response, float]:
    """GET with a small retry budget. Returns the response and elapsed seconds."""
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        started = time.monotonic()
        try:
            response = requests.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, allow_redirects=True
            )
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = requests.HTTPError(f"HTTP {response.status_code}")
            else:
                return response, time.monotonic() - started
        except requests.RequestException as exc:
            last_error = exc
        if attempt < RETRIES:
            time.sleep(BACKOFF * attempt)
    raise last_error if last_error else RuntimeError("unreachable")


def read_sources() -> list[dict[str, str]]:
    with config.SOURCES_CSV.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# SBI MF serves its custom 404 page with a 200 status, so status alone cannot
# tell a dead URL from a live one. The 404 template keeps the site chrome, so
# it yields text - just none of it about the scheme.
_SOFT_404_TITLE = re.compile(r"\b(404|page not found|not found|error)\b", re.I)
_SOFT_404_BODY = re.compile(
    r"(page (you|that) (are )?(looking for|requested).{0,60}(not (found|exist))"
    r"|the page you.{0,40}(not|couldn.{0,3}t be) (be )?found"
    r"|sorry.{0,40}(not found|doesn.{0,3}t exist)"
    r"|we can.{0,20}t find)",
    re.I | re.S,
)


def _reject_soft_404(url: str, title: str, blocks: list[str]) -> None:
    """Fail loudly when a live 200 is actually the site's error page.

    The campaign URLs in the problem statement are stale: most return 404, and
    the remainder return the 404 template with a 200. Ingesting either would
    quietly put site chrome into the corpus and look like a successful load.
    """
    head = " ".join(blocks[:12])
    if _SOFT_404_BODY.search(head) or _SOFT_404_TITLE.search(title or ""):
        raise ValueError(
            f"soft 404: page title {title!r} and no scheme content - URL is stale"
        )


def load_one(row: dict[str, str]) -> LoadedDocument:
    """Fetch and extract a single source. Raises on network/parse failure."""
    url = row["url"].strip()
    if not config.is_allowed_url(url):
        raise ValueError(f"domain not in ALLOWED_DOMAINS: {url}")

    response, elapsed = fetch(url)
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if response.status_code >= 400:
        raise ValueError(f"HTTP {response.status_code}")

    content_type = response.headers.get("Content-Type", "").lower()
    is_pdf = "application/pdf" in content_type or response.content[:5] == b"%PDF-"

    if is_pdf:
        title, blocks = extract_from_pdf(response.content)
        kind, pages = "pdf", len(PdfReader(io.BytesIO(response.content)).pages)
    else:
        title, blocks = extract_from_html(response.content)
        kind, pages = "html", 1

    _reject_soft_404(url, title, blocks)

    body = "\n\n".join(blocks)
    if len(body) < MIN_TEXT_CHARS:
        raise ValueError(
            f"only {len(body)} chars of text - page is likely JS-rendered or empty"
        )

    raw_name = f"{_slug(row['scheme'])}__{_slug(row['doc_type'])}.txt"
    raw_path = config.RAW_DIR / raw_name
    header = (
        f"# SOURCE_URL: {url}\n"
        f"# SCHEME: {row['scheme']}\n"
        f"# DOC_TYPE: {row['doc_type']}\n"
        f"# PAGE_TITLE: {title}\n"
        f"# FETCHED_AT: {fetched_at}\n"
        f"# EXTRACTED_CHARS: {len(body)}\n"
        f"{'-' * 72}\n"
    )
    raw_path.write_text(header + body + "\n", encoding="utf-8")

    source = LoadResult(
        url=url,
        scheme=row["scheme"],
        doc_type=row["doc_type"],
        kind=kind,
        http_status=response.status_code,
        bytes_downloaded=len(response.content),
        text_chars=len(body),
        pages=pages,
        fetch_seconds=round(elapsed, 2),
        fetched_at=fetched_at,
        raw_file=f"data/raw/{raw_name}",
        status="ok",
    )
    return LoadedDocument(source=source, title=title, blocks=blocks)


def _fingerprint(text: str) -> str:
    """A whitespace- and case-insensitive fingerprint of a block's words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(words)


# How many consecutive words make one shingle, and the two thresholds a block
# must clear to be called furniture. Both were set by measuring every HTML
# block in this corpus, not by taste; the numbers quoted in the docstrings
# below are the measured scores.
_SHINGLE_WORDS = 12
_FURNITURE_COVERAGE = 0.50

# Words that name site furniture rather than a fund. Deliberately generic
# navigation vocabulary and nothing scheme-specific, so it cannot match a
# sentence that happens to be about a fund.
_CHROME_VOCABULARY = frozenset("""
about corporate profile fund house expertise people desk registrars careers
csr forms downloads factsheets portfolios statement additional information
offer document sid kim tenders notices addendums tax reckoner expense ratio
navs alert idcw history esg disclosures valuation debt securities home menu
loading contact investor existing governance annual report board directors
auditor managerial personnel prescribed
""".split())
_CHROME_VOCABULARY_SHARE = 0.35
_SELF_REPEAT_SHARE = 0.50


def _boilerplate_counts(documents: list[LoadedDocument]) -> set[str]:
    """The leading shingles of blocks that are site furniture, not content.

    Site furniture is the navigation list, the breadcrumb trail, the footer and
    the "Loading..." shells. None of it is tagged as chrome, so the structural
    filters above cannot see it, and it survives into the corpus as content.

    That is not merely untidy. A nav block scores a middling dense similarity
    against any question, because it is built from the same words as every page
    on the site, and it then competes for one of the six context slots a real
    answer needs. Measured on "Where can I find the KIM/SID for SBI Balanced
    Advantage Fund?", two such blocks took slots 3 and 4 and pushed the
    load-structure chunk down the context the model actually reads.

    Two independent tests, because furniture hides from either one alone.

    **Shared shingles** catch text that appears on more than one page - the
    breadcrumb trail, the repeated CTA. The shingle is the first twelve words
    rather than the whole block, so a content block that opens with a phrase
    the site also uses elsewhere is not lost; only the prefix is matched, and
    the block is kept when the prefix is not shared.

    **Self-repetition plus chrome vocabulary** catch furniture confined to one
    page. The nav on `ways-to-invest` renders its whole menu in the header and
    again in the mobile drawer, inside a single block, so it appears exactly
    once to any cross-page test and survived the first version of this function
    intact. Both signals are required together, and the separation is wide
    enough to measure rather than argue: on this corpus the nav scores 0.78
    self-repeated and 0.53 chrome vocabulary, while the most repetitive genuine
    content - the ELSS campaign page, which restates its own lock-in sentence in
    a summary and again in the body - scores 1.00 and 0.06. Repetition alone
    would have deleted a real answer, and vocabulary alone would have caught
    prose that happens to discuss fund categories.
    """
    counts: dict[str, int] = {}
    repeated_here: set[str] = set()
    for document in documents:
        seen_here: set[str] = set()
        for block in document.blocks:
            words = _fingerprint(block).split()
            if not words:
                continue
            if len(words) < 6:
                continue
            if _is_self_repeating_furniture(words):
                shingle = " ".join(words[:_SHINGLE_WORDS])
                repeated_here.add(shingle)
                continue
            shingle = " ".join(words[:_SHINGLE_WORDS])
            if len(shingle.split()) < 6:
                continue
            if shingle in seen_here:
                repeated_here.add(shingle)
                continue
            seen_here.add(shingle)
            counts[shingle] = counts.get(shingle, 0) + 1

    return {k for k, v in counts.items() if v > 1} | repeated_here


def _is_self_repeating_furniture(words: list[str]) -> bool:
    """Whether a block repeats itself *and* is built from navigation words.

    Both conditions, because each is wrong on its own. The ELSS campaign page
    repeats a 12-word run across its whole block - measured share 1.00 - and is
    entirely genuine content, so repetition alone cannot be the test. The nav
    block repeats at 0.78 and carries 0.53 of its words from
    `_CHROME_VOCABULARY`; the next-most-chrome real block on the site is at
    0.15. Requiring both puts the boundary in a wide gap rather than on a
    cliff.
    """
    if len(words) < _SHINGLE_WORDS + 1:
        return False
    shared = sum(1 for word in words if word in _CHROME_VOCABULARY)
    if shared / len(words) < _CHROME_VOCABULARY_SHARE:
        return False
    return _repeat_share(words) >= _SELF_REPEAT_SHARE


def _repeat_share(words: list[str]) -> float:
    """Fraction of a block's words that fall inside a repeated shingle.

    Every window is scanned, not just the leading one, because a rendered menu
    does not have to start with the thing it repeats.
    """
    seen: dict[tuple[str, ...], int] = {}
    repeats: set[tuple[str, ...]] = set()
    for start in range(len(words) - _SHINGLE_WORDS + 1):
        window = tuple(words[start:start + _SHINGLE_WORDS])
        if window in seen:
            repeats.add(window)
        seen[window] = seen.get(window, 0) + 1
    if not repeats:
        return 0.0
    # Both copies count, not just the later one: a menu printed twice makes the
    # whole block furniture, and counting only the second copy would score it
    # at roughly half of what it is.
    covered: set[int] = set()
    for start in range(len(words) - _SHINGLE_WORDS + 1):
        if tuple(words[start:start + _SHINGLE_WORDS]) in repeats:
            covered.update(range(start, start + _SHINGLE_WORDS))
    return len(covered) / len(words)


def drop_shared_boilerplate(documents: list[LoadedDocument],
                            verbose: bool = False) -> int:
    """Remove site furniture shared across pages. Returns blocks dropped.

    Only the shared prefix is cut, never the whole block, because a content
    block can open with a phrase the site repeats and then continue into
    something specific. HTML only: the PDFs here are SIDs and factsheets, where
    a repeated paragraph is a genuine regulatory notice rather than navigation,
    and losing one would be a real loss.
    """
    shingles = _boilerplate_counts(
        [d for d in documents if d.source.kind == "html"]
    )
    if not shingles:
        return 0

    dropped = 0
    for document in documents:
        if document.source.kind != "html":
            continue
        kept: list[str] = []
        for block in document.blocks:
            shingle = " ".join(_fingerprint(block).split()[:12])
            if shingle in shingles:
                dropped += 1
                if verbose:
                    print(f"    boilerplate: {block.strip()[:70]}")
                continue
            kept.append(block)
        if not kept:
            # A page that is entirely furniture is a fetch problem, not
            # something to paper over by deleting it.
            raise ValueError(
                f"{document.source.url}: every block looked like site "
                f"furniture; refusing to store an empty document"
            )
        document.blocks = kept
    return dropped


def run_load(limit: int | None = None, verbose: bool = True) -> tuple[list[LoadedDocument], list[LoadResult]]:
    """Load every source. Failures are recorded in the manifest, not raised."""
    rows = read_sources()
    if limit:
        rows = rows[:limit]

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    documents: list[LoadedDocument] = []
    results: list[LoadResult] = []

    for index, row in enumerate(rows, start=1):
        try:
            document = load_one(row)
        except Exception as exc:  # noqa: BLE001 - one bad URL must not stop the run
            result = LoadResult(
                url=row["url"].strip(),
                scheme=row["scheme"],
                doc_type=row["doc_type"],
                status="error",
                error=f"{type(exc).__name__}: {exc}"[:300],
                fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            )
            results.append(result)
            if verbose:
                print(f"  [{index}/{len(rows)}] FAIL {row['scheme']:<28} {result.error}")
            continue
        documents.append(document)
        results.append(document.source)
        if verbose:
            print(
                f"  [{index}/{len(rows)}] ok   {row['scheme']:<28} "
                f"{document.source.kind:<4} {document.source.http_status} "
                f"{document.source.bytes_downloaded:>9,}B  "
                f"{document.source.text_chars:>7,}ch  {document.source.fetch_seconds:>5.2f}s"
            )

    write_manifest(results)
    return documents, results


MANIFEST_FIELDS = [
    "url", "scheme", "doc_type", "kind", "http_status", "bytes_downloaded",
    "text_chars", "pages", "fetch_seconds", "fetched_at", "raw_file", "status", "error",
]


def write_manifest(results: list[LoadResult]) -> Path:
    config.INGEST_MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with config.INGEST_MANIFEST_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return config.INGEST_MANIFEST_CSV

