# Mutual Fund FAQs — Facts-Only RAG Chatbot

A RAG (Retrieval-Augmented Generation) assistant that answers factual questions
about a small set of mutual fund schemes using **only** official public pages.
Every answer carries exactly one source link. No investment advice.

> **Facts-only. No investment advice.**

## Status

All six phases are built. See [`Docs/implementation.md`](Docs/implementation.md)
for the phase-by-phase plan and verification steps.

| Phase | Deliverable | Verified by |
|---|---|---|
| 1 | Project setup, config, requirements | `config.py` prints the key masked |
| 2 | Ingestion from official pages, chunking | `data/chunks.txt`, `tests/test_ingest.py` |
| 3 | Embedding store, hybrid retrieval, BM25 | `tests/test_retrieval.py` |
| 4 | Guardrails, PII, one-citation contract | `tests/test_guardrails.py` |
| 5 | Generation, conversation memory, end-to-end | `tests/test_pipeline.py`, `tests/test_memory.py` |
| 6 | Streamlit UI | `tests/test_ui.py` |

## Scope

**AMC:** SBI Mutual Fund (one AMC)

| Scheme | Type |
|---|---|
| SBI Flexicap Fund | Flexi-cap |
| SBI Large Cap Fund | Large-cap |
| SBI ELSS Tax Saver Fund | ELSS (tax) |
| SBI Small Cap Fund | Small-cap |
| SBI Balanced Advantage Fund | Balanced advantage |

**Sources:** official SBI MF / SEBI / AMFI pages only, listed in
[`data/sources.csv`](data/sources.csv). The allow-list is enforced by domain in
`config.ALLOWED_DOMAINS` and checked by the loader.

## Setup

Requires Python 3.14.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Add your free Groq key to `.env`:

```
GROQ_API_KEY=gsk_...
GROQ_MODEL=qwen/qwen3.8-27b
```

`.env` is git-ignored and must never be committed. On Render, set
`GROQ_API_KEY` in the service dashboard instead.

Check the configuration loads correctly:

```bash
python config.py
```

It prints the key masked, never in full.

## Run the app

The vector store is committed to the repo, so a fresh clone already has one and
there is nothing to ingest first.

```bash
.venv\Scripts\python -m streamlit run app.py
```

Streamlit serves on <http://localhost:8501>. The first run downloads the
embedding model (~90 MB); later runs start in a second or two.

The app opens with three example questions, keeps a message history, and shows
a **Sources** panel under every answer listing the exact chunks used, each with
its scheme, document type, similarity score, chunk id and fetch date. **Clear
chat** empties both the transcript and the conversation memory.

## Run from the command line

The same pipeline without the browser, useful for scripted checks:

One question and exit:

```bash
.venv\Scripts\python ask_cli.py -q "What is the exit load for SBI Large Cap Fund?"
```

With no arguments it starts a small interactive session. `/history` shows what
is remembered, `/reset` forgets it, `/filter` toggles the scheme filter, and
`--no-memory` runs questions standalone.

## Tests

```bash
.venv\Scripts\python -m tests.test_ui          # the app, headless
.venv\Scripts\python -m tests.test_pipeline    # 10 PRD questions end to end
.venv\Scripts\python -m tests.test_guardrails
.venv\Scripts\python -m tests.test_memory
```

`tests.test_ui` drives `app.py` through Streamlit's own `AppTest` harness with
the model stubbed, so it costs no API tokens. `tests.test_pipeline` and
`tests.test_memory` make real Groq calls and are subject to the free tier's
daily token limit.

`test_pipeline` currently exits `1` by design: 9 of the 10 PRD questions are
answered, and the tenth — where investors download a capital-gains statement —
has no qualifying official procedure on SBI MF's site to cite. See
[`samples/sample_qa.md`](samples/sample_qa.md) for the full evidence and the
adjudication of every decline.

## Deliverables

| Deliverable | Where |
|---|---|
| Working prototype | [`app.py`](app.py) |
| Source list | [`data/sources.csv`](data/sources.csv) |
| Inspectable chunks | `data/chunks.txt` |
| Persisted vector store | `data/chroma/` (committed on purpose, see below) |
| Sample Q&A | [`samples/sample_qa.md`](samples/sample_qa.md) |
| Disclaimer snippet | `config.DISCLAIMER`, shown in the UI |

## Deploying to Render

Root directory **empty** (`app.py` is at the repo root), build
`pip install -r requirements.txt` and start
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

Set `GROQ_API_KEY` and `PYTHON_VERSION=3.14` in the dashboard.

`data/chroma/` is deliberately committed: Render's filesystem is ephemeral, so
the persisted vector store must ship with the repo. The build does **not** run
the ingestion pipeline (doing so requires live third-party fetches and more
memory than the free tier has). See [`Docs/architecture.md`](Docs/architecture.md) §4.

## Known limits

- Corpus is limited to the schemes above and whatever their official pages
  state. Questions outside the corpus are refused rather than answered.
- Figures change. Each answer shows the source fetch date; always confirm
  against the linked official page before acting.
- No returns, performance, or comparison questions — these are refused and
  redirected to the official factsheet.
- No PII is accepted, stored, or logged. Queries containing a PAN, Aadhaar
  number, account number, OTP, email, or phone number are refused and not
  retained, including in the conversation buffer. A refused question is also
  redacted in the chat transcript rather than echoed back.
- Follow-up questions are resolved against the last 10 messages, so "what about
  its fees?" is searched as a question about the scheme under discussion.
  Unresolvable follow-ups are searched as typed, and the app says so on screen
  when it rewrites one. "Clear chat" empties both the transcript and this
  buffer. In the CLI, `/history`, `/reset` and `--no-memory` control it.
- The embedding model is downloaded on first run (~90 MB).
- A two-scheme question is searched against both, but the answer still carries
  one citation. Read the Sources panel to see which chunks backed it.
- Actual total TER — the all-in annual cost including fund and underlying scheme
  expenses — is not in the corpus for most schemes. The base TER is, and is what
  an expense-ratio question gets answered with; a question specifically about
  *total* TER may be declined.

## Documentation

- [`Docs/Problemstatement.txt`](Docs/Problemstatement.txt) — the brief
- [`Docs/PRD.md`](Docs/PRD.md) — goals, scope, success criteria, constraints
- [`Docs/architecture.md`](Docs/architecture.md) — components, data flow, stack
- [`Docs/implementation.md`](Docs/implementation.md) — phased build plan
