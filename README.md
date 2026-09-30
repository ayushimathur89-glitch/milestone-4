# Mutual Fund FAQs — Facts-Only RAG Chatbot

A RAG (Retrieval-Augmented Generation) assistant that answers factual questions
about a small set of mutual fund schemes using **only** official public pages.
Every answer carries exactly one source link. No investment advice.

> **Facts-only. No investment advice.**

## Status

Phase 1 (project setup) complete. Phases 2–6 are specified in
[`Docs/implementation.md`](Docs/implementation.md) and are not built yet.

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

## Run

Not yet available — the app is built in Phase 6. Ingestion is Phase 2–3.

## Deliverables

| Deliverable | Where |
|---|---|
| Working prototype | _pending (Phase 6)_ |
| Source list | [`data/sources.csv`](data/sources.csv) |
| Inspectable chunks | `data/chunks.txt` (Phase 2) |
| Sample Q&A | `samples/sample_qa.md` (Phase 5) |
| Disclaimer snippet | `config.DISCLAIMER` (shown in UI, Phase 6) |

## Deploying to Render

Free tier. Build `pip install -r requirements.txt`, start
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

`data/chroma/` is deliberately committed: Render's filesystem is ephemeral, so
the persisted vector store must ship with the repo or the app would re-ingest
on every cold start. See [`Docs/architecture.md`](Docs/architecture.md) §4.

## Known limits

- Corpus is limited to the schemes above and whatever their official pages
  state. Questions outside the corpus are refused rather than answered.
- Figures change. Each answer shows the source fetch date; always confirm
  against the linked official page before acting.
- No returns, performance, or comparison questions — these are refused and
  redirected to the official factsheet.
- No PII is accepted, stored, or logged. Queries containing a PAN, Aadhaar
  number, account number, OTP, email, or phone number are refused and not
  retained, including in the conversation buffer.
- Follow-up questions are resolved against the last 10 messages, so "what about
  its fees?" is searched as a question about the scheme under discussion.
  Unresolvable follow-ups are searched as typed, and the CLI prints the
  rewritten question whenever one is used. `/history`, `/reset` and
  `--no-memory` control the buffer.
- The embedding model is downloaded on first run (~90 MB).

## Documentation

- [`Docs/Problemstatement.txt`](Docs/Problemstatement.txt) — the brief
- [`Docs/PRD.md`](Docs/PRD.md) — goals, scope, success criteria, constraints
- [`Docs/architecture.md`](Docs/architecture.md) — components, data flow, stack
- [`Docs/implementation.md`](Docs/implementation.md) — phased build plan
