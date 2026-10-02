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

## Sources

Fourteen official SBI MF documents, all of them public and none login-gated.
Full rows with fetch notes are in [`data/sources.csv`](data/sources.csv); what
each was ingested for is below.

| Doc type | Document | Used for |
|---|---|---|
| SID | [`SBI Flexicap Fund`](https://www.sbimf.com/docs/default-source/sif-forms/sid---sbi-flexicap-fund.pdf) | riskometer, exit load slabs, capital gains, expense detail |
| SID | [`SBI Large Cap Fund`](https://www.sbimf.com/docs/default-source/sif-forms/sid---sbi-large-cap-fund-(formerly-known-as-bluechip-fund).pdf) | exit load slabs, benchmark, riskometer |
| SID | [`SBI ELSS Tax Saver Fund`](https://www.sbimf.com/docs/default-source/sif-forms/sid---sbi-elss-tax-saver-fund.pdf) | 3-year lock-in, exit load slabs, benchmark, tax treatment |
| SID | [`SBI Small Cap Fund`](https://www.sbimf.com/docs/default-source/sif-forms/sid---sbi-small-cap-fund.pdf) | riskometer, exit load slabs, expense detail |
| SID | [`SBI Balanced Advantage Fund`](https://www.sbimf.com/docs/default-source/sif-forms/sid---sbi-balanced-advantage-fund.pdf) | benchmark, KIM/SID location, riskometer |
| Factsheet | [`SBI Flexicap Fund — Mar 2026`](https://www.sbimf.com/docs/default-source/scheme-factsheets/sbi-flexicap-fund-factsheet-march-2026.pdf) | minimum SIP amount, as on 31/03/2026 |
| Factsheet | [`SBI Large Cap Fund — Jun 2025`](https://www.sbimf.com/docs/default-source/scheme-factsheets/sbi-large-cap-fund-factsheet-june-2025.pdf) | as on 30/06/2025, the latest published for this scheme |
| Factsheet | [`SBI ELSS Tax Saver Fund — Mar 2026`](https://www.sbimf.com/docs/default-source/scheme-factsheets/sbi-elss-tax-saver-fund-factsheet-march-2026.pdf) | 3-year lock-in, as on 31/03/2026 |
| Factsheet | [`SBI Small Cap Fund — Mar 2026`](https://www.sbimf.com/docs/default-source/scheme-factsheets/sbi-small-cap-fund-factsheet-march-2026.pdf) | minimum SIP amount, as on 31/03/2026 |
| Factsheet | [`SBI Balanced Advantage Fund — Mar 2026`](https://www.sbimf.com/docs/default-source/scheme-factsheets/sbi-balanced-advantage-fund-factsheet-march-2026.pdf) | minimum SIP amount, as on 31/03/2026 |
| TER notice | [`Base TER change w.e.f. 20/03/2026`](https://www.sbimf.com/docs/default-source/expense-ratio/base-ter-change-w-e-f-20-03-2026.pdf) | per-scheme base expense ratios, the 4 equity schemes |
| Investor services | [`Ways to invest`](https://www.sbimf.com/ways-to-invest) | how to obtain statements, m-Easy SMS keywords |
| Campaign page | [`SBI Flexicap Fund`](http://www.sbimf.com/campaign/flexicap-fund) | objective, scheme type, key highlights (no numeric facts) |
| Campaign page | [`SBI ELSS Tax Saver Fund`](http://www.sbimf.com/campaign/elss) | objective, scheme type, key highlights (no numeric facts) |

**The allow-list is a domain, not a URL list.** `config.ALLOWED_DOMAINS` is
`sbimf.com`, `amfiindia.com`, `sebi.gov.in`, `sebi.co.in`; `ingest/loader.py`
rejects a source outside it *before* any request is made. Official URLs may be
added when a fact is missing — non-official ones may not.

Two rows carry `scheme = "All schemes"` in the CSV, because one document covers
every scheme. Retrieval treats those as global rather than scheme-filtered
(`config.GLOBAL_SCHEME`), or the only document stating a real figure would be
excluded by its own scheme filter.

Factsheet index, used whenever an answer or refusal points at "the official
factsheet": [`sbimf.com/factsheets`](https://www.sbimf.com/factsheets).

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

Streamlit serves on <http://localhost:8501>. The first question downloads the
embedding model (~86 MB); later runs start in a second or two.

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

## Sample Q&A

Regenerated by `tests/test_pipeline.py`, which re-asks every question live.
Full run, including refusals, out-of-corpus behaviour, the interface and every
open gap, is in [`samples/sample_qa.md`](samples/sample_qa.md).

| # | Question | Outcome | Citation |
|---|---|---|---|
| 1 | Expense ratio of SBI Flexicap Fund | answered | `sbimf.com/total-expense-ratio` |
| 2 | Lock-in for SBI ELSS Tax Saver Fund | answered | ELSS SID |
| 3 | Exit load for SBI Large Cap Fund | answered | Large Cap SID |
| 4 | Minimum SIP for SBI Small Cap Fund | answered | Small Cap SID |
| 5 | Riskometer and benchmark, SBI Balanced Advantage | partly answered | Balanced Advantage SID |
| 6 | Download capital-gains statement for SBI ELSS | **declined** | ELSS SID |
| 7 | Download latest factsheet, SBI Flexicap | answered | `sbimf.com/factsheets` |
| 8 | Exit load after 1 year, SBI Large Cap | answered | Large Cap SID |
| 9 | Benchmark index for SBI ELSS | answered | ELSS SID |
| 10 | KIM/SID for SBI Balanced Advantage Fund | answered | Balanced Advantage SID |

Three of them, verbatim. Each carries exactly one citation and an app-written
fetch date, which is the whole answer contract:

> **What is the exit load for SBI Large Cap Fund?**
> The exit load for SBI Large Cap Fund is 0.25% for exits within 30 days from
> the date of allotment, 0.10% for exits after 30 days and within 90 days, and
> NIL for exits after 90 days.
> https://www.sbimf.com/docs/default-source/sif-forms/sid---sbi-large-cap-fund-(formerly-known-as-bluechip-fund).pdf
>
> Last updated from sources: 2026-09-29

> **What is the benchmark index for SBI ELSS?**
> The First Tier benchmark of the SBI ELSS Tax Saver Fund is the BSE 500 TRI
> Index.
> https://www.sbimf.com/docs/default-source/sif-forms/sid---sbi-elss-tax-saver-fund.pdf
>
> Last updated from sources: 2026-09-29

> **How do I download my capital-gains statement for SBI ELSS?**
> That information is not in the official sources I have.
> https://www.sbimf.com/docs/default-source/sif-forms/sid---sbi-elss-tax-saver-fund.pdf
>
> Last updated from sources: 2026-09-29

The decline is the point worth reading. Six ELSS chunks contain "capital gains",
but every one of them is tax *treatment*, never a download procedure — statement
downloads sit behind a login on a separate host this bot is not allowed to read.
Declining with a citation is correct; inventing a plausible procedure would not
be. See known gap 1 in [`samples/sample_qa.md`](samples/sample_qa.md).

Refused before retrieval, so no model call and no chunks are consumed:

| Question | Class | Behaviour |
|---|---|---|
| Should I buy SBI Small Cap Fund? | advisory | facts-only message, educational link |
| Is SBI Large Cap Fund good for me? | advisory | facts-only message, educational link |
| Which of these funds performed best last year? | performance | points to the official factsheet |
| What is the price of gold today? | off_topic | out of scope message |

Guardrail expectations are pinned separately in
[`samples/guardrail_cases.md`](samples/guardrail_cases.md) and asserted by
`tests.test_guardrails`.

## Deliverables

| Deliverable | Where |
|---|---|
| Working prototype | [`app.py`](app.py) |
| Source list | [`data/sources.csv`](data/sources.csv), summarised in [Sources](#sources) |
| Inspectable chunks | `data/chunks.txt` |
| Persisted vector store | `data/chroma/` (committed on purpose, see below) |
| Sample Q&A | [`samples/sample_qa.md`](samples/sample_qa.md), summarised in [Sample Q&A](#sample-qa) |
| Guardrail cases | [`samples/guardrail_cases.md`](samples/guardrail_cases.md) |
| Disclaimer | `config.DISCLAIMER`, shown in the UI; see [Disclaimer](#disclaimer) |

## Deploying to Render

## Deploying to Render

[`render.yaml`](render.yaml) is a Blueprint, so there is nothing to fill in by
hand. In the Render dashboard: **New → Blueprint**, pick this repo, and Render
reads the build and start commands, the region, the compute plan and the health
check from that file. It prompts once for `GROQ_API_KEY`; paste the key there
and it is stored as a Render secret. Never put the key in `render.yaml` — it
would land in the repository's history.

The service is then at `https://mf-faqs-rag.onrender.com`, in Singapore.

| Setting | Value | Why |
|---|---|---|
| Root directory | *(empty)* | `app.py` is at the repo root |
| Build | `pip install -r requirements.txt` | one install; there is no torch |
| Start | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` | `--server.headless` is mandatory in a container |
| Health check | `/_stcore/health` | Streamlit's own endpoint; the default `/` runs the whole app script on every probe |
| Plan | `free` — 0.1 CPU, 512 MB | see the memory budget below |
| Python | `.python-version` → `3.14` | see below |

**The memory budget is the binding constraint, and there is no torch in this
project.** Measured resident set for a real query:

| Stage | RSS |
|---|---|
| after the store is opened | 102 MB |
| **after the first query** | **235 MB** |
| free tier limit | 512 MB |

That 130 MB jump is the embedding model loading on first use. It used to be
~530 MB instead, because embeddings ran under `sentence-transformers` and
therefore PyTorch: torch's runtime needs half a gigabyte to serve an 86 MB
model. On a 512 MB instance that is fatal, and it failed in a way that looked
like nothing at all — the page rendered perfectly, because rendering never
loads the model, and the process was OOM-killed the instant somebody asked a
question. There is no traceback, because the kernel ends the process rather
than Python; the service logs just show the app restarting.

So the model is unchanged but its *runtime* is not. `ingest/embedder.py`
executes the float32 ONNX export of the same `all-MiniLM-L6-v2` weights under
`onnxruntime`, and `sentence-transformers` and `torch` are no longer
dependencies at all. This is safe for the retrieval contract because the
float32 export reproduces the torch vectors to `1.7e-07` — mean cosine 1.0000000
over the corpus — so `MIN_SIMILARITY`, `GLOBAL_ROW_BONUS`, the committed vector
store and the test suite all carry over untouched. The quantised exports
(`model_O4`, `model_qint8_*`) are deliberately *not* used: they are smaller
again, but they move every vector far enough to invalidate the thresholds
measured in `config.py`. If that ever changes, re-measure them.

**Why the version is a file, not an env var.** Render requires
`PYTHON_VERSION` to be fully qualified — `3.14` alone is rejected, it wants
`3.14.7`. A `.python-version` file may omit the patch and resolves to the
newest 3.14.x, which is the safer of the two: `requirements.txt` is verified
against 3.14.7, and a patch release is not a reason to fail a build.

**The free tier is genuinely free-tier, and it is the main caveat.** The
instance is 0.1 CPU with 512 MB, it sleeps after 15 minutes idle, and waking
takes about a minute. Worse, the filesystem is ephemeral *per spin-down*, so the
~86 MB embedding model is downloaded from Hugging Face again on every cold
start, and the first question after a wake-up is slow. Cold starts are the
normal case on a free tier, not the exception. A `0.5c-512mb` instance ($7/mo)
wakes faster but still redownloads *and still has the same 512 MB*, so it buys
CPU and nothing else; a persistent disk ($0.25/GB/mo) is what actually fixes
the model redownload, and neither is free.

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
- The embedding model is downloaded on first question (~86 MB), and it needs
  ~130 MB of memory resident while it is loaded. On a 512 MB free-tier instance
  that is the difference between working and being OOM-killed, so see the memory
  budget in [Deploying](#deploying) before adding any heavyweight dependency.
- A two-scheme question is searched against both, but the answer still carries
  one citation. Read the Sources panel to see which chunks backed it.
- Actual total TER — the all-in annual cost including fund and underlying scheme
  expenses — is not in the corpus for most schemes. The base TER is, and is what
  an expense-ratio question gets answered with; a question specifically about
  *total* TER may be declined.

## Disclaimer

**This is a prototype for demonstrating retrieval over public documents. It is
not investment advice, not a recommendation, and not a substitute for a
registered financial adviser. Do not act on any figure it gives you — read the
linked official page first.**

The notice the app itself shows is one line, `config.DISCLAIMER`:

> **Facts-only. No investment advice.**

It is rendered twice in the UI — in the sidebar and again under the title — and
it is appended to every refusal message in `guardrails.py`, so a user who never
scrolls past the header still sees it before the first refusal.

**What the bot will not do, enforced in `guardrails.py` and pinned by
`tests.test_guardrails`:**

| Requested | Response | Where it points |
|---|---|---|
| Should I buy / is this fund good for me | states it shares published facts only, then stops | [AMFI](https://www.amfiindia.com/aboutamfi) |
| Which fund performed best / returns | makes no performance claim and no comparison | [SBI MF](https://www.sbimf.com/) factsheets |
| Anything containing a PAN, Aadhaar, account number, OTP, email or phone | says nothing was saved; the identifier is never stored, logged or echoed | [SEBI](https://www.sebi.gov.in/) |
| A question about nothing in the corpus | names what was searched, invents nothing | factsheet index |
| Something off-topic entirely | redirects to the topics it can answer | [AMFI](https://www.amfiindia.com/aboutamfi) |

**Why the constraints are mechanical rather than prompt-level.** "No investment
advice" left to the model's own judgement is not a control. Classification is a
pattern screen run before retrieval, so a refused question never reaches an LLM
at all — `tests.test_guardrails` asserts `used_llm=False` and zero retrieved
chunks for every refusal. The post-check then re-reads generated text and
repairs or replaces it: invented date sentences and advice sentences are
dropped, the body is truncated to 3 sentences, extra links are dropped and the
single source link is appended if absent. If nothing factual survives the
repair, the answer is discarded in favour of `MSG_NO_CONTEXT` rather than a URL
and a date wrapped around a recommendation.

**Figures are as of the fetch date, not today.** Every answer shows
`Last updated from sources:`, written by app code from the chunk's `fetched_at`
in [`data/ingest_manifest.csv`](data/ingest_manifest.csv). The corpus proves a
figure was copied from retrieved text rather than invented; it cannot prove the
live page still says the same thing. Exit loads, expense ratios and lock-in
periods all change — always confirm against the linked page.

**No personal data is accepted or retained.** The conversation buffer is
in-memory only and is dropped when the process exits. A refused PII message is
redacted in the transcript rather than echoed back, and "Clear chat" empties
both the transcript and the buffer.

## Documentation

- [`Docs/Problemstatement.txt`](Docs/Problemstatement.txt) — the brief
- [`Docs/PRD.md`](Docs/PRD.md) — goals, scope, success criteria, constraints
- [`Docs/architecture.md`](Docs/architecture.md) — components, data flow, stack
- [`Docs/implementation.md`](Docs/implementation.md) — phased build plan
- [`samples/sample_qa.md`](samples/sample_qa.md) — every question answered, and
  every one that was not
- [`samples/guardrail_cases.md`](samples/guardrail_cases.md) — the refusal and
  PII expectations
