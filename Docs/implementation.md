# Implementation Plan: Mutual Fund FAQs RAG Chatbot

Derived from [PRD.md](./PRD.md) and [architecture.md](./architecture.md).

Six sequential phases. Do not start a phase until the previous phase's **Verify** steps pass — each phase has a working artifact you can inspect, so you always have something demoable even if you stop early.

Legend for **Verify**: commands are run from the project root (`Milestone 4/`).

---

## Phase 0: Prerequisites (one-time setup, not a build phase)

- Python 3.14 installed, `python -m venv .venv` created and activated.
- A free Groq API key obtained and available to paste into `.env`.
- Chrome/Edge available for manually opening the source URLs.
- Git initialised (so `.gitignore` protects `.env` from the first commit).

**Verify:** `python --version` shows 3.14.x; `.venv\Scripts\activate` succeeds; you can open `https://console.groq.com` and see a key.

---

## Phase 1: Project setup

### Files to create
- `requirements.txt`
- `.env`
- `.gitignore`
- `config.py`
- `data/sources.csv` (seeded with the 5 official SBI URLs from the problem statement)
- `README.md` (skeleton: setup steps, scope, known limits)
- `data/raw/`, `data/chroma/` (empty, with `.gitkeep`)

### What this phase does
Establishes the skeleton every other phase builds on. `config.py` becomes the single source of truth: `GROQ_API_KEY` loading, the embedding model name, the LLM model name, `CHUNK_MAX_WORDS` / `CHUNK_OVERLAP_WORDS`, `TOP_K`, the answer-contract constants (max sentences, citation requirement, the "Last updated from sources:" prefix), and `ALLOWED_DOMAINS` — the official domains (`sbimf.com`, `sebi.gov.in`, `amfiindia.com`).

`data/sources.csv` is the hand-curated source list (columns: url, scheme, doc_type, notes). It is a graded deliverable, so **the pipeline only ever reads it** — the loader writes its results to a separate `data/ingest_manifest.csv` instead.

Important: the "official sources only" rule is enforced at the **domain** level, not by pinning the 5 exact URLs from the brief. Those are campaign landing pages, and facts like exit-load slabs often live on a factsheet or SID instead. You may **add** an official URL when a fact is missing; you may never add a non-official one.

`.gitignore` must cover `.env`, `.venv/`, `__pycache__/`, and `data/raw/`. It must **not** cover `data/chroma/` — see Phase 3 for why.

### How to verify
1. `pip install -r requirements.txt` completes with no conflicts.
2. Run a one-liner that imports `config` and prints the loaded constants. The API key should print as a masked/`****` value, never in full.
3. **Confirm the Groq model id in the Groq console** and that it matches the one in `config.py`. Do this now — an invalid or retired model id otherwise surfaces as a confusing failure in Phase 5.
4. `git status --short` after `git add .` — confirm `.env` is **not** staged and `data/chroma/` is **not** ignored (`git check-ignore data/chroma/test` should print nothing).
5. Open `data/sources.csv` and confirm the 5 URLs from the brief are present and the file parses as valid CSV.

### Exit criteria
Config imports cleanly, key is masked, `.env` is untracked, sources list is complete.

---

## Phase 2: Loading & chunking

### Files to create
- `ingest/loader.py`
- `ingest/chunker.py`
- `ingest/run_ingestion.py` (load + chunk stages only for now; embed/store are Phase 3)
- `data/raw/*.txt` (generated)
- `data/ingest_manifest.csv` (generated)
- `data/chunks.txt` (generated)
- Possibly updated `data/sources.csv` (hand-edited, if a URL needs replacing)

### What this phase does
**Load:** `loader.py` reads each URL from `data/sources.csv`, rejects any domain not in `ALLOWED_DOMAINS`, extracts readable text from the HTML (strip nav, footer, cookie banners, script/style), and writes one clean `.txt` per source into `data/raw/`. It logs url, http status, byte count, fetch time, and resulting chunk count to `data/ingest_manifest.csv`. It does **not** modify `data/sources.csv`.

**Chunk:** `chunker.py` implements the strategy in architecture.md §2.3 — split on section heading → paragraph → sentence, `CHUNK_MAX_WORDS = 160`, `CHUNK_OVERLAP_WORDS = 30`, keep table headers attached to their rows so an exit-load or expense-ratio slab is never separated from its column meanings. Each chunk carries metadata: `chunk_id`, `scheme`, `doc_type`, `source_url`, `page_title`, `section`, `fetched_at`. (Chroma metadata values must be str/int/float/bool — never `None`; use empty string instead.)

The critical deliverable of this phase is `data/chunks.txt`: every chunk written out with its metadata so the corpus is reviewable without a vector store. This is a graded requirement and also your debugging surface for Phase 5 — if a retrieval misses, the first place to look is the chunk file.

### How to verify
1. Run the load stage. Confirm one `.txt` per URL in `data/raw/`, and that `data/ingest_manifest.csv` shows the expected status and byte count for each — a suspiciously small byte count means a JS-rendered page that returned no real content.
2. **Manually open 2 of the fetched files** and read them. Check for leftover navigation junk, truncated fee tables, or a page that came back mostly empty.
3. Run the chunk stage. Open `data/chunks.txt` and confirm: no chunk exceeds 160 words; overlap is visible at boundaries; every chunk has all 7 metadata fields with no null values; the source URL on each chunk resolves to an allowed domain.
4. **Targeted content check:** grep the chunk file for the specific facts the demo depends on — `expense ratio`, `exit load`, `lock-in`, `minimum SIP`, `benchmark`, `riskometer`, `capital gains`. Each should appear in at least one chunk. Any fact missing from the corpus cannot be answered later, so fix it now by **adding an official URL** (factsheet / SID / fee page) to `data/sources.csv` — never a third-party source.
5. Sanity check a table-bearing chunk: confirm an exit-load slab kept its header row and the whole slab in one chunk.

### Exit criteria
5 clean source files, `ingest_manifest.csv` shows no empty loads, `chunks.txt` complete and within limits, all 7 demo fact types present in the corpus.

---

## Phase 3: Embedding & vector store

### Files to create
- `ingest/embedder.py`
- `ingest/store.py`
- Update `ingest/run_ingestion.py` to the full load → chunk → embed → store pipeline
- `data/chroma/` (generated — persisted, and **committed to git**)

### What this phase does
`embedder.py` wraps `all-MiniLM-L6-v2`, downloading it once on first use (~86 MB, then cached locally forever — this is the "no API key" property from the brief). It is a **single shared instance** used by both this phase and `retrieval.py` in Phase 5, so chunk vectors and question vectors are guaranteed to come from the same model and occupy the same 384-dim space.

The weights are executed through the float32 ONNX export under `onnxruntime` rather than through sentence-transformers and PyTorch. This is a deployment-driven change made after Phase 6: torch's runtime needs ~530 MB resident to serve the model, the free-tier host has 512 MB, and the process was OOM-killed on the first question with nothing but an app restart in the logs. The float32 export reproduces the torch vectors to `1.7e-07`, so every similarity in `config.py`, the committed store and this phase's outputs are unchanged — see architecture.md §4.1.

`store.py` creates a ChromaDB `PersistentClient` rooted at `data/chroma/`, with one collection. Chunk text goes in the document, the 7 metadata fields go in the metadata, and the embedding is stored. Ingestion is idempotent — re-running replaces the collection rather than duplicating it.

By the end of this phase, ingestion runs once end-to-end and the vector store survives a process restart. **Commit `data/chroma/`** (21 MiB for 2,727 vectors) so that Render, whose filesystem is ephemeral, has a store to read on every wake-up. There is no runtime rebuild path: `app.py` opens the collection directly and `open_collection` raises if it is absent. See architecture.md §4.

### How to verify
1. Run full ingestion. It should print stage counts: chunks loaded, vectors stored, and the collection size.
2. **Restart-proof test (the key requirement from the brief):** stop the process, start a fresh one, and query the collection — vectors are still there. Ingestion did **not** re-run. This proves persistence.
3. Embedding sanity: embed one chunk and one question, assert both return shape `(384,)`, and print the cosine similarity. A sensible sentence should score higher than an unrelated one.
4. **Truncation check:** embed the longest chunk in `chunks.txt` and confirm the tokenized length is under 256. If it exceeds, your word→token ratio assumption in the chunking strategy is wrong — reduce `CHUNK_MAX_WORDS` and re-ingest. This catches silent data loss that would only show up as bad answers later.
5. Confirm `data/chroma/` is populated on disk and is **tracked** by git (`git status --short` should show it as a new, unignored directory).

### Exit criteria
Ingestion completes, collection size matches chunk count, vectors persist across restart, all vectors 384-dim, no truncation.

---

## Phase 4: Guardrails

### Files to create
- `guardrails.py`
- Optional: `samples/guardrail_cases.md` — the test cases below, written down

### What this phase does
Implements the constraint layer from the PRD *before* any LLM is wired in, so guardrails are provable in isolation and cost nothing to test (no Groq calls).

Two halves:

**Pre-check (routing).** Classify the incoming question into `factual`, `advisory`, `performance`, or `pii` and short-circuit the non-factual ones. `advisory` ("should I buy X?", "is X good for me?") returns the polite facts-only message plus a relevant educational link. `performance` ("which fund gave the best returns?", "compare returns of A vs B") returns a refusal that points to the official factsheet. `pii` is a regex/keyword screen for PAN, Aadhaar, account numbers, OTPs, emails, phone numbers — the query is refused, and the raw text is **not** echoed into the chat transcript, session state, or any log, since echoing it would mean storing the PII the PRD forbids.

This design also means the "no performance claims" and "no PII stored" constraints hold *structurally* — a performance question never reaches the LLM, so the model never gets the chance to invent a return figure.

**Post-check (answer contract).** Verifies the generated answer is ≤ 3 sentences and contains exactly one source URL. The `Last updated from sources:` line is **not** the LLM's to write — it is appended by app code from the cited chunk's `fetched_at` (architecture.md §2.2). The post-check only confirms the LLM did not invent a date of its own. On failure, the post-check repairs or falls back to a safe message rather than showing a bad answer.

### How to verify
Run the classifier directly over a table of test questions — this needs no API key and no vector store:

| Query | Expected class |
|---|---|
| "What is the expense ratio of SBI Flexicap Fund?" | `factual` |
| "What is the lock-in period for SBI ELSS Tax Saver Fund?" | `factual` |
| "Should I buy SBI Small Cap Fund?" | `advisory` |
| "Is SBI Large Cap Fund good for me?" | `advisory` |
| "Which of these funds performed best last year?" | `performance` |
| "My PAN is ABCDE1234F, tell me my returns" | `pii` |
| "Call me on 9876543210" | `pii` |

1. Every row classifies correctly. Pay special attention to the `performance` and `pii` rows — those are the two most likely to fall through.
2. For each refused class, print the exact response the user would see. Read it: it must be polite, must say facts-only, and must include a working educational link.
3. **PII non-persistence check:** grep the codebase and any log/output files for the test PAN and phone number. Neither string may appear anywhere except in the test file itself. This is the evidence for the "no PII stored" requirement. Verify in Phase 6 that the refused text is not echoed into the chat transcript either — a refusal that still displays the PAN has not solved anything.
4. Feed the post-check a deliberately bad answer (4 sentences, 0 citations, and an invented "Last updated" date) and confirm it is caught.

### Exit criteria
All 7 test cases classify correctly, refusals read well and link out, no PII written to disk, post-check catches bad answers.

---

## Phase 5: Retrieval + LLM answer generation

### Files to create
- `retrieval.py`
- `generator.py`
- `samples/sample_qa.md` (the graded 5–10 query deliverable)

### What this phase does
**Retrieve:** `retrieval.py` embeds the user question with the same `all-MiniLM-L6-v2` instance from Phase 3, runs a ChromaDB similarity search for `TOP_K` chunks, and applies a scheme metadata filter when the question names a scheme (e.g. an ELSS lock-in question filters to the ELSS scheme's chunks — this is why `scheme` is in the metadata). It returns chunk text plus the source URL.

**Generate:** `generator.py` calls Groq with a prompt that states the answer contract explicitly: answer only from the supplied context, ≤ 3 sentences, exactly one citation link drawn from the context metadata, refuse rather than speculate when the context does not contain the answer, and never give advice or compute returns. The prompt explicitly says **do not state or guess any date** — the app appends `Last updated from sources: <fetched_at>` itself from the cited chunk's metadata, so the model never has an opportunity to invent one. The post-check from Phase 4 runs on the output. Missing-context questions get an honest "not in the sources" response rather than a hallucinated expense ratio.

### How to verify
Run the 10 questions from PRD §5 through the full pipeline in a script (no UI yet) and inspect the output carefully:

1. **Grounding:** each answer's numbers match the source text. Open the cited URL in a browser and confirm the expense ratio, exit load, lock-in, and minimum SIP actually say what the answer claims. This is the most important check in the whole project — do not skip it.
2. **Citations:** exactly one URL per answer, and its domain is in `ALLOWED_DOMAINS`. Check for fabricated or third-party links.
3. **Length and date:** every answer body is ≤ 3 sentences, and the `Last updated from sources:` line is appended by the app with a date that matches the `fetched_at` in `data/ingest_manifest.csv`. If the model states its own date, the post-check should have caught it.
4. **Routing end-to-end:** "Should I buy SBI Small Cap Fund?" returns the Phase 4 refusal and makes **no** Groq call (confirm via the request log).
5. **Retrieval quality:** log the top-k chunks for 2–3 questions and read them in `data/chunks.txt`. If a fact was in the corpus but not retrieved, the cause is chunk granularity or the `TOP_K` value, not the LLM — tune here.
6. **Out-of-corpus test:** ask something the corpus genuinely does not cover (e.g. "What is the NAV of SBI Flexicap Fund today?") and confirm it refuses or says it isn't in the sources rather than inventing a number.
7. Save the passing Q&A pairs into `samples/sample_qa.md` — this is a required deliverable.

### Exit criteria
All 10 PRD questions answer with one verified citation, refusals route correctly without an LLM call, out-of-corpus questions refuse, `sample_qa.md` written.

---

## Phase 6: UI

### Files to create
- `app.py`
- Update `README.md` (final setup steps, scope, known limits, Render deploy notes)
- Final `samples/sample_qa.md`

### What this phase does
The small Streamlit app required by the brief: a welcome line, **3 example questions** (clickable, from PRD §5), the note "Facts-only. No investment advice.", and the disclaimer snippet as a deliverable. One text input, answers rendered with the citation link. Each turn runs the Phase 4 pre-check before touching the LLM, and the Phase 4 post-check before rendering.

Only cosmetic items are added now: page config, `st.set_page_config`, spinner during retrieval, and a sidebar showing the scope (AMC + the 5 schemes) so the demo audience can see the corpus boundary.

Nothing new to the RAG pipeline happens in this phase — the app is a thin client over the already-verified modules. If a Phase 5 check fails, fix it there, not with a UI workaround.

### How to verify
1. `streamlit run app.py` starts locally; the welcome line, 3 examples, facts-only note, and disclaimer are all visible on first load.
2. Click each of the 3 example questions — each produces an answer with a working link.
3. Type an advisory question in the UI and confirm the refusal appears and the app stays usable afterward (no crash, no stuck state).
4. Enter a PAN-shaped string and confirm it is refused, that the PAN text does **not** appear in the on-screen transcript, and that it is absent from the Streamlit logs.
5. Paste an answer containing a markdown link and confirm it renders as a clickable link, not raw text.
6. **Cold-start test:** stop the app, delete nothing, restart it. Confirm it does **not** re-ingest (Phase 3 persistence) and the first answer is fast.
7. **Render deploy:** commit `data/chroma/`, push to GitHub, create a Render web service with root directory **empty** (`app.py` is at the repo root), build `pip install -r requirements.txt` and start `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`, set `GROQ_API_KEY` and `PYTHON_VERSION=3.14` in the dashboard, and confirm the app loads and answers a question in the browser. Confirm the build log shows **no** ingestion output — the store is committed, and ingestion at build time was tried and reverted (see architecture.md §4). The first cold start is slow because the ~90 MB embedding model downloads; that is expected. **Then wait for the service to spin down and load the URL again**, which is the check that matters: the ephemeral filesystem wipes anything written at runtime, so only a committed store survives a restart.
8. Read `README.md` top to bottom as if you were a grader receiving this fresh: setup works, scope (AMC + 5 schemes) is stated, known limits are honest.

### Exit criteria
App runs locally and on Render, all UI elements present, guardrails visible in the UI, no re-ingestion on restart, README complete.

---

## Final checklist against PRD deliverables

| Deliverable | Built in |
|---|---|
| Working prototype (app) or ≤3-min demo video | Phase 6 |
| Source list CSV/MD of the URLs used | Phase 1, updated Phase 2 |
| README: setup, scope, known limits | Phase 1, finalized Phase 6 |
| Sample Q&A (5–10 queries + answers + links) | Phase 5, finalized Phase 6 |
| Disclaimer snippet in the UI | Phase 6 |
| `data/chunks.txt` (inspectable chunks) | Phase 2 |
| Citation on every answer, ≤ 3 sentences, "Last updated from sources:" | Phase 4 + 5 (date appended by app) |
| Facts-only refusals with educational link | Phase 4 |
| No PII accepted or stored (incl. not echoed to transcript) | Phase 4, verified Phase 6 |
| Groq key in `.env`, never committed | Phase 1, 6 |
| Persisted ChromaDB, ingest once | Phase 3 |
| Official sources only (domain allow-list) | Phase 1, 2 |
| Deployable to Render (free tier) | Phase 6 |

---

## Suggested order of attack

If time is short, the demo-critical path is Phases 1 → 2 → 3 → 4 → 5 → 6, but the two phases worth extra care are **Phase 2** (if the corpus is bad, nothing downstream can be good — and some AMC pages may need a saved local copy) and **Phase 5** (grounding verification is what a grader will actually check). Phases 4 and 6 are comparatively quick.
