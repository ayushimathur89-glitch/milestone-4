# Guardrail test cases

Reference for `guardrails.py`. The executable version is
`tests/test_guardrails.py`, which asserts every expectation below:

```
.venv\Scripts\python -m tests.test_guardrails
```

No API key, no vector store, no network. Phase 4 is built so the constraint
layer is provable on its own, before any LLM is wired in.

## Classification

| Query | Class | Enforced by |
|---|---|---|
| "What is the expense ratio of SBI Flexicap Fund?" | `factual` | falls through to retrieval |
| "What is the lock-in period for SBI ELSS Tax Saver Fund?" | `factual` | falls through to retrieval |
| "Should I buy SBI Small Cap Fund?" | `advisory` | `_ADVISORY` |
| "Is SBI Large Cap Fund good for me?" | `advisory` | `_ADVISORY` |
| "Which of these funds performed best last year?" | `performance` | `_PERFORMANCE` |
| "My PAN is ..., tell me my returns" | `pii` | `_PII_PATTERNS`, screened first |
| "Call me on ..." | `pii` | `_PII_PATTERNS`, screened first |

Order matters. PII is checked before everything else, so a query mixing a PAN
with a performance request is refused for the identifier and the identifier
never propagates. Advisory is checked before performance, because "should I
buy the fund that returned the most?" is a recommendation request, and that is
the more restrictive of the two.

## Refusals

Each carries the facts-only line and a link verified reachable when written:

| Class | Message contains | Link |
|---|---|---|
| `advisory` | facts-only position, no recommendation | `amfiindia.com/aboutamfi` |
| `performance` | no claims or comparisons | `sbimf.com` |
| `pii` | states nothing was saved | `sebi.gov.in` |
| `off_topic` | redirects to answerable topics | `amfiindia.com/aboutamfi` |

`MSG_NO_CONTEXT` is the fifth message, used when retrieval returns nothing
usable: it names the documents that were searched and points at the
factsheet.

## Post-check

`verify_answer` enforces the three PRD rules on generated text: at most
`config.MAX_ANSWER_SENTENCES` sentences, exactly one source link, and a
`Last updated` line written by app code from the chunk's `fetched_at`.

Repairs, in order: drop invented date sentences, drop advice sentences,
truncate to the sentence cap, drop extra links, append the source link if
absent. If nothing factual survives, the result is `MSG_NO_CONTEXT` rather
than a link and a date wrapped around a recommendation.

A bare citation is not counted as a sentence, so an otherwise-valid answer
does not look one sentence over.

## PII non-persistence

The test PAN, phone number, Aadhaar, email, account number, and OTP appear in
exactly two places: `tests/test_guardrails.py` (the fixture) and
`Docs/implementation.md` (the spec's own table). The check scans the working
tree and fails if they turn up anywhere else.

`.pyc` files are excluded because CPython caches string constants into them on
import, so the literals appear in a build artifact no matter what the source
says. `.gitignore` already covers `__pycache__/`, so nothing is committed.

Phase 6 must additionally confirm a refused query is not echoed into the chat
transcript. A refusal that still displays the PAN has not solved anything.

## Ambiguous cases

Pinned in `check_ambiguity` as current behaviour, so a future change to the
patterns has to be deliberate:

- "Should I put my gold savings into a small cap fund?" -> `advisory`. The
  gold-price off-topic pattern does not hijack it, because fund vocabulary is
  present and advisory is checked first.
- "Tell me a joke." -> `off_topic`, via an explicit keyword.
- "xyzzy plugh" -> `factual`. Names no fund and no topic, so it reaches
  retrieval and is handled by the similarity gate and the no-context message.
  Refusing on a guess here would risk blocking questions the corpus does
  cover.
- "What is the weather like for equity mutual funds in monsoon?" ->
  `factual`, same reason: fund vocabulary suppresses the off-topic screen.
