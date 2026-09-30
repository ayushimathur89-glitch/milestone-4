# Sample Q&A — Mutual Fund FAQs RAG Chatbot

Produced by `tests/test_pipeline.py` (run it to regenerate; it re-asks every
question live, so timings and scores move a little between runs).

- Collection: `mf_faqs`, 2,690 vectors, 384-dim `all-MiniLM-L6-v2`
- Model: `qwen/qwen3.8-27b` on Groq, temperature 0
- Retrieval: top 6 of 60 candidates, dense cosine floor 0.40, reranked on
  dense score + 0.35 x literal keyword overlap, plus 0.40 for a row of a
  global-scope table that names the queried scheme when the document's own
  URL and title agree with the question's subject
- Answer contract: <= 3 sentences, exactly one citation, app-written date

## Result summary

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

9 of 10 answered, 1 declined, 0 routed away. Question 5 answers the benchmark
half and states plainly that the riskometer level is not in the sources.

## Answers as the user saw them

**1. What is the expense ratio of SBI Flexicap Fund?**
> The SID states that the AMC has estimated that up to 2.25% of the daily net
> assets of the scheme will be charged as expenses. For the actual current
> expenses being charged, investors should refer to the fund's total expense
> ratio page.
> https://www.sbimf.com/total-expense-ratio
>
> Last updated from sources: 2026-09-29

**2. What is the lock-in period for SBI ELSS Tax Saver Fund?**
> The SBI ELSS Tax Saver Fund has a statutory lock-in period of 3 years.
> https://www.sbimf.com/docs/default-source/sif-forms/sid---sbi-elss-tax-saver-fund.pdf
>
> Last updated from sources: 2026-09-29

**3. What is the exit load for SBI Large Cap Fund?**
> The exit load for SBI Large Cap Fund is 0.25% for exits within 30 days from
> the date of allotment, 0.10% for exits after 30 days and within 90 days, and
> NIL for exits after 90 days.
> https://www.sbimf.com/docs/default-source/sif-forms/sid---sbi-large-cap-fund-(formerly-known-as-bluechip-fund).pdf
>
> Last updated from sources: 2026-09-29

**4. What is the minimum SIP amount for SBI Small Cap Fund?**
> The minimum SIP amount for SBI Small Cap Fund is INR 500 for Daily SIP,
> Rs. 1000 for Weekly SIP, and Rs. 500 for Monthly SIP.
> https://www.sbimf.com/docs/default-source/sif-forms/sid---sbi-small-cap-fund.pdf
>
> Last updated from sources: 2026-09-29

**5. What is the riskometer and benchmark for SBI Balanced Advantage Fund?**
> The benchmark for SBI Balanced Advantage Fund is the NIFTY 50 HYBRID
> COMPOSITE DEBT 50:50 INDEX. The provided sources do not state the specific
> riskometer level for the scheme.
> https://www.sbimf.com/docs/default-source/sif-forms/sid---sbi-balanced-advantage-fund.pdf
>
> Last updated from sources: 2026-09-29

**6. How do I download my capital-gains statement for SBI ELSS?**
> That information is not in the official sources I have.
> https://www.sbimf.com/docs/default-source/sif-forms/sid---sbi-elss-tax-saver-fund.pdf
>
> Last updated from sources: 2026-09-29

**7. Where can I download the latest factsheet for SBI Flexicap Fund?**
> The latest factsheet for SBI Flexicap Fund can be downloaded from
> https://www.sbimf.com/factsheets
>
> Last updated from sources: 2026-09-29

**8. Is there an exit load if I redeem SBI Large Cap Fund units after 1 year?**
> No, there is no exit load for redemption after 90 days from the date of
> allotment, which includes redemptions after 1 year.
> https://www.sbimf.com/docs/default-source/sif-forms/sid---sbi-large-cap-fund-(formerly-known-as-bluechip-fund).pdf
>
> Last updated from sources: 2026-09-29

**9. What is the benchmark index for SBI ELSS?**
> The First Tier benchmark of the SBI ELSS Tax Saver Fund is the BSE 500 TRI
> Index.
> https://www.sbimf.com/docs/default-source/sif-forms/sid---sbi-elss-tax-saver-fund.pdf
>
> Last updated from sources: 2026-09-29

**10. Where can I find the KIM/SID for SBI Balanced Advantage Fund?**
> The KIM/SID for SBI Balanced Advantage Fund can be found at
> https://www.sbimf.com/docs/default-source/sif-forms/sid---sbi-balanced-advantage-fund.pdf
>
> Last updated from sources: 2026-09-29

## Refusals (no Groq call, no retrieval)

| Question | Class | Behaviour |
|---|---|---|
| Should I buy SBI Small Cap Fund? | advisory | facts-only message, educational link |
| Is SBI Large Cap Fund good for me? | advisory | facts-only message, educational link |
| Which of these funds performed best last year? | performance | points to the official factsheet |
| What is the price of gold today? | off_topic | out of scope message |

Confirmed by `tests.test_guardrails` and section 3 of `tests.test_pipeline`:
`used_llm=False` and zero retrieved chunks for all four.

## Out-of-corpus behaviour

| Question | Behaviour |
|---|---|
| What is the NAV of SBI Flexicap Fund today? | declined, no figure invented |
| What will the returns be next year? | declined, no figure invented |

The corpus has no live NAV feed and no projections, and neither question
produced a number.

## Investigated: "What is the total expense ratio of SBI Large Cap Fund?"

**Before.** The bot answered with the SEBI regulatory ceiling - 2.25% on the
first Rs.500 crore of assets, 1.05% on the balance - and cited the Large Cap
SID. That is the *permissible maximum*, not the scheme's expense ratio. It
looked confident and was unsupported.

**What was actually wrong**, in order of discovery:

1. `data/sources.csv` stores the base-TER notice with `scheme = "All schemes"`
   because one table lists every scheme. An exact scheme filter therefore
   excluded the only document stating a real TER. `scheme_filter()` now unions
   the scheme with `config.GLOBAL_SCHEME`.
2. The notice is a PDF, and `pypdf` flattens its table onto a single line.
   `is_table_block()` only recognises a table once at least two
   newline-separated rows exist, so this one was invisible to it and was packed
   as prose. Only the *first* window of each table line kept the column names,
   so later windows held bare figures. `_window_split` and `_token_split` now
   re-seat the header on every piece of a tabular run.
3. `_token_split` cut chunks by calling `tokenizer.decode()` on a token slice.
   WordPiece decode lowercases and re-spaces punctuation, so `SBI` became `sbi`
   and `0.65` became `0. 65`. This corrupted **every number in the whole
   corpus** wherever the 256-token ceiling bound - exit loads, NAVs, dates and
   expense ratios alike. It now cuts the original string at token offsets, so
   the stored text is byte-identical to the source.
4. With the row readable, it still ranked 25th: its own header abbreviates the
   question ("Existing Base TER (%)" against "total expense ratio"), so it earns
   no keyword credit while notice prose that repeats the question's words earns
   a lot. `config.GLOBAL_ROW_BONUS` promotes a global-table row that names the
   queried scheme, but only when the document's own URL and title agree with the
   question's subject. Without that second condition the same row was promoted
   to first place for an exit-load question and question 8's citation
   silently switched to the TER notice.

**After.** The row is retrieved and quoted correctly
(`SBI Large Cap Fund Direct 0.65 0.66 20.03.2026`), and the bot declines. The
decline is correct - see gap 4. The suite is unchanged at 9/10 and the corpus
went from 2,685 to 2,690 chunks.

## Known gaps

1. **Capital-gains statement download (question 6) has no public official
   source, so the decline is final unless the corpus changes.**
   Six ELSS chunks mention capital gains, but only tax treatment ("long-term
   gains above Rs. 1 lakh are taxed at 20%"), never a statement-download
   procedure. Probing for a usable page found nothing:
   - twelve guessed paths (`/transaction`, `/account-statement`,
     `/tax-report`, `/capital-gains-statement`, and others) all return
     HTTP 404 with SBI MF's error page
   - crawling the nav of `/`, `/factsheets`, and the other live pages
     surfaced no account-statement page at all; the closest hits were
     `/tax-planning-calculator` (no mention of statements),
     `/annual-financial-reports`, and the SAI PDF
   - the SAI mentions "capital gain" 24 times and "statement" 61 times, but
     every statement reference is about trustee accounts or audits, never
     how an investor obtains a capital-gains statement

   Statement downloads are served from a separate login-gated app host, so
   the fact is not published on a page this bot is allowed to read. Adding an
   unrelated page to make the number look better would be worse than the
   honest decline. `Docs/implementation.md` line 175 wants all ten answered;
   this one stays open, and the suite exits non-zero while it does.
2. **Riskometer level for SBI Balanced Advantage Fund (question 5) is present
   in the corpus but unreadable.** Chunk `sbi-balanced-advantage-fund-2126`
   holds the product-labelling table, but the PDF's riskometer cell extracts
   as the string "As per AMFI Tier I Benchmark", so the level itself is
   absent. This is a table-extraction limit, not a retrieval miss.
3. **Grounding is checked against the fetched corpus, not a live page.** The
   test proves each figure was copied from retrieved text rather than
   invented; it cannot prove the live sbimf.com page still says the same
   thing. `data/ingest_manifest.csv` carries each `fetched_at` date for that
   reason, and every answer carries it on screen.
5. **No document in the corpus states a scheme's total expense ratio.** The SID
   gives only the regulatory ceiling (2.25% / 1.50% slabs), which is a limit
   and not the scheme's own figure, and the factsheet does not state it. The
   base-TER notice does give per-scheme figures - SBI Large Cap Fund direct
   moves 0.65% to 0.66% effective 2026-03-20 - but it defines base TER as
   "Total Expense Ratio excluding additional expenses provided in Regulation
   52(6A)(b), 52(6A)(c)", so it is a component and not the total. Answering
   "0.66%" to a total-TER question would overstate what the source says, so the
   bot declines. The same applies to question 1, which is why it defers to
   `sbimf.com/total-expense-ratio` rather than quoting 2.25%. A dedicated
   official TER page per scheme in `data/sources.csv` would close this.
6. **`"Can I redeem SBI Large Cap Fund units after 90 days?"` is routed to
   advisory instead of factual.** It is a question about whether a load
   applies, and the corpus answers it, but "can I ... redeem" matches the
   advisory vocabulary, so the bot refuses a question it can answer. The
   equivalent phrasing in the PRD set ("Is there an exit load if I redeem
   ...?") is classified correctly, so this is a phrasing gap in the advisory
   patterns rather than a retrieval problem. Left open deliberately: advisory
   vocabulary is a blunt instrument and narrowing it needs the same kind of
   measurement as gap 4, not a guess.

## Fixed during this pass

1. **Off-topic routing for named competitions and general knowledge.** "Who won
   the FIFA World Cup in 2022?" carries no sport word, so the vocabulary that
   caught "IPL" let it through to retrieval, where the best unrelated chunk
   scored 0.163 and the model was asked a football question out of
   mutual-fund documents. `guardrails.py` now covers world cups, FIFA, UEFA,
   the Olympics, named domestic leagues and clubs, `who won`, and general
   knowledge with no topic word ("what is the capital of France", "what time
   does the Mumbai train leave"). Ten cases are pinned in
   `tests/test_guardrails.py`; classification is 42/42.
2. **`MIN_SIMILARITY` recalibrated from 0.15 to 0.40, by measurement.** At 0.15
   the floor sat *below* every off-topic question tested, so it separated
   nothing. Over 20 answerable and 20 off-topic questions the lowest answerable
   scores 0.539 ("What is the benchmark index for SBI ELSS?") and the highest
   off-topic 0.209. Before the vocabulary fix the highest off-topic reached
   0.469 - a 0.070 margin that no threshold could have made safe - which is why
   the classifier is the primary gate and the floor is only a backstop. At 0.40
   there is 0.139 of headroom below the weakest real question and 0.191 above
   the worst off-topic one.
3. **Question 6 is no longer reported as a retrieval bug.** The declined-question
   check counted any corpus chunk containing the probe phrase, and six ELSS
   chunks contain "capital gains" - all tax treatment, none a procedure. It
   called a source gap "our bug". The test now distinguishes the two, verifies
   the gap claim rather than trusting it (a chunk only counts as corroborating if
   a procedure cue sits within 200 characters of the probe), and reports
   `0 unexplained declines`. A future corpus that does publish the procedure
   will flip this to a failure. The run still exits non-zero, because line 175
   wants all ten answered and nine are; the message now says which kind of
   problem each decline is.

