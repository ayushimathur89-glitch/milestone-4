# Sample Q&A — Mutual Fund FAQs RAG Chatbot

Produced by `tests/test_pipeline.py` (run it to regenerate; it re-asks every
question live, so timings and scores move a little between runs).

- Collection: `mf_faqs`, 2,685 vectors, 384-dim `all-MiniLM-L6-v2`
- Model: `qwen/qwen3.8-27b` on Groq, temperature 0
- Retrieval: top 4 of 25 candidates, dense cosine floor 0.15, reranked on
  dense score + 0.35 x literal keyword overlap
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
