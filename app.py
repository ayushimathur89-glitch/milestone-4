"""Streamlit chat UI over the verified Phase 5 pipeline.

    .venv\\Scripts\\streamlit run app.py

`Docs/implementation.md` Phase 6 is explicit that this file is a thin client:
"nothing new to the RAG pipeline happens in this phase". Every answer comes from
`generator.ask`, so the pre-check runs before the LLM and the post-check before
anything is rendered. This module formats what it is given and adds no
answering logic of its own - a Phase 5 defect gets fixed in Phase 5, never
papered over here.

Three things this file does care about, each of which is a correctness issue
rather than a cosmetic one:

1. **PII never reaches the transcript.** `generator.ask` already refuses a
   question containing a PAN or an account number, and `memory.Conversation`
   refuses to store it. But the user's own message is rendered by the chat
   widget, so a refusal would still put the PAN on screen next to the message
   saying it was not saved. Verification step 4 asks for the PAN to be absent
   from the transcript, and echoing the input verbatim is the one place it could
   reappear. Questions are therefore redacted for display whenever
   `guardrails.detect_pii` fires, which also covers the case where a future
   change lets such a question through to an answer.

2. **The store is opened once per process, not per message.** `store.get_client`
   and the collection are module-level caches, and Streamlit reruns this script
   on every keystroke, so the app holds them in `st.session_state`. Without that
   the embedding model would be reloaded on every interaction.

3. **The answer's date is the source's, not the renderer's.** `verify_answer`
   appends `Last updated from sources:` from the cited chunk's `fetched_at`, and
   this file prints `answer.answer` as-is rather than adding a timestamp of its
   own, so there is exactly one date and it came from the corpus.

Verification lives in `tests/test_ui.py`, which drives this file headlessly
through Streamlit's own `AppTest` harness rather than by importing functions.
Streamlit reruns the whole script on every interaction, so the only honest way
to test "the app is still usable after a refusal" is to run the app. The LLM is
stubbed there, which keeps the suite free of API cost and immune to a rate
limit; the refusal paths need no stub at all, since they never reach the model.
"""

from __future__ import annotations

import re

import streamlit as st

import config
import generator
import guardrails
import memory
from ingest import store

# The three examples the Phase 6 spec asks for, taken verbatim from PRD section
# 5. They are the questions a demo audience is most likely to try first.
#
# Two of them answer with a figure. The expense-ratio one answers by pointing at
# the official total-expense-ratio page, because the corpus carries each scheme's
# base TER but no scheme's total TER - see known gap 4 in samples/sample_qa.md.
# It is still an answer, and the Sources panel shows the retrieved rows, but it
# is worth knowing that clicking it will not produce a percentage.
EXAMPLE_QUESTIONS = (
    "What is the expense ratio of SBI Flexicap Fund?",
    "What is the lock-in period for SBI ELSS Tax Saver Fund?",
    "What is the exit load for SBI Large Cap Fund?",
)

SCOPE_SCHEMES = (
    "SBI Flexicap Fund",
    "SBI Large Cap Fund",
    "SBI ELSS Tax Saver Fund",
    "SBI Small Cap Fund",
    "SBI Balanced Advantage Fund",
)

WELCOME = (
    "Ask a factual question about an SBI mutual fund scheme. I answer only from "
    "official SBI MF, SEBI and AMFI pages, and I say so plainly when a source "
    "does not cover something."
)


def _redact(text: str) -> str:
    """Replace any PII in `text` with a note that something was removed.

    The value is never returned by `guardrails.detect_pii`, only its label, so
    the redaction is done by pattern and does not depend on knowing what was
    matched. The result is safe to render and safe to put in a log.
    """
    kind = guardrails.detect_pii(text)
    if not kind:
        return text
    # A blunt replacement of the whole message: the point is to keep the
    # identifier off the screen, and reconstructing a partially masked PAN is
    # not a smaller risk than hiding the message.
    return f"[{kind} removed - that kind of personal detail is not stored]"


def _sources_expander(answer: generator.Answer) -> None:
    """The retrieved chunks behind an answer, in an expander.

    The citation is the single link the contract allows. This is the evidence
    behind it, and it is collapsed by default so the transcript stays readable.
    Sub-threshold chunks are included deliberately: a question that fell short
    is easier to diagnose with the near-misses visible than without them.
    """
    # No expander at all when nothing was retrieved. A question refused by the
    # pre-check returns before retrieval by design, and an empty "Sources (0 used
    # of 0 retrieved)" panel next to it implies the sources were consulted and
    # came up short, which is the opposite of what happened.
    if not answer.hits:
        return

    with st.expander(f"Sources ({len(answer.above_floor)} used of "
                     f"{len(answer.hits)} retrieved)"):
        if answer.scheme_filter:
            st.caption(f"Scoped to: {answer.scheme_label}")
        if answer.declined:
            st.info("The context did not contain the answer, so the model "
                    "declined rather than guessing. The chunks below are the "
                    "nearest matches, not the answer.")
        for i, hit in enumerate(answer.hits, start=1):
            used = "used" if hit.similarity >= config.MIN_SIMILARITY else "below floor"
            st.markdown(
                f"**[{i}] {hit.scheme}** - {hit.doc_type}"
                + (f" / {hit.section}" if hit.section else "")
                + f" - similarity {hit.similarity:+.3f} ({used})"
            )
            st.caption(f"{hit.chunk_id} | fetched {hit.fetched_at}")
            if hit.matched_terms:
                st.caption("matched: " + ", ".join(hit.matched_terms))
            st.write(hit.preview(600))
            if hit.allowed_url:
                st.caption(hit.source_url)


def _split_citation(answer: generator.Answer) -> tuple[str, str]:
    """Split the answer into (body, link) so the citation shows up exactly once.

    `verify_answer` appends the citation to the answer as bare text, and
    Streamlit's markdown autolinks bare URLs - so printing `answer.answer` and
    then adding a `[url](url)` below it puts the same source on screen twice.
    That reads as two sources for one claim, which is exactly the impression the
    one-citation contract exists to prevent.

    The text handed back is the verified text with the citation lifted out; the
    link is re-attached as an explicit markdown link, so the source is still
    clickable and still points at the cited chunk. Nothing is reworded.
    """
    if not answer.source_url:
        return answer.answer, ""

    body = answer.answer.replace(answer.source_url, " ")
    body = "\n".join(line.rstrip() for line in body.splitlines())
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body, f"[{answer.source_url}]({answer.source_url})"


def _render_answer(answer: generator.Answer) -> None:
    body, link = _split_citation(answer)
    st.markdown(body)
    if link:
        # The label is the URL itself, deliberately plain: one link to the cited
        # chunk, not a link plus a synthesised deep link to a different document.
        st.markdown(link)

    if answer.declined:
        st.caption("Declined: the retrieved context did not contain the answer.")

    if answer.rewrite_notes:
        st.caption(f"Follow-up resolved from earlier context "
                   f"({answer.resolved_question}).")

    if answer.problems:
        # Two different things arrive here and the label has to tell them
        # apart. `repairs` non-empty means the post-check changed the text. An
        # empty `repairs` with a problem attached means the question was
        # refused by the pre-check or blocked before the model ran, and calling
        # that "the post-check changed this" would be plainly false.
        label = ("Why the post-check changed this answer" if answer.repairs
                 else "Why this was not answered")
        with st.expander(label):
            for problem in answer.problems:
                st.write(f"- {problem}")
            for repair in answer.repairs:
                st.write(f"- repaired: {repair}")

    _sources_expander(answer)


def _get_collection():
    """Open the store once per browser session.

    Streamlit reruns the whole script per interaction, and reloading the
    embedding model each time would make the app feel broken. The client and
    collection are cheap to hold and safe to reuse: the store is read-only here.
    """
    if "collection" not in st.session_state:
        st.session_state.collection = store.open_collection(store.get_client())
    return st.session_state.collection


def _get_conversation() -> memory.Conversation:
    if "conversation" not in st.session_state:
        st.session_state.conversation = memory.Conversation()
    return st.session_state.conversation


def main() -> None:
    st.set_page_config(
        page_title="SBI Mutual Fund FAQs",
        page_icon="📊",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.title("SBI Mutual Fund FAQs")
    st.caption(WELCOME)
    st.caption(f"**{config.DISCLAIMER}**")

    with st.sidebar:
        st.header("Scope")
        st.caption(
            "This assistant covers one AMC and the five schemes below, using "
            "only their official public pages. A question outside that boundary "
            "is refused rather than answered from general knowledge."
        )
        for scheme in SCOPE_SCHEMES:
            st.markdown(f"- {scheme}")
        st.divider()
        st.caption(f"Corpus: {len(SCOPE_SCHEMES)} schemes, "
                   f"official domains only.")
        st.caption(f"Similarity floor: {config.MIN_SIMILARITY}")
        st.caption(f"Model: {config.LLM_MODEL}")

    if not config.GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set. Add it to .env and restart. "
                 "The key is never read from the browser.")
        return

    conversation = _get_conversation()
    collection = _get_collection()

    if st.button("Clear chat", use_container_width=True):
        # Both halves matter: the visible transcript and the memory behind
        # follow-up rewriting. Clearing only the transcript would leave the bot
        # still resolving "its" against a conversation the user can no longer
        # see, which is the more confusing of the two failures.
        st.session_state.messages = []
        st.session_state.conversation = memory.Conversation()
        st.rerun()

    with st.container():
        st.caption("Try one of these:")
        for example in EXAMPLE_QUESTIONS:
            if st.button(example, key=f"example-{example[:24]}"):
                st.session_state.pending = example

    pending = st.session_state.pop("pending", None)
    typed = st.chat_input("Ask a question about an SBI mutual fund scheme")
    question = typed or pending

    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(_redact(message["text"]))
            else:
                _render_answer(message["answer"])

    if question:
        shown = _redact(question)
        with st.chat_message("user"):
            st.markdown(shown)

        with st.chat_message("assistant"):
            with st.spinner("Searching the official sources..."):
                answer = generator.ask(
                    question, collection=collection, conversation=conversation
                )
            _render_answer(answer)

        st.session_state.setdefault("messages", []).append(
            {"role": "user", "text": question}
        )
        st.session_state["messages"].append(
            {"role": "assistant", "answer": answer}
        )


main()
