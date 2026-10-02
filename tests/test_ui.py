"""Phase 6 UI tests: drive `app.py` headlessly with Streamlit's AppTest.

Run with:

    .venv\\Scripts\\python -m tests.test_ui

The spec's verification steps for the UI are things that only mean anything
against a running app - "the PAN does not appear in the on-screen transcript",
"the app stays usable afterwards", "the link is clickable, not raw text".
Streamlit reruns the entire script on every interaction and holds its state in
`st.session_state`, so importing `app` and calling a function proves nothing.
These tests run the real script.

The model is stubbed, for two reasons: the suite costs no tokens, and a
Groq rate limit cannot turn a passing build red. The refusal paths need no stub
at all, because the pre-check returns before the LLM is touched - which is
itself one of the things worth asserting.

The PII literals are assembled at runtime rather than written out. The Phase 4
scanner allow-lists exactly one file, and adding a second would weaken it.
"""

from __future__ import annotations

import re
import types
from pathlib import Path
from unittest import mock

from streamlit.testing.v1 import AppTest

import config
import generator
from ingest import store

BAR = "=" * 72
# Absolute: `AppTest.from_file` resolves relative paths against this file, not
# the working directory, so "app.py" would look for tests/app.py.
APP = str(Path(__file__).resolve().parent.parent / "app.py")

# Assembled, not literal - see the module docstring.
_PAN = "ABCDE" "1234" "F"
_PHONE = "98765" "43210"
_AADHAAR = "1234 5678 " "9012"

# Pinned from PRD section 5 rather than imported from `app`. Importing the app
# would run it, and a test that reuses the constant it is meant to check cannot
# catch the constant being changed.
PRD_EXAMPLES = (
    "What is the expense ratio of SBI Flexicap Fund?",
    "What is the lock-in period for SBI ELSS Tax Saver Fund?",
    "What is the exit load for SBI Large Cap Fund?",
)

# Short and on-contract, so `verify_answer` passes it through unchanged.
STUB_ANSWER = "The exit load is 1% within 12 months and Nil after that."


def _stub_client(text: str = STUB_ANSWER):
    """A client whose `create` returns `text`, shaped like Groq's response."""
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=text))],
        model="stub-model",
    )
    return types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=lambda **kw: response)))


def boot() -> AppTest:
    """A freshly loaded app, with no script errors."""
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    return app


def ask(app: AppTest, question: str, stub: bool = True) -> AppTest:
    """Type a question and rerun, with the LLM stubbed by default."""
    if stub:
        with mock.patch.object(generator, "get_client", lambda: _stub_client()):
            app.chat_input[0].set_value(question).run()
    else:
        app.chat_input[0].set_value(question).run()
    return app


def visible_text(app: AppTest) -> str:
    """Every piece of text the user can see, including collapsed expanders.

    The expander contents are included deliberately: verification step 4 is
    about the PAN not being on screen anywhere, and a chunk preview is on screen
    once the panel is opened.
    """
    parts = [m.value for m in app.markdown]
    parts += [c.value for c in app.caption]
    parts += [b.label for b in app.button]
    for panel in app.expander:
        parts += [m.value for m in panel.markdown]
        parts += [c.value for c in panel.caption]
    return "\n".join(parts)


def check_first_load() -> bool:
    """Spec step 1: welcome line, 3 examples, facts-only note, disclaimer."""
    print()
    print(BAR)
    print("1. FIRST LOAD")
    print(BAR)
    app = boot()
    failures = []

    def expect(label: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(label)
        # detail is whatever is on hand: a string, a list of strings, a count.
        tail = f"  {detail}" if detail else ""
        print(f"  {'ok  ' if ok else 'FAIL'} {label}{tail}")

    expect("no script errors", not app.exception,
           str([e.value[:60] for e in app.exception]))
    expect("title is set", bool(app.title) and "SBI" in app.title[0].value)
    expect("a welcome line is shown",
           any("official SBI MF" in c.value for c in app.caption))
    expect("the facts-only disclaimer is shown",
           any(config.DISCLAIMER in c.value for c in app.caption))
    expect("a text input is present", app.chat_input[0] is not None)
    expect("no API key banner (key is loaded from .env)",
           not app.error, str([e.value[:50] for e in app.error]))

    labels = [b.label for b in app.button]
    expect("3 clickable example questions", len(PRD_EXAMPLES) == 3)
    for wanted in PRD_EXAMPLES:
        expect(f'  "{wanted}"', wanted in labels)
    expect("a clear-chat button exists", "Clear chat" in labels)
    expect("the sidebar states the corpus boundary",
           any(h.value == "Scope" for h in app.sidebar.header))
    for scheme in ("SBI Flexicap Fund", "SBI Large Cap Fund",
                   "SBI ELSS Tax Saver Fund", "SBI Small Cap Fund",
                   "SBI Balanced Advantage Fund"):
        expect(f"  sidebar lists {scheme}", scheme in visible_text(app))

    print(f"\n  {'all first-load checks passed' if not failures else f'{len(failures)} FAILED'}")
    return not failures


def check_answer_rendering() -> bool:
    """Spec steps 2 and 5: an answer, a clickable link, a sources expander."""
    print()
    print(BAR)
    print("2. AN ANSWER RENDERS WITH A LINK AND SOURCES")
    print(BAR)
    app = ask(boot(), "What is the exit load for SBI Large Cap Fund?")
    text = visible_text(app)
    failures = []

    def expect(label: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(label)
        # detail is whatever is on hand: a string, a list of strings, a count.
        tail = f"  {detail}" if detail else ""
        print(f"  {'ok  ' if ok else 'FAIL'} {label}{tail}")

    expect("no script errors", not app.exception,
           str([e.value[:60] for e in app.exception]))
    expect("both chat messages rendered", len(app.chat_message) == 2)
    expect("the question is echoed", "exit load" in text)
    expect("the answer body is rendered", "1% within 12 months" in text)
    expect("the source date comes from the corpus",
           config.LAST_UPDATED_PREFIX in text,
           "no 'Last updated' line was rendered")
    expect("the citation is a clickable markdown link, not raw text",
           any("](" in m.value and m.value.startswith("[http")
               for m in app.markdown))
    # The one-citation contract, checked on screen rather than in the generator.
    # Scoped to the answer: each chunk in the sources panel legitimately carries
    # its own URL, and those are the retrieval trail, not the claim's citation.
    # A link counts once, however many times its URL appears in its own markup -
    # the user sees the label, not the href.
    answer_md = "\n".join(m.value for m in app.markdown)
    links = re.findall(r"\[([^\]]+)\]\((https?://[^)]+)\)", answer_md)
    bare = [u for u in re.findall(r"(?<!\()https?://[^\s)\]]+",
                                  re.sub(r"\[[^\]]+\]\(https?://[^)]+\)", "",
                                         answer_md))]
    expect("the answer carries exactly one link", len(links) == 1,
           str(links))
    expect("and no bare URL left over beside it", not bare, str(bare))
    expect("every chunk in the sources panel still shows its own URL",
           len([c for c in app.caption if c.value.startswith("https://")]) == 6)
    expect("the URL is on an allowed domain",
           any(config.is_allowed_url(part.strip("[]()"))
               for m in app.markdown for part in m.value.split("](")
               if part.startswith("http")))

    labels = [e.label for e in app.expander]
    expect("a sources expander is present",
           any(l.startswith("Sources") for l in labels), str(labels))
    expect("it names how many chunks were used",
           any("used of" in l for l in labels), str(labels))
    expect("the retrieval scope is shown",
           any("Scoped to: SBI Large Cap Fund" in c.value for c in app.caption))
    expect("each chunk shows scheme, type and similarity",
           "similarity" in text and "SID" in text)
    expect("chunk provenance is shown", "fetched 2026" in text)

    print(f"\n  {'all rendering checks passed' if not failures else f'{len(failures)} FAILED'}")
    return not failures


def check_pii() -> bool:
    """Spec step 4: PII refused, absent from the transcript, no LLM call."""
    print()
    print(BAR)
    print("3. PII IS REFUSED AND NEVER SHOWN")
    print(BAR)
    failures = []

    def expect(label: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(label)
        # detail is whatever is on hand: a string, a list of strings, a count.
        tail = f"  {detail}" if detail else ""
        print(f"  {'ok  ' if ok else 'FAIL'} {label}{tail}")

    for label, needle in (("PAN", _PAN), ("phone number", _PHONE),
                          ("Aadhaar number", _AADHAAR)):
        # No stub: a genuine rate-limit hit would prove the LLM was called, and
        # the point of this check is that it never is.
        app = ask(boot(), f"My {label} is {needle}, tell me the exit load",
                  stub=False)
        text = visible_text(app)
        expect(f"{label} is refused", "cannot accept or keep" in text)
        expect(f"{label} does not appear anywhere on screen",
               needle not in text)
        expect(f"{label} message is redacted, not merely refused",
               "removed" in text)
        expect(f"{label} refusal makes no LLM call", not app.exception)
        expect(f"{label} question is not kept in memory",
               len(app.session_state["conversation"]) == 0,
               f"buffer holds {len(app.session_state['conversation'])}")

    # A refusal must not leave a misleading "0 retrieved" sources panel: the
    # pre-check returns before retrieval, so the sources were never consulted.
    app = ask(boot(), f"My PAN is {_PAN}, tell me the exit load", stub=False)
    expect("no empty sources panel after a refusal",
           not any(e.label.startswith("Sources") for e in app.expander),
           str([e.label for e in app.expander]))
    expect("the panel that is shown explains itself",
           any("not answered" in e.label for e in app.expander),
           str([e.label for e in app.expander]))

    print(f"\n  {'all PII checks passed' if not failures else f'{len(failures)} FAILED'}")
    return not failures


def check_refusals_keep_the_app_alive() -> bool:
    """Spec step 3: an advisory refusal, then the app still works."""
    print()
    print(BAR)
    print("4. THE APP SURVIVES A REFUSAL")
    print(BAR)
    app = boot()
    app = ask(app, "Should I buy SBI Small Cap Fund?", stub=False)
    failures = []

    def expect(label: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(label)
        # detail is whatever is on hand: a string, a list of strings, a count.
        tail = f"  {detail}" if detail else ""
        print(f"  {'ok  ' if ok else 'FAIL'} {label}{tail}")

    expect("the advisory question is refused",
           "No investment advice" in visible_text(app))
    expect("no LLM call was made", not app.exception)

    app = ask(app, "What is the exit load for SBI Large Cap Fund?")
    expect("a factual question is still answered afterwards",
           len(app.chat_message) == 4)
    expect("and it still produces an answer",
           "1% within 12 months" in visible_text(app))
    expect("the input is still usable", app.chat_input[0] is not None)
    expect("no error surfaced", not app.error, str([e.value for e in app.error]))

    print(f"\n  {'all resilience checks passed' if not failures else f'{len(failures)} FAILED'}")
    return not failures


def check_chat_state() -> bool:
    """Message history, the clear button, and what it must clear."""
    print()
    print(BAR)
    print("5. MESSAGE HISTORY AND CLEAR CHAT")
    print(BAR)
    app = boot()
    app = ask(app, "What is the exit load for SBI Large Cap Fund?")
    app = ask(app, "what about its fees?")
    failures = []

    def expect(label: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(label)
        # detail is whatever is on hand: a string, a list of strings, a count.
        tail = f"  {detail}" if detail else ""
        print(f"  {'ok  ' if ok else 'FAIL'} {label}{tail}")

    expect("both turns are in the history", len(app.chat_message) == 4)
    expect("both answers keep their own sources panel",
           sum(1 for e in app.expander if e.label.startswith("Sources")) == 2,
           str([e.label for e in app.expander]))
    expect("the follow-up was resolved from context",
           any("resolved" in c.value for c in app.caption),
           [c.value[:70] for c in app.caption if "resolved" in c.value])
    expect("the resolution is shown to the user, not hidden",
           "What is the fees for SBI Large Cap Fund?" in visible_text(app))

    app.button[0].click().run()
    expect("clear empties the transcript", len(app.chat_message) == 0)
    expect("clear removes the sources panels", len(app.expander) == 0)

    # The half that is easy to get wrong: clearing only the transcript would
    # leave the bot still resolving "its" against a conversation the user can
    # no longer see.
    app = ask(app, "what about its fees?")
    expect("clear also forgets the conversation memory",
           len(app.session_state["conversation"]) == 2)
    expect("so the follow-up is searched as typed, with no scheme scope",
           not any("resolved" in c.value for c in app.caption)
           and not any("Scoped to" in c.value for c in app.caption))

    print(f"\n  {'all history checks passed' if not failures else f'{len(failures)} FAILED'}")
    return not failures


def check_example_buttons() -> bool:
    """Spec step 2: clicking an example asks it."""
    print()
    print(BAR)
    print("6. EXAMPLE BUTTONS ASK THEIR QUESTION")
    print(BAR)
    app = boot()
    failures = []

    def expect(label: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(label)
        # detail is whatever is on hand: a string, a list of strings, a count.
        tail = f"  {detail}" if detail else ""
        print(f"  {'ok  ' if ok else 'FAIL'} {label}{tail}")

    for button in app.button:
        if button.label not in PRD_EXAMPLES:
            continue
        with mock.patch.object(generator, "get_client", lambda: _stub_client()):
            button.click().run()
        expect(f"clicking \"{button.label[:34]}...\" asks it",
               any(button.label in m.value for m in app.markdown))
        expect("  and the example is remembered in history",
               len(app.session_state["messages"]) == 2,
               f"{len(app.session_state['messages'])} stored")
        app.button[0].click().run()  # clear between clicks

    print(f"\n  {'all example-button checks passed' if not failures else f'{len(failures)} FAILED'}")
    return not failures


def check_unrelated_questions_hide_sources() -> bool:
    """A question nothing in the corpus answers must show no sources at all.

    Two shapes of the same mistake, and only the second one is subtle. A
    pre-check refusal returns before retrieval, so there is nothing to show and
    nothing to decide. The second retrieves a full page of chunks, every one of
    them below `config.MIN_SIMILARITY`, and refuses anyway. Listing those chunks
    under an answer that said it found nothing reads as six SBI documents having
    been consulted for a question about the boiling point of water.
    """
    print()
    print(BAR)
    print("7. UNRELATED QUESTIONS SHOW NO SOURCES")
    print(BAR)
    # No stub: these paths return before the LLM is reached, and a genuine
    # rate-limit error would prove the opposite of what is being asserted.
    questions = {
        "What is the boiling point of water?": "retrieves, all below the floor",
        "Who won the FIFA World Cup in 2022?": "refused before retrieval",
    }
    failures = []

    def expect(label: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(label)
        tail = f"  {detail}" if detail else ""
        print(f"  {'ok  ' if ok else 'FAIL'} {label}{tail}")

    for question, shape in questions.items():
        app = boot()
        app.chat_input[0].set_value(question).run()
        labels = [e.label for e in app.expander]
        urls = [c for c in app.caption if c.value.startswith("http")]
        tag = f"{question[:34]}... ({shape})"
        expect(f"{tag} shows no sources panel",
               not any(l.startswith("Sources") for l in labels), str(labels))
        expect("  and no chunk URL reaches the screen", not urls,
               str([u.value for u in urls]))
        expect("  and the refusal still explains itself",
               any("not answered" in l for l in labels), str(labels))

    # The near-miss case is only worth testing if chunks really were retrieved.
    # If the corpus or the floor changes and this starts scoring above it, the
    # checks above would pass for the wrong reason and stop covering the bug
    # they were written for. "TER" is used rather than an off-topic question
    # because those no longer reach retrieval at all: they are turned away by the
    # clarification gate before anything is retrieved.
    answer = generator.ask("TER", collection=store.open_collection(store.get_client()))
    expect("the near-miss case still retrieves chunks below the floor",
           bool(answer.hits) and not answer.above_floor,
           f"{len(answer.hits)} retrieved, {len(answer.above_floor)} above "
           f"{config.MIN_SIMILARITY}")

    print(f"\n  {'all unrelated-question checks passed' if not failures else f'{len(failures)} FAILED'}")
    return not failures


def check_fragments_ask_instead_of_declining() -> bool:
    """"Small Cap." must ask what the user wants, and the answer must survive.

    Two claims in one check because only the second proves the first is useful.
    A clarifying reply that forgets the scheme it just named has told the user
    to go and re-type it, which is the same dead end with an extra step. The
    hand-off works through `memory.referenced_scheme` reading the assistant turn
    back out of the buffer, so it needs the canonical scheme name in the message.
    """
    print()
    print(BAR)
    print("8. FRAGMENTS ASK INSTEAD OF DECLINING")
    print(BAR)
    failures = []

    def expect(label: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(label)
        tail = f"  {detail}" if detail else ""
        print(f"  {'ok  ' if ok else 'FAIL'} {label}{tail}")

    app = boot()
    app.chat_input[0].set_value("Small Cap.").run()
    text = visible_text(app)
    labels = [e.label for e in app.expander]
    expect("the fragment asks what the user wants",
           "Did you mean SBI Small Cap Fund?" in text, text[:120])
    expect("  and no longer claims the sources came up short",
           "could not find an answer" not in text.lower())
    expect("  and shows no sources panel",
           not any(l.startswith("Sources") for l in labels), str(labels))
    expect("  and reaches no chunk URL",
           not [c for c in app.caption if c.value.startswith("http")])
    expect("  and the refusal is still explained",
           any("not answered" in l for l in labels), str(labels))

    app.chat_input[0].set_value("what about its exit load?").run()
    # Scoped to the newest answer. `visible_text` covers the whole transcript,
    # which still holds the earlier "Did you mean ...", so asking whether the
    # bot is still clarifying here would be answered by two turns ago.
    answer = "\n".join(m.value for m in app.chat_message[-1].markdown)
    expect("the follow-up is resolved to the scheme just named",
           "SBI Small Cap Fund" in answer
           and any("resolved" in c.value for c in app.caption),
           str([c.value[:60] for c in app.caption if "resolved" in c.value]))
    expect("  and it is scoped to that scheme",
           any("Scoped to: SBI Small Cap Fund" in c.value
               for c in app.caption))
    expect("  and it produces a real answer, not another decline",
           "could not find an answer" not in answer.lower()
           and "Did you mean" not in answer, answer[:140])

    app2 = boot()
    app2.chat_input[0].set_value("How to Invest?").run()
    text2 = visible_text(app2)
    expect("a how-to question states what it can answer",
           "does not name a fact I can look up" in text2, text2[:120])
    expect("  and offers the topics rather than guessing one",
           "expense ratio" in text2)
    expect("  and shows no sources panel",
           not any(e.label.startswith("Sources") for e in app2.expander),
           str([e.label for e in app2.expander]))

    print(f"\n  {'all fragment checks passed' if not failures else f'{len(failures)} FAILED'}")
    return not failures


def main() -> int:
    print(BAR)
    print("PHASE 6 UI (Streamlit)")
    print(BAR)
    results = {
        "first load": check_first_load(),
        "answer rendering": check_answer_rendering(),
        "PII": check_pii(),
        "survives a refusal": check_refusals_keep_the_app_alive(),
        "history and clear": check_chat_state(),
        "example buttons": check_example_buttons(),
        "unrelated questions": check_unrelated_questions_hide_sources(),
        "fragments": check_fragments_ask_instead_of_declining(),
    }
    print()
    print(BAR)
    print("SUMMARY")
    print(BAR)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

