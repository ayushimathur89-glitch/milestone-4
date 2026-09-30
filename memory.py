"""Short-horizon conversation memory for resolving follow-up questions.

Holds the last `config.MEMORY_MAX_MESSAGES` messages and uses them to rewrite a
follow-up into a standalone question *before* retrieval, so that

    "what about its fees?"

is answered from SBI Flexicap Fund chunks rather than from whatever chunk in the
whole collection happens to contain the word "fees".

The rewrite decides which scheme filter engages, and a wrong filter silently
hides the correct answer: no error, no warning, just confident non-answers.
That is the same failure mode `retrieval.detect_scheme` is written to avoid, and
it is why this module refuses to guess.

**Deliberately no LLM call for the rewrite.** Three reasons:

1. A rewrite would run before retrieval, so it would add a Groq call to every
   follow-up even when the guardrails were about to refuse the turn outright.
2. The rewrite is safety-relevant, not cosmetic. It selects the filter.
3. Rules can be read by a test and fail loudly; a model rewrite fails quietly,
   differently every run, with no regression to catch.

**Anything the rules do not recognise is passed through unchanged.** That is
today's behaviour and a safe floor: an unfiltered search using the user's own
wording retrieves weakly, while a confidently wrong filter retrieves nothing.
An unresolved follow-up is a known limitation, not a silent corruption.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import config
import guardrails
import retrieval

# Lead-ins that carry no retrieval signal and are stripped before rewriting, so
# "ok so what about its fees" does not become a question about "ok". The
# trailing `[\s,.!]*` is inside the repeated group so each filler consumes the
# punctuation that follows it.
_LEAD_IN = re.compile(
    r"^\s*(?:(?:ok(?:ay)?|alright|sure|great|thanks?|hmm+|huh|yes|no|so|now|"
    r"then|well|and|also)\b[\s,.!]*)+",
    re.I,
)
_ASK_ABOUT = re.compile(r"^\s*(?:what|how|tell)\s+about\s+", re.I)
_LEAD_ASK = re.compile(r"^\s*(?:and|so|now|then|ok(?:ay)?)\s+", re.I)

# Possessive stand-ins keep the noun that follows: "its fees" becomes
# "SBI Flexicap Fund's fees" rather than dropping the fees. Losing the noun would
# leave a question with nothing to retrieve, which is worse than not rewriting.
_POSSESSIVE = re.compile(r"\b(?:its|their|his|her)\b\s+(\w+)", re.I)
_PRONOUN = re.compile(r"\b(?:its|their|his|her|it)\b", re.I)
_DEMONSTRATIVE = re.compile(
    r"\b(?:this|that|these|those|the\s+same|the\s+aforementioned)\s+"
    r"(?:fund|scheme|one|product)\b",
    re.I,
)

# "there" is a stand-in for the scheme, but replacing the word with a name is
# ungrammatical: "is there a lock-in?" would become "is SBI Flexicap Fund a
# lock-in?". So this one is handled by scoping the question instead of
# substituting, which reads correctly: "is there a lock-in for SBI Flexicap
# Fund?".
_THERE = re.compile(r"\bthere\b", re.I)

# A question that already has its own subject and verb needs no frame, so the
# rewrite does not add one. "Can I redeem it?" is already a question.
_QUESTION_FRAME = re.compile(
    r"\b(?:what|which|whose|when|where|who|whom|why|how|is|are|was|were|do|does"
    r"|did|can|could|will|would|should|shall|may|might|must|has|have|had"
    r"|am|tell|explain)\b",
    re.I,
)

# An elliptical fragment carries the topic but no subject: "lock-in period?",
# "minimum sip amount?". These are follow-ups in everything but wording, and
# without a rewrite they search the entire collection on one ambiguous term.
_MAX_FRAGMENT_WORDS = 8
_CONTENT_WORDS = re.compile(r"[^a-z0-9'\-/]+", re.I)

# Other fund houses. A turn naming one of these is asking about a different fund
# entirely, so it is not a follow-up to whatever scheme came earlier, and
# inheriting the previous scheme would filter the answer out of existence. This
# is the same reasoning as `retrieval.detect_scheme` refusing a bare "SBI".
_FOREIGN_HOUSES = re.compile(
    r"\b(?:kotak|hdfc|axis|icici|dsp|mirae|tata|nippon|franklin|quant|canara"
    r"|baroda)\b",
    re.I,
)

# A possessive clitic glued to the scheme name, as in "SBI Flexicap Fund's fees"
# after substitution. Stripped before the subject-frame rule reads the rest, so
# the "'s" is not mistaken for the start of a word.
_POSSESSIVE_SUFFIX = re.compile(r"^'?s\b\s*", re.I)

# The fragment template supplies its own article.
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.I)


@dataclass(frozen=True)
class Message:
    """One stored turn."""

    role: str
    text: str


@dataclass(frozen=True)
class Rewrite:
    """The outcome of resolving a question against the conversation.

    `original` is what the user typed and is always preserved, because the UI
    and the audit trail must show the real question even when the retrieval
    question is a rewrite of it. `changed` distinguishes a rewrite from a
    pass-through, so callers can report the rewrite without guessing.
    """

    original: str
    resolved: str
    changed: bool = False
    scheme: str | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def rewrite_notes(self) -> str:
        """One-line explanation for the CLI, empty when nothing was rewritten."""
        if not self.changed:
            return ""
        return (f"follow-up resolved to \"{self.scheme}\" from context "
                f"({', '.join(self.reasons)})")


class Conversation:
    """The last `config.MEMORY_MAX_MESSAGES` messages, oldest first.

    Not thread-safe and not persistent, both deliberately. This is a REPL
    session buffer: a single process, dropped on exit. Persisting user turns to
    disk would turn an in-memory convenience into a record of what someone asked
    about their finances, which is exactly the kind of thing `guardrails`
    refuses to write anywhere.
    """

    def __init__(self, limit: int | None = None) -> None:
        self._limit = max(0, config.MEMORY_MAX_MESSAGES if limit is None else limit)
        self._messages: list[Message] = []

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def add(self, role: str, text: str) -> bool:
        """Store one turn, trimming to the newest `limit` messages.

        Returns False when the turn was not stored, which happens for blank
        input and for anything `guardrails.detect_pii` recognises. Holding a PAN
        or an account number in a buffer that a later turn can quote back is a
        leak even if the buffer never reaches disk, so PII is refused at the
        door rather than filtered at the exit.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return False
        if guardrails.detect_pii(cleaned):
            return False
        self._messages.append(Message(role=role, text=cleaned))
        overflow = len(self._messages) - self._limit
        if overflow > 0:
            del self._messages[:overflow]
        return True

    def clear(self) -> None:
        self._messages.clear()

    def referenced_scheme(self) -> str | None:
        """The scheme the user was last talking about, or None.

        Newest-first, and user turns are preferred over answers: a user turn
        naming a scheme is a statement of what they are asking about, whereas
        the bot's own answer may name other schemes while comparing facts. The
        answers are still searched as a fallback, because a follow-up often
        follows a turn whose subject only the answer spells out.
        """
        for roles in (("user",), ("user", "assistant")):
            for message in reversed(self._messages):
                if message.role in roles:
                    scheme = retrieval.detect_scheme(message.text)
                    if scheme:
                        return scheme
        return None

    def transcript(self) -> str:
        """Readable dump of the buffer, for the CLI's /history command."""
        if not self._messages:
            return "  (empty)"
        width = max(len(m.role) for m in self._messages)
        return "\n".join(f"  {m.role:<{width}} : {m.text}" for m in self._messages)


def _content_words(text: str) -> list[str]:
    return [w for w in _CONTENT_WORDS.sub(" ", text).split() if w]


def _is_fragment(text: str) -> bool:
    """Whether a turn is a topic with no subject: "lock-in period?".

    Requires all three: no question frame at all, few enough words to be a noun
    phrase, and no foreign fund house. Requiring the absence of a question word
    is what keeps "is it taxable?" out of this branch, where the stand-in is
    already found and handled by substitution instead.
    """
    words = _content_words(text)
    if not words or len(words) > _MAX_FRAGMENT_WORDS:
        return False
    if _QUESTION_FRAME.search(text):
        return False
    return not _FOREIGN_HOUSES.search(text)


def _substitute(text: str, scheme: str) -> str:
    """Replace stand-ins for the scheme with the scheme's name."""
    text = _POSSESSIVE.sub(lambda m: f"{scheme}'s {m.group(1)}", text)
    text = _PRONOUN.sub(scheme, text)
    text = _DEMONSTRATIVE.sub(scheme, text)
    return re.sub(r"\s+", " ", text).strip(" ?!.,")


def _scope(text: str, scheme: str) -> str:
    """Add the scheme as a trailing scope rather than replacing a word.

    Used for "there" and for bare fragments, where inserting the name mid-
    sentence would be ungrammatical. "lock-in period" becomes "lock-in period
    for SBI Flexicap Fund".
    """
    stripped = text.rstrip()
    trailing = stripped[len(stripped.rstrip(" ?.!").rstrip()):]
    core = stripped.rstrip(" ?.!").rstrip()
    if not core:
        return text
    if retrieval.detect_scheme(core):
        return text
    return f"{core} for {scheme}{trailing}"


def _frame(text: str, scheme: str) -> str | None:
    """Turn a rewritten fragment into a question that names the scheme.

    "SBI Flexicap Fund fees" becomes "What is the fees for SBI Flexicap Fund?"
    rather than "What is the SBI Flexicap Fund fees?", because in the second
    form the scheme reads as the subject and the model treats the scheme name
    as the thing being asked about.

    Returns None when the rewrite left no predicate to ask about, as in "what
    about it?" -> "SBI Flexicap Fund". There is no question to recover, so the
    caller keeps the original wording instead of inventing one.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    if _QUESTION_FRAME.search(text):
        return text if text.endswith(("?", ".")) else f"{text}?"

    subject = re.match(rf"^{re.escape(scheme)}\b\s*(.*)$", text, re.I)
    if subject:
        rest = _POSSESSIVE_SUFFIX.sub("", subject.group(1)).strip()
        if not rest:
            return None
        return f"What is the {rest} for {scheme}?"
    return f"What is the {text} for {scheme}?"


def resolve_followup(question: str, conversation: Conversation | None) -> Rewrite:
    """Rewrite a follow-up into a standalone question, or pass it through.

    The order of the checks is the whole safety argument: a turn that already
    names a scheme is never rewritten, and neither is one that names a
    different fund house. Only after both is the conversation consulted for
    what the stand-in referred to.
    """
    original = (question or "").strip()
    result = Rewrite(original=original, resolved=original)
    if not original or not config.MEMORY_ENABLED or conversation is None:
        return result
    if not len(conversation):
        return result

    # Self-contained already: rewriting could only lose information.
    if retrieval.detect_scheme(original):
        return result
    # Asking about a different fund house is not a follow-up to this thread.
    if _FOREIGN_HOUSES.search(original):
        return result

    scheme = conversation.referenced_scheme()
    if not scheme:
        return result

    body = _ASK_ABOUT.sub("", _LEAD_IN.sub("", original))
    body = _LEAD_ASK.sub("", body).strip()
    if not body:
        # The whole turn was filler ("thanks", "ok", "sure"). There is no
        # question in it to recover, and treating the leftover word as a topic
        # produced "What is the thanks for SBI Flexicap Fund!".
        return result

    if _POSSESSIVE.search(body) or _PRONOUN.search(body) or _DEMONSTRATIVE.search(body):
        reason = ("possessive pronoun" if _POSSESSIVE.search(body)
                  else "pronoun" if _PRONOUN.search(body)
                  else "demonstrative reference")
        rewritten = _frame(_substitute(body, scheme), scheme)
    elif _THERE.search(body):
        reason = "'there' as a stand-in"
        rewritten = _frame(_scope(body, scheme), scheme)
    elif _is_fragment(body):
        # Built directly rather than through `_scope`, because `_scope` has
        # already appended the scheme and `_frame` would append it a second
        # time. Naming the topic first reads as a question instead of a note.
        # A leading article is dropped, or the template produces "What is the
        # the fund house for ...".
        reason = "topic fragment"
        topic = _LEADING_ARTICLE.sub("", body.rstrip(" ?!."))
        rewritten = f"What is the {topic} for {scheme}?"
    else:
        return result

    if not rewritten or rewritten.strip().lower() == original.lower():
        return result
    return Rewrite(
        original=original,
        resolved=rewritten,
        changed=True,
        scheme=scheme,
        reasons=(reason,),
    )
