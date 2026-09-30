"""Constraint layer, built before any LLM is wired in.

implementation.md Phase 4: guardrails are implemented and provable in
isolation, with no Groq calls and no vector store, so they cost nothing to
test and cannot be quietly weakened by a prompt change later.

Two halves:

**Pre-check (routing).** `classify` sorts an incoming question into
`factual`, `advisory`, `performance`, `pii`, or `off_topic` and returns the
exact response the user should see for the refused classes.

The point of routing rather than prompting is structural. A performance
question never reaches the LLM, so the model is never in a position to
invent a return figure; a PII-bearing query never reaches the LLM, so the
identifier is never in a completion request that might get logged upstream.
Those two PRD constraints hold because of how the code is wired, not because
a prompt asks nicely.

**Post-check (answer contract).** `verify_answer` enforces the three rules
from the PRD - at most 3 sentences, exactly one source link, and a
`Last updated` line that app code writes rather than the model. On failure
it repairs what it can and falls back to a safe message otherwise.

A note on the two judgement calls, since they are the ones a reviewer
should push back on if they disagree:

- **PII is matched by pattern, so it is a screen rather than a proof.** The
  patterns cover Indian PAN, Aadhaar, account numbers, OTPs, emails and
  phone numbers, which is what the PRD enumerates. Anything outside that list
  is not detected. Treated as a speed bump on a public FAQ bot, not a
  compliance boundary for regulated data.
- **Off-topic is deliberately high-precision.** Only clearly non-fund topics
  are refused here. Anything ambiguous is let through to retrieval, where the
  `config.MIN_SIMILARITY` gate in Phase 5 and the "I don't know" response
  catch it instead. Refusing on a guess would block legitimate questions
  about schemes this bot knows nothing about, which is the worse failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import config

# --- Educational links ---
# Verified reachable (HTTP 200) while building this module, and all on
# config.ALLOWED_DOMAINS. Checked rather than assumed because the PRD
# requires the refusal to include a *working* link, and several plausible
# AMFI paths return 404.
LINK_AMFI = "https://www.amfiindia.com/aboutamfi"
LINK_SEBI = "https://www.sebi.gov.in/"
LINK_SBIMF = "https://www.sbimf.com/"

EDUCATIONAL_LINKS = (LINK_AMFI, LINK_SEBI, LINK_SBIMF)

# --- Refusal messages ---
# Every one states the facts-only position and carries a working link, per
# implementation.md Phase 4 verification step 2. None of them echo the
# user's text back, which is what keeps a refused PII query out of the
# transcript and the logs.

MSG_ADVISORY = (
    "I can share the published facts about a scheme, but choosing an "
    "investment is yours to make, so I will not tell you what to buy. "
    f"{config.DISCLAIMER} AMFI explains how mutual funds work, which is a "
    f"better starting point than a recommendation from a bot: {LINK_AMFI}"
)

MSG_PERFORMANCE = (
    "I do not make performance claims or compare returns between schemes. "
    f"{config.DISCLAIMER} Each scheme's own published returns are in its "
    f"official factsheet: {LINK_SBIMF}"
)

MSG_PII = (
    "I cannot accept or keep personal identifiers such as a PAN, Aadhaar "
    "number, account number, OTP, email address or phone number, so I have "
    f"not saved your message. {config.DISCLAIMER} Please ask again without "
    f"including personal details: {LINK_SEBI}"
)

MSG_OFF_TOPIC = (
    "That does not look like a question about a mutual fund scheme, so I "
    f"have nothing to look up. {config.DISCLAIMER} If you have a scheme "
    "question, try asking about its expense ratio, exit load, benchmark, "
    f"minimum SIP, riskometer or lock-in period: {LINK_AMFI}"
)

MSG_NO_CONTEXT = (
    "I could not find an answer to that in the official SBI mutual fund "
    f"documents I have. {config.DISCLAIMER} The official factsheet covers "
    f"expense ratio, exit load, benchmark, minimum SIP and riskometer: "
    f"{LINK_SBIMF}"
)


# --- PII screen ---
# Ordered most-specific first. The lookarounds matter: without them a PAN
# regex happily matches a 10-character slice of a 12-digit Aadhaar, and a
# 10-digit phone pattern matches inside a longer account number.

_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pan", re.compile(r"(?<![A-Za-z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Za-z0-9])")),
    ("aadhaar", re.compile(r"(?<!\d)(?:[XxX]?[Zz]{1,4}[\s-]?)?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"(?<![\d.])(?:\+?91[\s-]?)?[6-9]\d{9}(?![\d.])")),
    ("account", re.compile(
        r"(?i)\b(?:a/?c|acc(?:ount)?|folio|bank)\s*(?:no|number|#)?\s*[:\-]?\s*"
        r"(?:\d[\s-]?){9,18}\b")),
    ("otp", re.compile(
        r"(?i)\b(?:otp|one[\s-]time\s+(?:password|code)|verification\s+code)\b"
        r"[^\d]{0,12}\d{4,8}\b")),
)

# Digits standing alone are not PII; a bare number in a fund FAQ is far more
# likely to be a NAV or a folio count than an account number, so a
# context-free digit run is deliberately not flagged.
_PII_LABELS = tuple(label for label, _ in _PII_PATTERNS)


def detect_pii(text: str) -> str | None:
    """Return the label of the first PII kind found in `text`, else None.

    Returns only the label, never the matched value, so that a refusal can
    name the problem without reproducing the sensitive string.
    """
    if not text:
        return None
    for label, pattern in _PII_PATTERNS:
        if pattern.search(text):
            return label
    return None


# --- Classifiers ---

# Opinion or recommendation seeking.
_ADVISORY = (
    r"\bshould\s+(?:i|we|you|my)\b",
    r"\b(?:is|are)\s+it\s+(?:good|worth|safe|wise|better|a\s+good\s+idea)\b",
    r"\b(?:is|are)\s+[\w\s-]{0,40}\bgood\s+for\s+(?:me|us|my|our)\b",
    r"\b(?:can|could|should)\s+(?:i|we)\s+(?:buy|sell|invest|redeem|switch|purchase)\b",
    r"\b(?:do|does|would|will)\s+it\s+make\s+sense\b",
    r"\brecommend\w*\b",
    r"\badvise\s+me\b",
    r"\bworth\s+(?:investing|buying|it)\b",
    r"\bsuitab\w+\b",
    r"\bwhich\s+(?:should|one)\b",
    r"\bbest\s+(?:for\s+)?(?:me|my|us|our)\b",
    r"\bopinions?\s+(?:on|about)\b",
    r"\bhelp\s+me\s+choose\b",
    r"\bwhat\s+should\s+i\b",
)

# Return or performance comparison.
_PERFORMANCE = (
    r"\b(?:best|top|highest|strongest|worst|lowest)\s+"
    r"(?:perform\w*|return\w*|performer|nav\s+growth|growth)\b",
    r"\bperform\w*\s+(?:the\s+)?(?:best|better|worse|well|poorly|worst)\b",
    r"\bcompare\s+(?:the\s+|their\s+)?(?:return\w*|perform\w*|nav|growth|funds?)\b",
    r"\boutperform\w*\b",
    r"\b(?:return\w*|cagr|nav\s+growth)\s+(?:of|for|of\s+[\w\s]{0,20})\s*"
    r"(?:vs\.?|versus|and)\b",
    r"\bwhich\s+(?:fund|scheme|one)\s+(?:gave|has|had|performed|returned)\b",
    r"\b(?:return\w*|performance)\s+(?:last|past|previous)\s+"
    r"(?:year|month|quarter|1|2|3|5)\b",
    r"\bhow\s+much\s+did\s+.+\s+(?:return|grow|earn)\b",
    r"\btop\s+performer\w*\b",
)

# Clearly not a mutual-fund question. Intentionally conservative: only
# unambiguous topics, because a false refusal is worse than routing an odd
# question to retrieval and letting the similarity gate handle it.
_OFF_TOPIC = (
    r"\b(?:recipe|cook\w*|bake\w*|baking|sourdough|pasta|cuisine|ingredients)\b",
    r"\b(?:cricket|ipl|test\s+match|odi\b|t20|goal[s]?\s+scored|innings)\b",
    r"\b(?:election|vote[s]??\s+for|parliament|modi|rahul|senate|congress)\b",
    r"\b(?:weather|forecast|rainfall|temperature\s+today|cyclone)\b",
    r"\b(?:doctor|diabet\w+|symptom\w*|diagnos\w+|medicine|hospital|covid)\b",
    r"\b(?:bitcoin|crypto\w*|ethereum|blockchain|doge\s?coin)\b",
    # Both word orders: "gold price" and "price of gold".
    r"\b(?:gold|silver|bitcoin|crude\s+oil)\s+(?:price|rate)\b",
    r"\b(?:price|rate|value)\s+of\s+(?:gold|silver|bitcoin)\b",
    r"\b(?:horoscope|zodiac|astrolog\w+|nakshatra)\b",
    r"\b(?:joke|poem|song\s+lyrics|write\s+a\s+story)\b",
    r"\b(?:python|java\s+program|javascript|sql\s+query|debug\s+this\s+code)\b",
    r"\b(?:football|cricket|tennis|badminton|ipl)\b",
    # Named competitions and bodies, not just the sports. "Who won the FIFA
    # World Cup in 2022?" contains no sport word, so the line above missed it
    # and the question was routed to retrieval, where the best unrelated chunk
    # still cleared the similarity floor and the model was asked a football
    # question out of mutual-fund documents.
    r"\b(?:world\s?cup|fifa|uefa|olympic\w*|champions\s?league|"
    r"premier\s?league|la\s?liga|serie\s?a\b|bundesliga|world\s?series|"
    r"nba\b|nfl\b|formula\s?1\b|f1\b|ashes\b|ryder\s?cup)\b",
    r"\bwho\s+won\b",
    # General-knowledge shapes that a mutual-fund corpus will never answer but
    # that carry no topic word to match on. These measured high enough to
    # matter: "Who won the election in Maharashtra?" reaches 0.469 and "What
    # time does the Mumbai train leave?" 0.460, against 0.539 for the *least*
    # answerable fund question in the calibration set.
    r"\b(?:timetable|train\s+time|flight\s+time|bus\s+time|next\s+train|"
    r"next\s+flight|how\s+far\s+is|how\s+tall\s+is|how\s+much\s+does\s+\w+\s+weigh)\b",
    r"\b(?:train|bus|flight|ship|ferry)\s+(?:leave|depart|arriv|reach|cancel)\w*",
    r"\btime\s+does\s+the\s+\w+\s+(?:train|bus|flight)\b",
    r"\bcapital\s+of\s+\w+",
    r"\bwho\s+(?:is|was|are|were)\s+the\s+(?:president|prime\s+minister|"
    r"ceo\s+of\s+\w+)\b",
    r"\b(?:homework|essay\s+on|exam\s+question)\b",
)

# Vocabulary that signals the question is plausibly about a fund at all.
_FUND_VOCAB = re.compile(
    r"\b(?:mutual\s?fund|fund|scheme|elss|flexicap|large\s?cap|small\s?cap|"
    r"mid\s?cap|balanced|advantage|dividend|index|sip|nav|exit\s?load|"
    r"expense\s?ratio|ter\b|benchmark|riskometer|lock[\s-]?in|folio|"
    r"switch\w*|redeem\w*|invest\w*|tax\s?sav\w*|80c|equity|debt|"
    r"sbi|amfi|factsheet|sebi|ter\b)\b",
    re.I,
)

_ADVISORY_RE = re.compile("|".join(_ADVISORY), re.I)
_PERFORMANCE_RE = re.compile("|".join(_PERFORMANCE), re.I)
_OFF_TOPIC_RE = re.compile("|".join(_OFF_TOPIC), re.I)


@dataclass(frozen=True)
class Decision:
    """The routing outcome for one question, plus what to show the user."""

    category: str
    allowed: bool
    message: str = ""
    reason: str = ""
    pii_kind: str = ""
    problems: tuple[str, ...] = field(default=(), compare=False)

    @property
    def refused(self) -> bool:
        return not self.allowed


def classify(question: str) -> Decision:
    """Route a question. `allowed=True` means "factual, go to retrieval".

    Order is deliberate. PII is screened first so a query that mixes a PAN
    with a performance request is refused for the identifier, and the
    identifier never propagates. Advisory is checked before performance
    because "should I buy the fund that returned the most?" is a request for
    a recommendation, and that is the more restrictive of the two.
    """
    text = (question or "").strip()
    if not text:
        return Decision("off_topic", False, MSG_OFF_TOPIC, "empty question")

    pii = detect_pii(text)
    if pii:
        return Decision("pii", False, MSG_PII,
                        f"query contained a {pii}", pii_kind=pii)

    if _ADVISORY_RE.search(text):
        return Decision("advisory", False, MSG_ADVISORY,
                        "asks for a recommendation or suitability opinion")

    if _PERFORMANCE_RE.search(text):
        return Decision("performance", False, MSG_PERFORMANCE,
                        "asks for a returns or performance comparison")

    if _OFF_TOPIC_RE.search(text) and not _FUND_VOCAB.search(text):
        return Decision("off_topic", False, MSG_OFF_TOPIC,
                        "not a mutual fund question")

    return Decision("factual", True, reason="retrievable")


# --- Post-check: the answer contract ---

_URL_RE = re.compile(r"https?://[^\s<>)\]]+|www\.[^\s<>)\]]+")
# Abbreviations whose trailing period is not a sentence end.
_ABBREV_RE = re.compile(
    r"\b(?:e\.g|i\.e|vs|no|nos|rs|fig|approx|etc|dr|mr|mrs|ms|st|inc|ltd|"
    r"a/c|ex|am|pm|w\.e\.f)\.",
    re.I,
)
# A link on its own, with no surrounding prose.
_BARE_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>)\]]+")
# Anything that looks like the model inventing its own freshness date.
_INVENTED_DATE_RE = re.compile(
    r"(?i)\b(?:last\s+(?:updated|refreshed)|as\s+of|updated\s+on|"
    r"current\s+as\s+of|as\s+on)\b[^\n]{0,60}",
)


def split_sentences(text: str) -> list[str]:
    """Split into sentences, treating URLs and abbreviations as atomic.

    Naive splitting on `[.!?]` over-counts badly here, because every answer
    must contain a URL and `https://www.sbimf.com` alone contributes two
    false sentence ends. The post-check enforces the 3-sentence cap with
    this, so a bad split means a bad contract.

    URLs are replaced with a placeholder and restored afterwards, because
    `_ABBREV_RE` would otherwise chew through `e.g.` and `vs.` inside a link.
    """
    urls: list[str] = []

    def park(match: re.Match[str]) -> str:
        urls.append(match.group(0))
        return f" URL{len(urls) - 1}END "

    masked = _URL_RE.sub(park, text or "")
    masked = _ABBREV_RE.sub(" ", masked)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", masked.strip()) if p.strip()]
    restored = []
    for part in parts:
        part = re.sub(r"URL(\d+)END", lambda m: urls[int(m.group(1))], part)
        restored.append(part.strip())
    # A citation standing alone is not a sentence. The contract is "at most 3
    # sentences, plus exactly one source link", so counting the link as a
    # fourth sentence would fail every otherwise-valid answer.
    return [p for p in restored if p and not _BARE_URL_RE.fullmatch(p)]


def count_sentences(text: str) -> int:
    return len(split_sentences(text))


def count_urls(text: str) -> int:
    return len(_URL_RE.findall(text or ""))


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of the post-check.

    `problems` and `repairs` are separate because a repair that succeeded is
    not a failure, and conflating them makes a working pipeline look broken.
    A missing citation, for example, is *expected* on every answer: the
    generator prompt tells the model not to emit a URL, so the app adds one.
    Reporting that as a problem each time would train a reader to ignore the
    field, which is exactly when a real violation would go unnoticed.
    """

    ok: bool
    answer: str
    problems: tuple[str, ...] = ()
    repairs: tuple[str, ...] = ()
    # The link actually shown to the user, which is not always the cited
    # chunk's own URL: a question whose answer *is* a link adopts the official
    # URL from the context instead, so the answer keeps exactly one link.
    citation: str = ""


# Advice phrasing in a *generated answer*. The pre-check catches "should I buy",
# but the model can volunteer a recommendation unprompted, which the PRD
# forbids just as firmly. Caught here because by this point the text is final.
_ADVICE_IN_ANSWER = re.compile(
    r"(?i)\b(?:you\s+should|you\s+must|you\s+(?:can|may)\s+consider|"
    r"i\s+(?:recommend|suggest|advise)|we\s+(?:recommend|suggest|advise)|"
    r"is\s+(?:a\s+)?good\s+(?:for|choice|option|investment)|"
    r"you\s+ought\s+to|consider\s+investing|should\s+invest|"
    r"would\s+be\s+(?:a\s+)?good\s+idea|best\s+option\s+for\s+you)\b"
)


def find_advice(text: str) -> str | None:
    """Return the advice phrase found in a generated answer, else None."""
    match = _ADVICE_IN_ANSWER.search(text or "")
    return match.group(0) if match else None


# The exact wording the generator prompt instructs the model to use when the
# context does not answer the question. Detecting it is what lets the pipeline
# report "the model declined" instead of counting a refusal as a delivered
# answer, which is the difference between 10/10 and 7/10 on the PRD list.
DECLINE_SENTENCE = "That information is not in the official sources I have."

_DECLINE = re.compile(
    r"(?i)\b(?:that\s+information\s+is\s+not\s+in|"
    r"the\s+(?:provided|retrieved|given)\s+(?:sources?|context|documents?)\s+"
    r"(?:do(?:es)?\s+not|don'?t)\s+contain|"
    r"not\s+(?:available|mentioned|stated|specified|provided|found)\s+in\s+"
    r"(?:the\s+)?(?:provided\s+|retrieved\s+|source\s+)?"
    r"(?:sources?|context|documents?)|"
    r"do(?:es)?\s+not\s+contain\s+(?:that|this|the\s+answer)|"
    r"i\s+(?:do\s+not|don'?t)\s+have\s+(?:that|this|the)\s+information|"
    r"cannot\s+answer\s+from\s+the\s+(?:provided\s+)?(?:sources?|context))"
)


def is_decline(text: str) -> bool:
    """Whether the generated answer is a refusal rather than a response.

    Kept separate from `verify_answer` because a decline is *contract
    compliant*: it has no invented facts, needs no repair, and passes every
    check. The only thing wrong with it is that it does not answer, which is
    a retrieval problem, not a guardrail one. Folding it into the same flag
    would make the two indistinguishable when reading the results.
    """
    body = (text or "").split(config.LAST_UPDATED_PREFIX)[0]
    return bool(_DECLINE.search(body))


def context_urls(context: str) -> frozenset:
    """Every URL that appears in the supplied source context.

    Used to decide whether a URL the model wrote is a real citation or an
    invention. A model is perfectly capable of producing a plausible
    sbimf.com path that is in no document, and an unverified link is worse
    than no link, so provenance is checked rather than assumed.
    """
    return frozenset(url.rstrip(".,;)") for url in _URL_RE.findall(context or ""))


def verify_answer(answer: str, source_url: str, fetched_at: str,
                  context: str = "") -> VerifyResult:
    """Check an LLM answer against the PRD contract, repairing what is safe.

    The rules, from PRD section on success criteria:
      - at most config.MAX_ANSWER_SENTENCES sentences
      - exactly one source link
      - the "Last updated" line is app-written, from `fetched_at`

    Repairs truncate to the sentence cap, drop extra links, and strip an
    invented date before the real one is appended. Anything that cannot be
    repaired into something non-empty and on-contract falls back to
    `MSG_NO_CONTEXT`, because showing a malformed answer is worse than
    showing none.
    """
    problems: list[str] = []
    repairs: list[str] = []
    body = (answer or "").strip()

    if _INVENTED_DATE_RE.search(body):
        repairs.append("dropped a self-invented date line")
        # Drop the whole sentence, not just the date phrase. Stripping only
        # "Last updated on 12 January 2024" from "Last updated on 12 January
        # 2024. The expense ratio is 2.25%." leaves the mangled orphan "is
        # 2.25%." behind, which reads as nonsense.
        kept = [s for s in split_sentences(body) if not _INVENTED_DATE_RE.search(s)]
        if kept:
            body = " ".join(kept).strip()
        else:
            body = _INVENTED_DATE_RE.sub("", body).strip()

    # The pre-check stops the user being *asked* for advice, but the model can
    # offer it unprompted. Remove those sentences before the length check so a
    # removed sentence does not also count as a truncation.
    advice = find_advice(body)
    if advice:
        repairs.append(f"removed advice ({advice!r})")
        kept = [s for s in split_sentences(body) if not find_advice(s)]
        if kept:
            body = " ".join(kept).strip()
        else:
            # Every sentence was advice, so there is no factual answer left to
            # repair. Showing the safe message is the only honest option.
            problems.append("answer was entirely advice, no factual content")
            return VerifyResult(False, MSG_NO_CONTEXT, tuple(problems),
                                tuple(repairs))

    sentences = count_sentences(body)
    if sentences > config.MAX_ANSWER_SENTENCES:
        repairs.append(
            f"truncated {sentences} sentences to the "
            f"{config.MAX_ANSWER_SENTENCES}-sentence cap"
        )
        kept = split_sentences(body)
        if len(kept) > config.MAX_ANSWER_SENTENCES:
            kept = kept[: config.MAX_ANSWER_SENTENCES]
        body = " ".join(part for part in kept if part).strip()
        # Truncation can cut mid-clause, which reads badly. Drop a trailing
        # fragment that has no terminal punctuation rather than shipping it.
        if body and body[-1] not in ".!?":
            trimmed = re.sub(r"[^.!?]*$", "", body).strip()
            if trimmed:
                body = trimmed
    # A model-written URL is adopted as the citation only when it appears
    # verbatim in the supplied context and sits on an allowed domain. This is
    # what makes "where do I download the KIM/SID?" answerable: the URL *is*
    # the answer, so the app adopts it instead of appending a second link and
    # breaking the one-citation contract. An unrecognised or off-policy URL is
    # an invention, so it is stripped and the chunk's own URL is used.
    urls = _URL_RE.findall(body)
    keep: list[str] = []
    adopted = ""
    if urls:
        known = context_urls(context)
        for url in urls:
            clean = url.rstrip(".,;)")
            if (
                not adopted
                and clean in known
                and config.is_allowed_url(clean)
            ):
                adopted = clean
                keep.append(clean)
            else:
                problems.append(
                    f"dropped link not present in the source context: {clean}"
                )
                repairs.append("removed a model-written link")
        body = _URL_RE.sub(" ", body)
        body = " ".join(body.split()).strip()
        if adopted:
            repairs.append("used the official link the answer itself points to")
            body = f"{body} {adopted}".strip()

    citation = adopted or source_url
    if count_urls(body) == 0:
        # Expected, not a defect: the prompt tells the model to omit the
        # citation link and lets the app add it from cited-chunk metadata.
        # Reporting this on every answer would drown out the real problems.
        repairs.append("added the citation link from chunk metadata")
        body = f"{body} {citation}".strip() if body else citation

    if not body.strip() or not count_sentences(body):
        problems.append("answer was empty after repair")
        return VerifyResult(False, MSG_NO_CONTEXT, tuple(problems), tuple(repairs))

    final = f"{body}\n\n{config.LAST_UPDATED_PREFIX} {fetched_at}".strip()
    return VerifyResult(
        True, final, tuple(problems), tuple(repairs), citation=citation
    )


def no_context_answer() -> str:
    """The 'I don't know' response for when retrieval finds nothing usable."""
    return MSG_NO_CONTEXT
