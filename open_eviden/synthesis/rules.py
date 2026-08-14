"""A deterministic, offline, rule-based extractor — and the patterns everything reuses.

This is the baseline that has to be beaten, and it is not a strawman. Clinical
trial reports are written to a house style enforced by CONSORT, so a surprising
amount of what a reviewer extracts by hand is stated in a small number of stock
phrasings. "A total of 240 patients were randomly assigned", "the primary outcome
was", "registered at ClinicalTrials.gov (NCT01234567)".

Two properties make it a good baseline rather than a placeholder:

**It cannot hallucinate.** Every value it reports is read out of a span it
matched, and the quotation it returns is that span's sentence, sliced out of the
document text. Its grounding rate is 1.0 by construction, and it is not being
virtuous — it has no mechanism for inventing anything. Any method that scores
better on recall while grounding worse is trading away something this one gives
away free.

**Its failures are the interesting ones.** It matches text anywhere in the
article, including the Discussion, where authors routinely write "in the trial by
Smith et al., 1,204 patients were randomised". That number is real, quoted
correctly, and about somebody else's trial. `sections.py` is the same patterns
with that hole closed, and the gap between the two is the clearest thing on the
leaderboard.

The patterns live here rather than in a data file because half of them need a
comment explaining what they must not match.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..records import normalize_whitespace
from .base import Claim, Document

#: Hyphens, as publishers actually emit them.
#:
#: JATS full text is full of typographic hyphens — U+2010 HYPHEN and U+2011
#: NON-BREAKING HYPHEN — where a reader sees an ordinary dash. "double‐blind"
#: with U+2010 does not match `double[- ]blind`, so a pattern written with an
#: ASCII hyphen silently misses every article from those publishers. This was
#: worth four of six missed blinding annotations before it was fixed, and it is
#: invisible in a terminal: the two characters render identically.
HYPHEN = r"[-\u2010\u2011\u2012\u2013 ]"

#: Minimum characters of context returned as the quotation.
#:
#: The claim contract rejects quotations under 20 characters, and it is right to:
#: a quotation short enough to appear in any article evidences nothing. This
#: expands a match to its enclosing sentence, which is both longer than the floor
#: and the unit a human reviewer would actually read.
MIN_QUOTE_CHARS = 20

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def sentence_around(text: str, start: int, end: int) -> str:
    """The sentence containing `text[start:end]`, verbatim.

    Sliced straight out of `text`, never rebuilt, so the result is guaranteed to
    be a substring of the document and therefore guaranteed to ground. If the
    sentence comes out shorter than the floor — an abbreviated heading, a table
    cell — the window widens by characters rather than returning something that
    will be rejected downstream.

    Sentence splitting on punctuation is wrong in clinical text ("p < 0.05.",
    "vs.", "Dr."). It is used anyway because the consequence of being wrong is a
    quotation with a neighbouring clause attached, which a reader can live with,
    and the alternative is a sentence-splitting dependency for a benchmark whose
    selling point is that it runs offline from a clone.
    """
    left = 0
    for match in _SENTENCE_END.finditer(text, 0, start):
        left = match.end()
    right_match = _SENTENCE_END.search(text, end)
    right = right_match.start() + 1 if right_match else len(text)

    quote = text[left:right].strip()
    if len(quote) >= MIN_QUOTE_CHARS:
        return quote

    pad = MIN_QUOTE_CHARS - len(quote)
    left = max(0, left - pad)
    right = min(len(text), right + pad)
    return text[left:right].strip()


#: Words for the people in a trial.
_PEOPLE = (
    r"patients|participants|subjects|individuals|adults|children|women|men|"
    r"infants|survivors|volunteers|staff|cyclists|athletes"
)

#: Words that mean an allocation happened.
#:
#: Enrolment and screening are deliberately absent: a paper reporting "312 were
#: screened, 240 were randomised" states both, and 312 is the wrong answer.
#: `recruit` and `enrol` appear only in the *sentence filter* below, never as
#: the thing a count attaches to, because a sentence can legitimately give both
#: numbers and only the allocated one is wanted.
_ALLOCATED = r"randomi[sz]|random\s+alloc|allocat|assign"

#: Sentences that could carry the randomised count. The count is then pulled
#: from inside the matched sentence.
#:
#: Two stages rather than one big pattern, because trial reports state this
#: sentence in genuinely unbounded ways — "Twelve trained cyclists took part in
#: a ... trial", "a total of 214 patients underwent either X or Y after
#: randomization", "randomised 36 ICU survivors (1:1:1)", "Staff ( n = 28) were
#: randomised to SEI or SI". A single regex covering those either misses most of
#: them or matches any number in any sentence containing the word "random".
#: Splitting it lets the sentence filter be liberal and the number extraction be
#: strict.
_RANDOMISED_SENTENCE = re.compile(_ALLOCATED, re.IGNORECASE)

#: How a count is written, in priority order within a qualifying sentence.
_COUNT_IN_SENTENCE = (
    # "a total of 214 patients", "randomised 36 ICU survivors"
    re.compile(
        rf"\b(?:a\s+total\s+of\s+)?(\d[\d,]{{1,6}})\s+(?:\w+[\s-]){{0,3}}?(?:{_PEOPLE})\b",
        re.IGNORECASE,
    ),
    # "( n = 28 )" — bounded to a plausible trial size so a p-value or a year
    # cannot be read as a participant count.
    re.compile(r"\(?\s*[nN]\s*=\s*(\d[\d,]{1,6})\s*\)?"),
    # "Twelve trained cyclists", "Thirty sub-acute stroke patients". Spelled-out
    # counts open a sentence often enough in trial reports that skipping them
    # loses a real share of the field.
    re.compile(
        r"\b([A-Z][a-z]+(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?)\s+"
        rf"(?:\w+[\s-]){{0,3}}?(?:{_PEOPLE})\b"
    ),
)

_WORD_NUMBERS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}

#: Below this, a number next to a participant noun is almost always something
#: else — a number of sites, arms, or sessions. Trials smaller than this exist,
#: and this baseline will miss them; that is the trade, and it is stated here
#: rather than discovered from the scores.
MIN_PLAUSIBLE_N = 8

#: Design terms, most specific first. Order matters: a cluster-randomised
#: crossover trial is a cluster trial, and "parallel" appears in the boilerplate
#: of almost every report, so it must not win over a more specific term.
DESIGN_PATTERNS = (
    ("cluster-randomised", re.compile(rf"\bcluster{HYPHEN}randomi[sz]ed\b", re.IGNORECASE)),
    ("factorial", re.compile(r"\b(?:2\s*[x×]\s*2\s+)?factorial\b", re.IGNORECASE)),
    ("crossover", re.compile(rf"\bcross{HYPHEN}?over\b", re.IGNORECASE)),
    ("single-arm", re.compile(rf"\bsingle{HYPHEN}arm\b", re.IGNORECASE)),
    (
        "parallel-group",
        re.compile(rf"\bparallel{HYPHEN}(?:group|arm|design)|\btwo{HYPHEN}arm\b", re.IGNORECASE),
    ),
    # Fallback. Most reports never write the words "parallel group" — they say
    # "randomised 1:1", "two-arm", "the control group", and leave the structure
    # implied. A reviewer reads that as parallel-group and records it as such,
    # so a baseline that refuses to infer it is not modelling the task, it is
    # modelling a stricter task nobody performs.
    #
    # It fires last, so any explicitly named design above still wins — a
    # crossover trial also has a control group, and must not be caught here.
    (
        "parallel-group",
        re.compile(
            rf"\b(?:two|three|four){HYPHEN}arms?\b|\b\d\s*:\s*\d(?:\s*:\s*\d)?\b|"
            r"\b(?:the\s+)?(?:control|placebo|comparison|usual[- ]care|waitlist)\s+group\b|"
            r"\b(?:intervention|experimental|treatment)\s+group\b",
            re.IGNORECASE,
        ),
    ),
)

#: Masking. Again most specific first — "double-blind" contains no substring of
#: "single-blind", but "blinded" alone appears in both and in neither.
#:
#: There is no pattern for absence. A trial that never describes its masking
#: produces no claim here, which is the intended behaviour: see the note on
#: absence in records.FIELDS and in the README.
BLINDING_PATTERNS = (
    ("triple-blind", re.compile(rf"\btriple{HYPHEN}blind(?:ed)?\b", re.IGNORECASE)),
    ("double-blind", re.compile(rf"\bdouble{HYPHEN}blind(?:ed)?\b", re.IGNORECASE)),
    ("single-blind", re.compile(rf"\bsingle{HYPHEN}blind(?:ed)?\b", re.IGNORECASE)),
    (
        "open-label",
        re.compile(
            rf"\bopen{HYPHEN}label\b|\bun{HYPHEN}?blinded\b|"
            rf"\bnon{HYPHEN}?blinded\b|\bno\s+blinding\b",
            re.IGNORECASE,
        ),
    ),
    # Assessor-only masking is single-blinding by CONSORT's description of who
    # was masked, and reporting it as such is more useful than reporting nothing.
    # It is listed last so an explicit "double-blind" always wins.
    (
        "single-blind",
        re.compile(
            rf"\b(?:outcome\s+)?assessors?{HYPHEN}?(?:were\s+)?(?:blinded|masked)\b|"
            rf"\bassessor{HYPHEN}blinded\b",
            re.IGNORECASE,
        ),
    ),
)

#: Follow-up duration, with the unit captured so it can be converted.
FOLLOW_UP_PATTERNS = (
    re.compile(
        r"\bfollow(?:ed)?[- ]?up\s+(?:period\s+|duration\s+)?(?:of|was|lasted|for)?\s*"
        r"(\d+(?:\.\d+)?)\s*(day|week|month|year)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d+(?:\.\d+)?)\s*(day|week|month|year)s?\s+(?:of\s+)?follow[- ]?up\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bprimary\s+(?:outcome|endpoint)[^.]{0,80}?\bat\s+(\d+(?:\.\d+)?)\s*"
        r"(day|week|month|year)s?\b",
        re.IGNORECASE,
    ),
    # "a 14-week intervention", "an 8-week blended program", "12-week trial".
    # The duration of the intervention is not always the duration of follow-up,
    # and where a paper reports both this will take the wrong one. It is
    # included because reports state the period this way far more often than
    # they write "the follow-up period was", and a baseline that only matches
    # the formal phrasing scores near zero on a field papers do report.
    re.compile(
        r"\b(\d+(?:\.\d+)?)[-‐\s](day|week|month|year)s?\s+"
        r"(?:blended\s+|remotely\s+delivered\s+|home[-\s]based\s+)?"
        r"(?:follow[- ]?up|intervention|programme|program|trial|study|treatment|course)\b",
        re.IGNORECASE,
    ),
    # "over 12 weeks", "for 2 weeks", "at 12 months"
    re.compile(
        r"\b(?:over|for|during|across)\s+(\d+(?:\.\d+)?)\s*(day|week|month|year)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bat\s+(\d+(?:\.\d+)?)\s*(day|week|month|year)s?\b(?![^.]{0,20}\bbaseline\b)",
        re.IGNORECASE,
    ),
    # "at one year postoperatively" — spelled-out durations, same reasoning as
    # the spelled-out participant counts above.
    re.compile(
        r"\bat\s+(one|two|three|six|twelve)\s+(day|week|month|year)s?\b",
        re.IGNORECASE,
    ),
)

#: Weeks per unit. A month is 365.25/12/7 weeks rather than 4, because a "12
#: month" follow-up scored against a reference of 52 weeks would otherwise fail
#: by four weeks. The scorer's tolerance is not wide enough to absorb that, and
#: widening the tolerance to hide a conversion error would be the wrong fix.
WEEKS_PER = {"day": 1 / 7, "week": 1.0, "month": 365.25 / 12 / 7, "year": 365.25 / 7}

#: Durations papers spell out — "at one year", "over six months".
_WORD_NUMBERS_SMALL = {"one": 1, "two": 2, "three": 3, "six": 6, "twelve": 12}

#: Trial registry identifiers. Each registry publishes its own id format; these
#: are the eight that appear on the WHO ICTRP primary registry list and account
#: for essentially all registered trials in the OA literature.
REGISTRATION_PATTERNS = (
    re.compile(r"\bNCT\s?\d{8}\b"),
    re.compile(r"\bISRCTN\s?\d{8}\b"),
    re.compile(r"\bChiCTR[-\w]*\d{6,}\b"),
    re.compile(r"\bACTRN\s?\d{14}\b"),
    re.compile(r"\bDRKS\s?\d{8}\b"),
    re.compile(r"\bUMIN\s?\d{9}\b"),
    re.compile(r"\bIRCT\s?\d{11,}N\d+\b"),
    re.compile(r"\b\d{4}-\d{6}-\d{2}\b"),  # EudraCT
)

#: Words that cannot be part of an outcome name, read backwards from a label.
#:
#: Only used by the apposition pattern below, which reads right to left and needs
#: somewhere to stop. Stopping at a preposition, article, conjunction or copula
#: leaves "time trial performance" rather than "the effect of the intervention on
#: time trial performance", which is the whole clause the outcome sits in.
_NOT_IN_OUTCOME = (
    r"of|on|in|at|for|to|with|by|from|and|or|the|a|an|our|its|their|this|that|"
    r"these|those|was|were|is|are|be|been|as|than|between|during"
)

#: One word of an outcome name.
_OUTCOME_WORD = rf"(?!(?:{_NOT_IN_OUTCOME})\b)[\w'’‐-]+"

#: The primary outcome, as the authors define it.
#:
#: Capped at 200 characters and stopped at sentence-ending punctuation, because
#: an unbounded capture runs into the next three sentences and the token-overlap
#: score then rewards verbosity — a longer capture contains the reference tokens
#: by accident.
#:
#: CONSORT 2010 item 6a asks authors for "completely defined pre-specified
#: primary outcome measures" (https://doi.org/10.1136/bmj.c332) and says nothing
#: about the construction used to define them, so a report is free to name the
#: outcome after a copula, in apposition inside brackets, or after a qualifier
#: that separates the noun from the verb. Per @nextbaires (#1), reading only the
#: first of those returned nothing at all on three of the four articles this
#: field missed — and nothing reads as "the trial declared no primary outcome"
#: rather than "the tool could not parse the sentence".
#:
#: Order is priority: the first pattern that matches anywhere in the text wins,
#: so the explicit definitions stay ahead of the appositions, which are in turn
#: ahead of the loosened copula. That ordering is load-bearing on at least one
#: article, which names its outcome in brackets in the sample-size calculation
#: and again, differently worded, in a later "the primary outcome of this trial
#: was …" sentence.
PRIMARY_OUTCOME_PATTERNS = (
    re.compile(
        r"\b(?:the\s+)?(?:co-?)?primary\s+(?:outcome|endpoint|efficacy\s+endpoint)"
        r"(?:\s+measure)?s?\s+(?:was|were|is|are)\s+(?:the\s+)?([^.;]{5,200})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the\s+)?(?:co-?)?primary\s+(?:outcome|endpoint)(?:\s+measure)?s?[:\-]\s*"
        r"([^.;]{5,200})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwe\s+(?:defined|specified)\s+(?:the\s+)?primary\s+(?:outcome|endpoint)\s+as\s+"
        r"([^.;]{5,200})",
        re.IGNORECASE,
    ),
    # "…an effect size of 0.86 for the primary outcome (premenstrual syndrome
    # severity)…" — the label first, the outcome in apposition after it. Bounded
    # to what is inside the brackets, so it cannot run past the closing one.
    re.compile(
        r"\b(?:the\s+)?(?:co-?)?primary\s+(?:outcome|endpoint)(?:\s+measure)?s?\s*"
        r"\(([^)]{5,120})\)",
        re.IGNORECASE,
    ),
    # "…the effect of the intervention on time trial performance (our primary
    # outcome)…" — the same apposition with the label second. Per @nextbaires
    # (#1) this is the shape a rule that only ever looks rightward cannot reach.
    # The capture is the run of words immediately before the bracket that
    # contains no preposition, article or copula; five words is long enough for
    # "work completed during the first time trial" and short enough that it
    # cannot swallow the clause the outcome is embedded in.
    re.compile(
        # The leading \b is load-bearing: without it the capture starts inside
        # the barrier word instead of skipping it, and "…intervention on time
        # trial performance" yields "n time trial performance".
        rf"\b({_OUTCOME_WORD}(?:\s+{_OUTCOME_WORD}){{0,4}})\s*"
        r"\((?:our|the|its|a)?\s*(?:pre-?registered\s+|pre-?specified\s+)?(?:co-?)?"
        r"primary\s+(?:outcome|endpoint)(?:\s+measure)?s?\)",
        re.IGNORECASE,
    ),
    # "The primary outcomes *of this feasibility trial* were intervention
    # practicality and acceptability." Per @nextbaires (#1) the intervening
    # phrase is ordinary English, and requiring noun and copula to be adjacent
    # rejects it.
    #
    # The qualifier is the tightest thing that admits the reported shape: a
    # prepositional phrase, no punctuation inside it, at most 40 characters.
    # That is deliberate, and it is where the proposal's own falsification
    # condition bites — a qualifier allowed to contain a comma also matches
    # "no difference in the primary outcome, which was consistent with earlier
    # work", a discussion sentence that names nothing, and one allowed to start
    # anywhere matches "the primary outcome rate in the control group is 20%",
    # a sample-size sentence whose capture would be a percentage.
    re.compile(
        r"\b(?:the\s+)?(?:co-?)?primary\s+(?:outcome|endpoint)(?:\s+measure)?s?\s+"
        r"(?:(?:of|in|for|during|at|across)\s+[^.;:,()]{1,40}?\s+)?"
        r"(?:was|were|is|are)\s+(?:the\s+)?([^.;]{5,200})",
        re.IGNORECASE,
    ),
)

#: A numbered citation marker inside a captured outcome — "[ 30 ]", "[16, 17]".
_CITATION_MARKER = re.compile(r"\s*\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]")


def clean_outcome(value: str) -> str:
    """Trim a captured outcome down to the outcome.

    The capture stops at sentence punctuation, which leaves the author's
    referencing apparatus inside it: "changes in the PI (according to Silness
    and Loe [30]) and GI (accord…" on a trial whose outcome is the plaque index.
    Per @nextbaires (#1) the capture is correct and the padding is not part of
    what a reviewer records.

    Two things are removed and nothing else: numbered citation markers, and a
    trailing fragment left dangling by an unclosed bracket, which is what the
    200-character cap and the sentence-punctuation stop produce when they land
    inside a parenthesis. Nothing is reordered and no word is rewritten.

    This trims the *value*. The quotation is unaffected — it stays the sentence
    sliced out of the document — so grounding is exactly as it was, which
    `tests/test_extractors.py` asserts on the corpus.
    """
    cleaned = _CITATION_MARKER.sub("", value)
    for opener, closer in (("(", ")"), ("[", "]")):
        while cleaned.count(opener) > cleaned.count(closer):
            cleaned = cleaned[: cleaned.rindex(opener)]
    return cleaned.strip().strip(",;:-–—").strip()


def _first_match(patterns, text: str):
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match
    return None


def _parse_count(raw: str) -> int | None:
    """Parse a count written as digits or as a word."""
    cleaned = raw.replace(",", "").strip()
    if cleaned.isdigit():
        return int(cleaned)

    parts = re.split(r"[- ]", cleaned.lower())
    if not parts or parts[0] not in _WORD_NUMBERS:
        return None
    total = _WORD_NUMBERS[parts[0]]
    # "Twenty-four" — tens word followed by a units word.
    if len(parts) > 1 and parts[1] in _WORD_NUMBERS and _WORD_NUMBERS[parts[1]] < 10:
        total += _WORD_NUMBERS[parts[1]]
    return total


def find_n_randomised(text: str) -> tuple[int, tuple[int, int]] | None:
    """The randomised participant count, and where it was found.

    Sentence first, number second. Every sentence mentioning allocation is
    considered in document order, and within it the count forms are tried in
    priority order. The first plausible count wins.

    Returns `(count, (start, end))` so the caller can quote the span; returning
    the number alone would leave the caller re-searching for it, and the second
    search can land somewhere else.
    """
    position = 0
    for sentence in _SENTENCE_END.split(text):
        start = text.find(sentence, position)
        if start < 0:  # pragma: no cover - defensive; split pieces are substrings
            start = position
        position = start + len(sentence)

        if not _RANDOMISED_SENTENCE.search(sentence):
            continue

        for pattern in _COUNT_IN_SENTENCE:
            for match in pattern.finditer(sentence):
                count = _parse_count(match.group(1))
                if count is not None and count >= MIN_PLAUSIBLE_N:
                    return count, (start + match.start(), start + match.end())

    return None


def extract_claims(text: str, section: str = "full-text") -> list[Claim]:
    """Apply every pattern to `text` and return grounded-by-construction claims.

    Shared by both deterministic baselines. `sections.py` calls it on one section
    at a time; this module calls it on the whole article. Everything that differs
    between the two baselines is *which text this sees*, which is the comparison
    the leaderboard is there to make.
    """
    claims: list[Claim] = []

    def add(field: str, value: str, match: re.Match) -> None:
        claims.append(
            Claim(
                field=field,
                value=value,
                quote=sentence_around(text, match.start(), match.end()),
                section=section,
            )
        )

    found_n = find_n_randomised(text)
    if found_n:
        count, span = found_n
        claims.append(
            Claim(
                field="n_randomised",
                value=str(count),
                quote=sentence_around(text, span[0], span[1]),
                section=section,
            )
        )

    for value, pattern in DESIGN_PATTERNS:
        found = pattern.search(text)
        if found:
            add("design", value, found)
            break

    for value, pattern in BLINDING_PATTERNS:
        found = pattern.search(text)
        if found:
            add("blinding", value, found)
            break

    match = _first_match(FOLLOW_UP_PATTERNS, text)
    if match:
        raw = match.group(1)
        amount = float(raw) if raw[0].isdigit() else float(_WORD_NUMBERS_SMALL[raw.lower()])
        weeks = amount * WEEKS_PER[match.group(2).lower()]
        # Two decimals: a month-to-week conversion is not precise enough to
        # justify more, and a full float repr in the output implies a precision
        # the source sentence does not have.
        add("follow_up_weeks", f"{weeks:.2f}", match)

    match = _first_match(REGISTRATION_PATTERNS, text)
    if match:
        add("registration_id", match.group(0).strip(), match)

    match = _first_match(PRIMARY_OUTCOME_PATTERNS, text)
    if match:
        outcome = clean_outcome(match.group(1))
        # A capture that cleans down to nothing — all citation, no outcome — is
        # possible in principle. Returning it would be an empty-value rejection
        # counted against the grounding rate, where saying nothing is the
        # abstention the protocol already knows how to score.
        if outcome:
            add("primary_outcome", outcome, match)

    return claims


class RegexExtractor:
    """Pattern matching over the whole article, start to finish.

    The naive baseline, and the one whose errors are worth reading: it will
    happily take a participant count out of a sentence describing a different
    trial in the Discussion.
    """

    name = "regex"

    def extract(self, document: Document) -> Sequence[Claim]:
        return extract_claims(normalize_whitespace(document.text))

    def describe(self) -> dict:
        return {
            "extractor": "RegexExtractor",
            "approach": "regular expressions over the full text",
            "deterministic": True,
            "requires_network": False,
            "can_hallucinate": False,
            "scope": "full-text",
        }
