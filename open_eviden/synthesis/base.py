"""The extractor interface, and the rule that makes the benchmark mean something.

**An extracted value counts only if the extractor can point at the text it came
from.** Not "should"; counts. `ground` takes what an extractor returned and the
document it was given, searches the document's text for each quotation, and
discards every claim whose quotation is not there. Only survivors reach scoring.

This is the load-bearing design decision in the repository, and it is here for
the same reason `seismic-bench` builds its training windows so that no model can
widen them: the interesting failure mode has to be made impossible by the
structure, not merely discouraged in a guideline.

The failure mode is specific. A language model has read a great deal of clinical
literature. Asked how many participants a well-known trial randomised, it can
often say — from memory, without reading the document in front of it. On a naive
benchmark that scores values against a reference, recalling the answer and
extracting the answer earn identical marks, and the benchmark stops measuring
extraction. Requiring a verbatim span collapses that: a remembered value has no
quotation attached to it, a fabricated quotation does not appear in the text, and
either way the claim is dropped before it can score.

What this does NOT guarantee, and the distinction matters enough that it is also
in DISCLAIMER.md: a grounded claim points at real text. It does not point at text
that *supports* the value. An extractor can quote a genuine sentence from the
Methods and attach a number that sentence does not contain, and grounding will
pass it. That is what the reference set is for, and it is why the extracted quote
is carried all the way through to the output — so a human can read it.

The `Document` handed to an extractor carries the article and nothing else. There
is no argument that reveals the reference annotations, no handle on the rest of
the corpus, and no split label. `tests/test_grounding.py` asserts this, including
against an extractor written specifically to cheat.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Protocol, runtime_checkable

import pandas as pd

from ..records import FIELDS, normalize_whitespace


@dataclass(frozen=True)
class Section:
    """One section of an article, as character offsets into `Document.text`.

    Offsets rather than a copy of the text, so that a span taken from a section
    is still literally a span of the document and grounds without any special
    handling. `title` is the publisher's own heading, unmodified — mapping
    "Materials and methods" onto a canonical role is a modelling choice and
    belongs in the extractor that makes it, not in the data.
    """

    title: str
    start: int
    end: int


@dataclass(frozen=True)
class Document:
    """One article, as an extractor sees it.

    Frozen, and deliberately narrow. Everything an extractor is allowed to use is
    an attribute here; there is nothing else to reach for. In particular there is
    no `corpus`, no `gold`, and no `split` — an extractor cannot discover whether
    it is being run on the development set or the test set, so it cannot behave
    differently on them.
    """

    document_id: str
    doi: str
    title: str
    journal: str
    year: int | None
    text: str
    sections: tuple[Section, ...] = ()

    @classmethod
    def from_row(cls, row) -> Document:
        year = row["year"]
        raw = row.get("sections") if hasattr(row, "get") else row["sections"]
        sections = tuple(
            Section(title=str(s["title"]), start=int(s["start"]), end=int(s["end"]))
            for s in json.loads(raw)
        ) if isinstance(raw, str) and raw else ()
        return cls(
            document_id=str(row["document_id"]),
            doi=str(row["doi"]),
            title=str(row["title"]),
            journal=str(row["journal"]),
            year=None if pd.isna(year) else int(year),
            text=str(row["text"]),
            sections=sections,
        )

    def section_text(self, section: Section) -> str:
        """The span of `text` this section covers.

        A slice, never a rebuild. Anything an extractor quotes out of the result
        is therefore still a substring of `text` and grounds unchanged.
        """
        return self.text[section.start : section.end]


@dataclass(frozen=True)
class Claim:
    """One extracted statement about one document.

    `value` is always a string. Typed comparison — an integer count, a number of
    weeks, a term from a controlled vocabulary — happens at scoring time against
    the field's declared kind, so an extractor never has to guess what Python
    type the harness wants.

    `quote` is the span the value came from, copied verbatim out of the document.
    `section` is where the extractor believes it found it and is advisory: it is
    reported, never scored, because section labelling is inconsistent across
    journals and scoring it would measure the journal's typesetting.
    """

    field: str
    value: str
    quote: str
    section: str = ""


@dataclass
class Grounding:
    """What survived, and what did not — with the reason.

    Rejections are counted and reported rather than dropped quietly. An extractor
    that produces forty claims of which thirty-five are ungrounded is behaving
    very differently from one that produces five grounded claims, and a benchmark
    that showed both as "5 claims" would be concealing the most diagnostic thing
    it knows.
    """

    claims: list[Claim] = dataclass_field(default_factory=list)
    rejected: list[tuple[Claim, str]] = dataclass_field(default_factory=list)

    @property
    def n_returned(self) -> int:
        return len(self.claims) + len(self.rejected)

    @property
    def grounding_rate(self) -> float:
        """Fraction of returned claims that pointed at real text.

        NaN, not 1.0, when nothing was returned. An extractor that returns
        nothing has not achieved perfect grounding; it has abstained, and the
        recall term is where that shows up.
        """
        return len(self.claims) / self.n_returned if self.n_returned else float("nan")

    def reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, reason in self.rejected:
            counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items()))


#: Why a claim was discarded before scoring.
REJECTION_REASONS = (
    "unknown_field",  # not in records.FIELDS
    "empty_value",
    "quote_missing",  # extractor returned no quotation at all
    "quote_not_in_document",  # the quotation is not in the text — fabricated or paraphrased
    "duplicate_field",  # a second claim for a field already claimed
)


def ground(claims: Iterable[Claim], document: Document) -> Grounding:
    """Keep only claims whose quotation is verbatim in `document.text`.

    Matching is an exact substring test, case-sensitive, after collapsing runs of
    whitespace on both sides. Two deliberate choices there:

    *Whitespace is normalized* because a quotation that is correct in every
    respect except that the publisher broke a line inside it should not be
    treated as fabricated. The same normalization ran over the text at ingest, so
    this is idempotent on the document side.

    *Case is not normalized* because "verbatim" is the entire claim being made. A
    quotation that has been re-cased has been edited, and an extractor that edits
    quotations is one that will eventually edit a number.

    At most one claim per field survives, and it is the first one offered. An
    extractor that returns three candidate values for `n_randomised` and hopes one
    matches is guessing, and letting it keep the lucky guess would price guessing
    at zero. The dropped ones are reported as `duplicate_field`, so the behaviour
    is visible rather than punished silently.
    """
    haystack = normalize_whitespace(document.text)
    result = Grounding()
    seen: set[str] = set()

    for claim in claims:
        reason = _rejection_reason(claim, haystack, seen)
        if reason is None:
            seen.add(claim.field)
            result.claims.append(claim)
        else:
            result.rejected.append((claim, reason))

    return result


def _rejection_reason(claim: Claim, haystack: str, seen: set[str]) -> str | None:
    if claim.field not in FIELDS:
        return "unknown_field"
    if claim.field in seen:
        return "duplicate_field"
    if not str(claim.value).strip():
        return "empty_value"
    if not str(claim.quote).strip():
        return "quote_missing"
    if normalize_whitespace(str(claim.quote)) not in haystack:
        return "quote_not_in_document"
    return None


@runtime_checkable
class Extractor(Protocol):
    """What a submission implements. Three members, no base class to inherit."""

    name: str

    def extract(self, document: Document) -> Sequence[Claim]:
        """Return claims about `document`. Fewer, grounded claims beat more, guessed ones."""
        ...

    def describe(self) -> dict:
        """Parameters and provenance, for the leaderboard record."""
        ...


def validate_claims(claims: Sequence[Claim], extractor_name: str) -> list[Claim]:
    """Check the *shape* of what an extractor returned, before grounding.

    Grounding decides which claims count. This decides whether the extractor is
    even speaking the protocol, and it raises rather than filtering, because a
    return type this wrong is a bug in the submission and should stop the run
    with the extractor's name in the message — not quietly score zero and leave
    someone reading a leaderboard wondering why their method looks terrible.
    """
    if claims is None:
        raise TypeError(f"{extractor_name}.extract returned None; return an empty list instead")
    bad = [c for c in claims if not isinstance(c, Claim)]
    if bad:
        raise TypeError(
            f"{extractor_name}.extract returned {type(bad[0]).__name__} where Claim was expected"
        )
    return list(claims)
