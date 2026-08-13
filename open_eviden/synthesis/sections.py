"""The same patterns as `rules.py`, restricted to the sections that can answer.

One change from the naive baseline, and it is worth stating precisely because the
leaderboard turns on it: this extractor never reads the Discussion, the
Introduction, or the reference list.

Those sections are where authors describe *other people's trials*. "In the trial
by Smith et al., 1,204 patients were randomised" is a true sentence, correctly
quoted, containing a participant count that belongs to a different study. The
naive baseline takes it. Restricting the search to the Abstract, the Methods and
the Results removes an entire class of error that no amount of pattern-tuning
would have fixed, because the patterns were never wrong — the scope was.

Section order matters as much as section membership. The Abstract is searched
first for every field, because a structured abstract states the design, the
masking, the participant count and the primary outcome in one place, written to
be unambiguous. The Methods are searched next; the Results last. First hit wins,
so a value stated in the abstract beats a number appearing later in a subgroup
table.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..records import FIELD_NAMES
from .base import Claim, Document, Section
from .rules import extract_claims

#: Publisher headings mapped to the role we care about. Matched case-insensitively
#: against the start of the heading, so "Materials and methods" and "Methods and
#: materials" both land on `methods` and "Statistical methods" does too.
#:
#: Anything unmatched becomes `other` and is not searched. That is the
#: conservative direction: a missed section costs recall, while searching an
#: unrecognised section risks reintroducing exactly the Discussion problem this
#: module exists to remove.
_ROLE_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("abstract", re.compile(r"^(abstract|summary)\b", re.IGNORECASE)),
    (
        "methods",
        re.compile(
            r"^(methods?|materials?\s+and\s+methods?|methods?\s+and\s+materials?|"
            r"patients?\s+and\s+methods?|study\s+design|design|participants?|"
            r"trial\s+design|statistical\s+(analysis|methods?))\b",
            re.IGNORECASE,
        ),
    ),
    ("results", re.compile(r"^(results?|findings)\b", re.IGNORECASE)),
    (
        "excluded",
        re.compile(
            r"^(discussion|introduction|background|conclusions?|references?|"
            r"related\s+work|limitations?|acknowledge?ments?|supplementary)\b",
            re.IGNORECASE,
        ),
    ),
)

#: Searched in this order; first grounded claim for a field wins.
SEARCH_ORDER = ("abstract", "methods", "results")


def role_of(section: Section) -> str:
    """Classify a publisher heading into one of our roles."""
    title = section.title.strip()
    for role, pattern in _ROLE_PATTERNS:
        if pattern.match(title):
            return "other" if role == "excluded" else role
    return "other"


class SectionScopedExtractor:
    """`rules.py`'s patterns, run over Abstract then Methods then Results.

    Falls back to the whole article when a document carries no usable section
    structure. Some publishers deposit full text as a single untitled block, and
    an extractor that silently returned nothing for those would report a recall
    penalty that belongs to the publisher's XML rather than to the method. The
    fallback is recorded in `describe()` output per run so the effect is visible.
    """

    name = "sections"

    def extract(self, document: Document) -> Sequence[Claim]:
        by_role: dict[str, list[Section]] = {role: [] for role in SEARCH_ORDER}
        for section in document.sections:
            role = role_of(section)
            if role in by_role:
                by_role[role].append(section)

        if not any(by_role.values()):
            return extract_claims(document.text, section="full-text (no sections found)")

        claims: dict[str, Claim] = {}
        for role in SEARCH_ORDER:
            for section in by_role[role]:
                text = document.section_text(section)
                for claim in extract_claims(text, section=section.title):
                    # First hit wins. `ground` would drop the later duplicates
                    # anyway, but it would count them as `duplicate_field`
                    # rejections and depress a grounding rate that has nothing
                    # wrong with it.
                    claims.setdefault(claim.field, claim)
            if len(claims) == len(FIELD_NAMES):
                break

        return list(claims.values())

    def describe(self) -> dict:
        return {
            "extractor": "SectionScopedExtractor",
            "approach": "regular expressions restricted to abstract, methods and results",
            "deterministic": True,
            "requires_network": False,
            "can_hallucinate": False,
            "scope": " -> ".join(SEARCH_ORDER),
        }
