"""Offline provenance checks: does every claim still point where it says it does.

Runs with no network, on every push, over the committed sample. These are the
checks that can be made absolute — a quotation either appears in its document or
it does not — and they are kept separate from the DOI resolution check in
`doi.py` precisely because that one depends on the internet being up and on
publishers keeping their records alive, and a check that fails for reasons
outside the repository trains people to ignore failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..contracts import DOI_PATTERN, REDISTRIBUTABLE_LICENSES
from ..records import FIELDS, normalize_whitespace


@dataclass
class Problem:
    check: str
    subject: str  # document_id, or "document_id/field" for a claim
    detail: str

    def __str__(self) -> str:
        return f"{self.check}: {self.subject} — {self.detail}"


@dataclass
class Report:
    problems: list[Problem] = field(default_factory=list)
    n_documents: int = 0
    n_claims: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    def by_check(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for problem in self.problems:
            counts[problem.check] = counts.get(problem.check, 0) + 1
        return dict(sorted(counts.items()))


def check(documents: pd.DataFrame, gold: pd.DataFrame) -> Report:
    """Verify the committed corpus and reference annotations against each other.

    Every failure is collected rather than raised on the first one. A run that
    stops at the first bad row tells you about one problem; this tells you
    whether you have one problem or forty, which is the difference between a
    typo and a broken ingest.
    """
    report = Report(n_documents=len(documents), n_claims=len(gold))
    text_by_id = dict(zip(documents["document_id"], documents["text"], strict=True))

    for _, row in documents.iterrows():
        document_id = str(row["document_id"])

        doi = str(row["doi"] or "")
        if not doi:
            report.problems.append(Problem("doi_missing", document_id, "no DOI"))
        elif not DOI_PATTERN.match(doi):
            report.problems.append(Problem("doi_malformed", document_id, doi))

        license_name = str(row["license"] or "").lower().strip()
        if license_name not in REDISTRIBUTABLE_LICENSES:
            report.problems.append(
                Problem(
                    "license_not_redistributable",
                    document_id,
                    f"{license_name!r} is not in {sorted(REDISTRIBUTABLE_LICENSES)}; "
                    "this article cannot be committed to this repository",
                )
            )

        if not str(row["text"] or "").strip():
            report.problems.append(Problem("text_missing", document_id, "empty text"))

    for _, row in gold.iterrows():
        document_id = str(row["document_id"])
        field_name = str(row["field"])
        subject = f"{document_id}/{field_name}"

        if field_name not in FIELDS:
            report.problems.append(
                Problem("field_unknown", subject, f"{field_name!r} is not an extractable field")
            )

        text = text_by_id.get(document_id)
        if text is None:
            report.problems.append(
                Problem("orphan_claim", subject, "no document with this id in the corpus")
            )
            continue

        quote = str(row["quote"] or "")
        if not quote.strip():
            report.problems.append(Problem("quote_missing", subject, "empty quotation"))
        elif normalize_whitespace(quote) not in normalize_whitespace(str(text)):
            # The check the whole repository rests on. If a reference quotation
            # is not in its own source document, every score computed against it
            # is meaningless, and no amount of care in the scorer recovers that.
            report.problems.append(
                Problem(
                    "quote_not_in_document",
                    subject,
                    f"quotation is not present in the source: {quote[:60]!r}...",
                )
            )

        if not str(row["value"] or "").strip():
            report.problems.append(Problem("value_missing", subject, "empty value"))

    return report


def check_gold_is_exhaustive(documents: pd.DataFrame, gold: pd.DataFrame) -> list[str]:
    """Return documents with no reference annotations at all.

    Separate from `check` because it is a different kind of statement. The
    scorer treats a missing (document, field) pair as "the article does not
    report this", which is what makes a false positive meaningful. A document
    with *no* annotations would therefore assert that the article reports
    nothing — almost never true, and it would silently punish every extractor.
    """
    annotated = set(gold["document_id"])
    return [d for d in documents["document_id"] if d not in annotated]
