"""Scoring: did the extractor get the value right, and did it point at real text.

Two questions, kept apart on purpose, because a method can fail either one and
the failures mean opposite things. An extractor with high grounding and low
recall is cautious. One with high recall and low grounding is confabulating. A
single headline number would rank them identically and tell you nothing about
which one you could put in front of a reviewer.

Counting
--------
A *target* is a (document, field) pair the reference set has a value for. The
reference set is exhaustive per document: if a field has no reference row, the
article does not report it, and a claim about it is a false positive rather than
an unscored extra. That property is checked in `tests/test_gold.py` — without it
precision would silently mean nothing.

    true positive   grounded claim exists, and its value matches the reference
    false positive  grounded claim exists, and it does not (wrong, or not reported)
    false negative  reference value exists, no grounded claim

Ungrounded claims are not false positives. They never entered scoring at all —
they were discarded by `synthesis.base.ground` before this module saw them — and
they are reported separately as the grounding rate. Folding them into precision
would let a method improve its precision by fabricating quotations badly enough
to be caught, which is the wrong incentive by a wide margin.

Comparison is per field kind, defined in `records.FIELDS`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field

from ..records import FIELDS, FieldKind
from ..synthesis.base import Claim, Grounding

#: Token overlap at or above this counts a free-text field as correct.
#:
#: There is no principled value for this and pretending otherwise would be
#: dishonest. 0.6 was picked by looking at the development set: it accepts
#: "change in HAM-D score at 12 weeks" against "change from baseline in HAM-D
#: total score" and rejects "adverse events". Every score involving
#: `primary_outcome` inherits that arbitrariness, which is why the per-field
#: table is reported alongside the totals — so a reader can see how much of a
#: result rests on this number.
TEXT_MATCH_THRESHOLD = 0.6

#: Tolerance for `follow_up_weeks`. Papers report duration in months and reviewers
#: convert with different constants — 3 months is 12 weeks to one annotator and
#: 13.04 to another. Half a week absolute, 2% relative, so a year-long follow-up
#: is not failed over a rounding convention while a 4-vs-6-week disagreement
#: still counts as a disagreement.
NUMBER_ABS_TOL = 0.5
NUMBER_REL_TOL = 0.02

_TOKEN = re.compile(r"[a-z0-9]+")

#: Words carrying no discriminative content in outcome descriptions. Kept
#: deliberately short: an aggressive stop list would let "change in mortality"
#: match "change in pain" on the word "change".
_STOPWORDS = frozenset({"the", "a", "an", "of", "in", "at", "to", "and", "or", "for", "from"})


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def token_f1(predicted: str, reference: str) -> float:
    """Bag-of-tokens F1, the standard span-comparison measure.

    Order-insensitive, which is right here: "HAM-D change from baseline" and
    "change from baseline in HAM-D" are the same outcome described by an author
    with a different house style.
    """
    p, r = _tokens(predicted), _tokens(reference)
    if not p or not r:
        return 1.0 if not p and not r else 0.0

    overlap = 0
    remaining = list(r)
    for token in p:
        if token in remaining:
            remaining.remove(token)
            overlap += 1
    if overlap == 0:
        return 0.0

    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def _normalize_identifier(value: str) -> str:
    """Registry ids differ in case, spacing and hyphens across a single paper."""
    return re.sub(r"[\s\-_]", "", value).upper()


def values_match(field: str, predicted: str, reference: str) -> bool:
    """Compare a predicted value to the reference, by the field's declared kind."""
    spec = FIELDS[field]
    predicted, reference = str(predicted).strip(), str(reference).strip()

    if spec.kind is FieldKind.INTEGER:
        try:
            # int(float(...)) so that "240.0" from a JSON round-trip still
            # compares equal to 240. "240 participants" does not parse and is a
            # mismatch, not a crash: the field asks for a count, not a phrase.
            return int(float(predicted)) == int(float(reference))
        except (TypeError, ValueError):
            return False

    if spec.kind is FieldKind.NUMBER:
        try:
            return math.isclose(
                float(predicted), float(reference), rel_tol=NUMBER_REL_TOL, abs_tol=NUMBER_ABS_TOL
            )
        except (TypeError, ValueError):
            return False

    if spec.kind is FieldKind.CATEGORICAL:
        # A value outside the vocabulary is wrong rather than an error. The
        # vocabulary is published in records.FIELDS and in the issue template; an
        # extractor that invents a term has made a mistake the same way it would
        # by choosing the wrong term.
        return predicted.strip().lower() == reference.strip().lower()

    if spec.kind is FieldKind.IDENTIFIER:
        return _normalize_identifier(predicted) == _normalize_identifier(reference)

    return token_f1(predicted, reference) >= TEXT_MATCH_THRESHOLD


@dataclass
class FieldScore:
    field: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    #: Mean token F1 over targets where both a prediction and a reference exist.
    #: Only meaningful for TEXT fields; NaN elsewhere.
    mean_token_f1: float = float("nan")

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else float("nan")

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if not (p > 0) or not (r > 0):
            return 0.0 if (p == p and r == r) else float("nan")
        return 2 * p * r / (p + r)

    def as_dict(self) -> dict:
        return {
            "field": self.field,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_token_f1": self.mean_token_f1,
        }


@dataclass
class DocumentScore:
    document_id: str
    per_field: dict[str, FieldScore] = dataclass_field(default_factory=dict)
    n_returned: int = 0
    n_grounded: int = 0
    rejection_reasons: dict[str, int] = dataclass_field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "n_returned": self.n_returned,
            "n_grounded": self.n_grounded,
            "grounding_rate": self.n_grounded / self.n_returned
            if self.n_returned
            else float("nan"),
            "rejection_reasons": self.rejection_reasons,
            "per_field": {name: score.as_dict() for name, score in self.per_field.items()},
        }


def score_document(
    document_id: str,
    grounding: Grounding,
    reference: Sequence[Claim],
) -> DocumentScore:
    """Score one document's grounded claims against its reference annotations."""
    predicted = {c.field: c for c in grounding.claims}
    expected = {c.field: c for c in reference}

    result = DocumentScore(
        document_id=document_id,
        n_returned=grounding.n_returned,
        n_grounded=len(grounding.claims),
        rejection_reasons=grounding.reason_counts(),
    )

    for name in FIELDS:
        score = FieldScore(field=name)
        prediction = predicted.get(name)
        target = expected.get(name)

        if prediction is not None and target is not None:
            if values_match(name, prediction.value, target.value):
                score.true_positives = 1
            else:
                score.false_positives = 1
                score.false_negatives = 1
            if FIELDS[name].kind is FieldKind.TEXT:
                score.mean_token_f1 = token_f1(prediction.value, target.value)
        elif prediction is not None:
            # The reference set is exhaustive, so no target means the article
            # does not report this field and the extractor invented a value for it.
            score.false_positives = 1
        elif target is not None:
            score.false_negatives = 1

        result.per_field[name] = score

    return result


def aggregate(document_scores: Sequence[DocumentScore]) -> dict:
    """Micro-average across documents, plus the per-field breakdown.

    Micro rather than macro: a field reported by two articles out of thirty
    should not weigh the same as one reported by all thirty. Macro would let a
    rare field dominate the headline number, and `registration_id` is exactly
    that rare on this corpus.
    """
    per_field: dict[str, FieldScore] = {name: FieldScore(field=name) for name in FIELDS}
    text_f1s: dict[str, list[float]] = {name: [] for name in FIELDS}

    n_returned = n_grounded = 0
    reasons: dict[str, int] = {}

    for doc in document_scores:
        n_returned += doc.n_returned
        n_grounded += doc.n_grounded
        for reason, count in doc.rejection_reasons.items():
            reasons[reason] = reasons.get(reason, 0) + count
        for name, score in doc.per_field.items():
            total = per_field[name]
            total.true_positives += score.true_positives
            total.false_positives += score.false_positives
            total.false_negatives += score.false_negatives
            if score.mean_token_f1 == score.mean_token_f1:  # not NaN
                text_f1s[name].append(score.mean_token_f1)

    for name, values in text_f1s.items():
        if values:
            per_field[name].mean_token_f1 = sum(values) / len(values)

    tp = sum(s.true_positives for s in per_field.values())
    fp = sum(s.false_positives for s in per_field.values())
    fn = sum(s.false_negatives for s in per_field.values())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision > 0 and recall > 0) else 0.0

    return {
        "n_documents": len(document_scores),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_claims_returned": n_returned,
        "n_claims_grounded": n_grounded,
        "grounding_rate": n_grounded / n_returned if n_returned else float("nan"),
        "rejection_reasons": dict(sorted(reasons.items())),
        "per_field": {name: score.as_dict() for name, score in per_field.items()},
    }
