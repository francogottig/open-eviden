"""Scoring behaviour: what counts, what does not, and why."""

from __future__ import annotations

import math

from open_eviden.bench import metrics
from open_eviden.synthesis.base import Claim, Grounding


def _grounded(*claims: Claim) -> Grounding:
    return Grounding(claims=list(claims))


def test_integer_comparison_tolerates_a_json_round_trip():
    assert metrics.values_match("n_randomised", "240.0", "240")
    assert not metrics.values_match("n_randomised", "241", "240")


def test_an_integer_field_given_a_phrase_is_wrong_not_an_error():
    """The field asks for a count. "240 participants" is a mistake, not a crash."""
    assert not metrics.values_match("n_randomised", "240 participants", "240")


def test_number_comparison_absorbs_a_month_conversion_disagreement():
    """Three months is 12 weeks to one annotator and 13.04 to another."""
    assert metrics.values_match("follow_up_weeks", "13.04", "13.0")
    assert metrics.values_match("follow_up_weeks", "52.18", "52.0")
    # But a real disagreement is still a disagreement.
    assert not metrics.values_match("follow_up_weeks", "4", "6")


def test_identifiers_compare_across_spacing_and_case():
    assert metrics.values_match("registration_id", "nct 012 34567", "NCT01234567")
    assert not metrics.values_match("registration_id", "NCT01234568", "NCT01234567")


def test_a_categorical_value_outside_the_vocabulary_is_wrong_not_an_error():
    assert not metrics.values_match("blinding", "double blinded", "double-blind")
    assert metrics.values_match("blinding", "Double-Blind", "double-blind")


def test_token_f1_is_order_insensitive():
    """Two house styles describing the same outcome are the same outcome."""
    a = "change from baseline in HAM-D total score"
    b = "HAM-D total score change from baseline"
    assert metrics.token_f1(a, b) == 1.0


def test_text_matching_accepts_a_paraphrase_and_rejects_a_different_outcome():
    reference = "change from baseline in HAM-D total score"
    assert metrics.values_match("primary_outcome", "change in HAM-D score", reference)
    assert not metrics.values_match("primary_outcome", "adverse events", reference)


def test_a_claim_about_a_field_the_article_does_not_report_is_a_false_positive():
    """The reference set is exhaustive, which is what makes precision mean something."""
    prediction = Claim(field="blinding", value="double-blind", quote="q")
    score = metrics.score_document("d1", _grounded(prediction), reference=[])

    assert score.per_field["blinding"].false_positives == 1
    assert score.per_field["blinding"].true_positives == 0


def test_a_wrong_value_counts_against_both_precision_and_recall():
    """Saying the wrong thing is worse than saying nothing, and is scored as such."""
    prediction = Claim(field="n_randomised", value="999", quote="q")
    reference = [Claim(field="n_randomised", value="240", quote="q")]

    score = metrics.score_document("d1", _grounded(prediction), reference)
    field = score.per_field["n_randomised"]

    assert (field.false_positives, field.false_negatives, field.true_positives) == (1, 1, 0)


def test_an_abstention_costs_recall_only():
    reference = [Claim(field="n_randomised", value="240", quote="q")]
    score = metrics.score_document("d1", _grounded(), reference)
    field = score.per_field["n_randomised"]

    assert (field.false_positives, field.false_negatives) == (0, 1)


def test_ungrounded_claims_never_reach_precision():
    """Fabricating badly enough to be caught must not improve a method's precision.

    The rejected claim is reported in the grounding rate and is absent from the
    true/false positive counts entirely.
    """
    grounding = Grounding(
        claims=[Claim(field="n_randomised", value="240", quote="q")],
        rejected=[(Claim(field="blinding", value="double-blind", quote="invented"),
                   "quote_not_in_document")],
    )
    reference = [Claim(field="n_randomised", value="240", quote="q")]

    score = metrics.score_document("d1", grounding, reference)

    assert score.per_field["blinding"].false_positives == 0
    assert score.n_returned == 2
    assert score.n_grounded == 1
    assert score.rejection_reasons == {"quote_not_in_document": 1}


def test_aggregate_micro_averages_and_keeps_the_per_field_breakdown():
    reference = [Claim(field="n_randomised", value="240", quote="q")]
    right = _grounded(Claim(field="n_randomised", value="240", quote="q"))
    wrong = _grounded(Claim(field="n_randomised", value="1", quote="q"))

    totals = metrics.aggregate(
        [
            metrics.score_document("d1", right, reference),
            metrics.score_document("d2", wrong, reference),
        ]
    )

    assert totals["true_positives"] == 1
    assert totals["precision"] == 0.5
    assert totals["recall"] == 0.5
    assert totals["per_field"]["n_randomised"]["f1"] == 0.5


def test_aggregate_reports_nan_for_a_field_nobody_predicted_or_referenced():
    """No data is not the same as a score of zero, and must not be shown as one."""
    totals = metrics.aggregate([metrics.score_document("d1", _grounded(), reference=[])])

    assert math.isnan(totals["per_field"]["blinding"]["precision"])
