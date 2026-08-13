"""The structural guarantee: a value counts only if its quotation is real.

This is `seismic-bench`'s `test_split.py` for this repository — the file that
asserts the thing the benchmark's meaning rests on, including against
extractors written specifically to get around it.

If a change makes these tests pass by weakening them, the change is wrong.
"""

from __future__ import annotations

import dataclasses

import pytest

from open_eviden.bench.corpus import Corpus
from open_eviden.records import FIELDS
from open_eviden.synthesis.base import Claim, Document, ground, validate_claims


class RecallingExtractor:
    """Returns correct values with no quotation — the memorisation failure mode.

    This is what a language model does when it answers from what it read during
    training rather than from the document in front of it. Every value here is
    right; not one of them is grounded.
    """

    name = "recalling"

    def extract(self, document: Document):
        return [Claim(field="n_randomised", value="240", quote="", section="")]

    def describe(self) -> dict:
        return {}


class FabricatingExtractor:
    """Returns correct values with invented quotations.

    Fluent, plausible, and not in the document. The most dangerous failure mode
    in the repository, because the output looks *more* trustworthy than an
    honest abstention.
    """

    name = "fabricating"

    def extract(self, document: Document):
        return [
            Claim(
                field="n_randomised",
                value="240",
                quote="A total of 240 participants were enrolled in this study.",
                section="Methods",
            )
        ]

    def describe(self) -> dict:
        return {}


def test_a_correct_value_without_a_quotation_does_not_count(synthetic_document):
    """Right answer, no evidence, no credit."""
    grounding = ground(RecallingExtractor().extract(synthetic_document), synthetic_document)

    assert grounding.claims == []
    assert grounding.reason_counts() == {"quote_missing": 1}
    assert grounding.grounding_rate == 0.0


def test_a_correct_value_with_a_fabricated_quotation_does_not_count(synthetic_document):
    """The quotation is fluent and plausible. It is not in the document."""
    grounding = ground(FabricatingExtractor().extract(synthetic_document), synthetic_document)

    assert grounding.claims == []
    assert grounding.reason_counts() == {"quote_not_in_document": 1}


def test_a_near_miss_quotation_is_rejected(synthetic_document):
    """One word changed is one word too many.

    A quotation that has been lightly edited — a tense fixed, a word swapped —
    is not a quotation. Accepting near misses would reintroduce exactly the gap
    that grounding closes, because a remembered sentence is usually *almost*
    right.
    """
    real = "The primary outcome was change in HAM-D score at 12 weeks."
    assert real in synthetic_document.text

    edited = real.replace("change", "improvement")
    grounding = ground(
        [Claim(field="primary_outcome", value="HAM-D", quote=edited)], synthetic_document
    )

    assert grounding.claims == []
    assert grounding.reason_counts() == {"quote_not_in_document": 1}


def test_whitespace_differences_are_forgiven(synthetic_document):
    """A line break inside a quotation is the publisher's doing, not a fabrication."""
    real = "The primary outcome was change in HAM-D score at 12 weeks."
    rewrapped = real.replace(" ", "\n   ", 3)

    grounding = ground(
        [Claim(field="primary_outcome", value="HAM-D", quote=rewrapped)], synthetic_document
    )

    assert len(grounding.claims) == 1


def test_case_differences_are_not_forgiven(synthetic_document):
    """"Verbatim" is the claim being made; re-casing is editing."""
    real = "The primary outcome was change in HAM-D score at 12 weeks."

    grounding = ground(
        [Claim(field="primary_outcome", value="HAM-D", quote=real.lower())], synthetic_document
    )

    assert grounding.claims == []


def test_only_the_first_claim_per_field_survives(synthetic_document):
    """Offering three candidate values and hoping one lands is guessing."""
    real = "The primary outcome was change in HAM-D score at 12 weeks."
    guesses = [
        Claim(field="n_randomised", value=str(v), quote=real) for v in (240, 250, 260)
    ]

    grounding = ground(guesses, synthetic_document)

    assert len(grounding.claims) == 1
    assert grounding.claims[0].value == "240"
    assert grounding.reason_counts() == {"duplicate_field": 2}


def test_an_unknown_field_is_rejected(synthetic_document):
    real = "The primary outcome was change in HAM-D score at 12 weeks."
    grounding = ground(
        [Claim(field="funding_source", value="NIH", quote=real)], synthetic_document
    )

    assert grounding.claims == []
    assert grounding.reason_counts() == {"unknown_field": 1}


def test_grounding_rate_is_nan_when_nothing_was_returned(synthetic_document):
    """Abstaining is not perfect grounding.

    Reporting 1.0 for an extractor that returned nothing would put a method that
    does nothing at the top of the grounding column.
    """
    grounding = ground([], synthetic_document)

    assert grounding.grounding_rate != grounding.grounding_rate  # NaN


def test_the_document_carries_no_route_to_the_reference_answers(corpus):
    """An extractor cannot reach the reference set, the corpus, or its split.

    The guarantee is structural rather than procedural: there is no argument to
    pass and no attribute to follow. This test fails the moment somebody adds a
    convenience handle to `Document` — which is exactly when it should.
    """
    document = corpus.as_documents()[0]

    assert set(vars(document)) == {
        "document_id",
        "doi",
        "title",
        "journal",
        "year",
        "text",
        "sections",
    }

    for attribute in ("gold", "corpus", "split", "reference", "answers", "claims"):
        assert not hasattr(document, attribute), (
            f"Document.{attribute} would let an extractor see what it is being scored against"
        )


def test_documents_are_frozen(sample_document):
    """An extractor must not be able to edit the text it is quoted against."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample_document.text = "something else"


def test_a_section_slice_is_still_a_span_of_the_document(sample_document):
    """Quoting out of a section must ground without special handling."""
    for section in sample_document.sections:
        assert sample_document.section_text(section) in sample_document.text


def test_every_reference_quotation_is_verbatim_in_its_document(corpus):
    """The reference set is held to the rule it enforces.

    If a reference quotation is not in its own source, every score computed
    against it is meaningless — and no amount of care elsewhere recovers that.
    """
    text_by_id = dict(zip(corpus.documents["document_id"], corpus.documents["text"], strict=True))

    for row in corpus.gold.itertuples():
        haystack = " ".join(str(text_by_id[row.document_id]).split())
        needle = " ".join(str(row.quote).split())
        assert needle in haystack, f"{row.document_id}/{row.field}: reference quote is not verbatim"


def test_reference_fields_are_all_extractable_fields(corpus):
    assert set(corpus.gold["field"]) <= set(FIELDS)


def test_validate_claims_rejects_a_wrong_return_type():
    """A submission returning the wrong shape gets its name in the error.

    Filtering silently instead would score the submission zero and leave its
    author reading a leaderboard wondering why their method looks terrible.
    """
    with pytest.raises(TypeError, match="broken"):
        validate_claims([{"field": "design", "value": "crossover"}], "broken")

    with pytest.raises(TypeError, match="broken"):
        validate_claims(None, "broken")


def test_corpus_split_is_a_view_not_a_copy_of_the_gold(corpus):
    """A split's reference set must cover exactly that split's documents."""
    for split in ("dev", "test"):
        view = Corpus.split(corpus, split)
        assert set(view.gold["document_id"]) <= set(view.documents["document_id"])
