"""The corpus, the frozen split, the ingest contract, and the provenance checks."""

from __future__ import annotations

import pandas as pd
import pytest

from open_eviden.bench.corpus import DEV_PERCENT, SPLITS, split_of, split_summary
from open_eviden.contracts import CLAIM_CONTRACT, DOCUMENT_CONTRACT, REDISTRIBUTABLE_LICENSES
from open_eviden.qa import provenance
from open_eviden.records import conform_documents, deduplicate_documents, normalize_whitespace

# --- the split -------------------------------------------------------------


def test_the_split_is_stable_across_processes(corpus):
    """Python's `hash` is randomized per process; this must not be.

    With `hash`, the split would change between two runs on the same machine and
    every published score would quietly stop being comparable — a bug that takes
    a week to see.
    """
    first = {d: split_of(d) for d in corpus.documents["document_id"]}
    second = {d: split_of(d) for d in corpus.documents["document_id"]}

    assert first == second
    assert set(first.values()) <= set(SPLITS)


def test_adding_a_document_does_not_move_an_existing_one(corpus):
    """The reason the split hashes the id instead of shuffling with a seed.

    A seeded shuffle reassigns everything on every ingest, so scores from before
    an ingest silently stop being comparable to scores from after it.
    """
    before = {d: split_of(d) for d in corpus.documents["document_id"]}

    for new_id in ("pmc:PMC99999999", "pmc:PMC00000001", "doi:10.1234/new"):
        split_of(new_id)  # a newcomer gets an assignment...

    after = {d: split_of(d) for d in corpus.documents["document_id"]}
    assert before == after  # ...and nobody else moves


def test_both_splits_are_populated(corpus):
    summary = split_summary(corpus.documents)

    assert summary["dev"] > 0, "no development set to tune or eyeball on"
    assert summary["test"] > 0, "nothing to publish a score against"
    assert sum(summary.values()) == len(corpus)


def test_the_split_proportion_is_in_the_right_neighbourhood(corpus):
    """Loose bounds on purpose: this is a hash, not a quota.

    On a corpus this small the realised fraction can sit well away from the
    nominal one, and tightening this test would make it fail on a legitimate
    ingest.
    """
    summary = split_summary(corpus.documents)
    dev_fraction = summary["dev"] / sum(summary.values())

    assert 0 < dev_fraction < 2 * DEV_PERCENT / 100


def test_an_extractor_cannot_tell_which_split_it_is_on(corpus):
    """Documents carry no split label, so behaviour cannot differ between them."""
    for document in corpus.split("test").as_documents():
        assert not hasattr(document, "split")


# --- the ingest contract ---------------------------------------------------


def _document_row(**overrides) -> dict:
    row = {
        "document_id": "pmc:PMC1",
        "doi": "10.1234/abc",
        "pmcid": "PMC1",
        "pmid": "1",
        "title": "A trial",
        "journal": "J Test",
        "year": 2025,
        "license": "cc by",
        "source": "europepmc",
        "text": "x" * 5000,
        "sections": "[]",
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"doi": ""}, "doi_missing"),
        ({"doi": "not-a-doi"}, "doi_malformed"),
        ({"license": "cc by-nc"}, "license_not_redistributable"),
        ({"license": "cc by-nc-nd"}, "license_not_redistributable"),
        ({"text": "too short"}, "text_too_short"),
        ({"title": ""}, "title_missing"),
        ({"year": 1500}, "year_implausible"),
    ],
)
def test_the_contract_quarantines_bad_records_with_a_reason(overrides, reason):
    frame = pd.DataFrame([_document_row(), _document_row(document_id="pmc:PMC2", **overrides)])
    clean, quarantined = DOCUMENT_CONTRACT.apply(frame)

    assert len(clean) == 1
    assert len(quarantined) == 1
    assert reason in quarantined.iloc[0]["_reasons"]


def test_a_bad_record_does_not_fail_the_whole_batch():
    """One malformed article must not block a corpus build."""
    rows = [_document_row(document_id=f"pmc:PMC{i}") for i in range(5)]
    rows.append(_document_row(document_id="pmc:PMCbad", doi=""))

    clean, quarantined = DOCUMENT_CONTRACT.apply(pd.DataFrame(rows))

    assert len(clean) == 5
    assert len(quarantined) == 1


def test_every_broken_rule_is_recorded_not_just_the_first():
    """No DOI *and* no licence is a different kind of broken from either alone."""
    frame = pd.DataFrame([_document_row(doi="", license="cc by-nc")])
    _, quarantined = DOCUMENT_CONTRACT.apply(frame)
    reasons = quarantined.iloc[0]["_reasons"]

    assert "doi_missing" in reasons
    assert "license_not_redistributable" in reasons


def test_non_redistributable_licences_are_the_point_of_that_rule():
    """CC BY-NC is open to read and not ours to republish."""
    assert "cc by" in REDISTRIBUTABLE_LICENSES
    assert "cc by-nc" not in REDISTRIBUTABLE_LICENSES
    assert "cc by-nc-nd" not in REDISTRIBUTABLE_LICENSES


def test_the_claim_contract_rejects_a_fragment_quotation():
    frame = pd.DataFrame(
        [
            {"document_id": "d", "field": "n_randomised", "value": "1", "quote": "short"},
            {
                "document_id": "d",
                "field": "n_randomised",
                "value": "1",
                "quote": "a quotation long enough to evidence something",
            },
        ]
    )
    clean, quarantined = CLAIM_CONTRACT.apply(frame)

    assert len(clean) == 1
    assert "quote_too_short" in quarantined.iloc[0]["_reasons"]


# --- conforming ------------------------------------------------------------


def test_conform_lowercases_the_doi_and_strips_the_resolver_prefix():
    frame = pd.DataFrame([_document_row(doi="https://doi.org/10.1234/ABC")])
    out = conform_documents(frame)

    assert out.iloc[0]["doi"] == "10.1234/abc"


def test_conform_is_idempotent_on_whitespace():
    """Section offsets are computed against normalized text; normalizing twice
    must not shift them."""
    text = "one   two\n\nthree"
    assert normalize_whitespace(normalize_whitespace(text)) == normalize_whitespace(text)


def test_conform_refuses_to_invent_a_missing_column():
    """In a repository whose claim is checkable provenance, a silently invented
    identifier is the worst possible bug."""
    frame = pd.DataFrame([{k: v for k, v in _document_row().items() if k != "doi"}])

    with pytest.raises(ValueError, match="missing columns"):
        conform_documents(frame)


def test_deduplicate_keeps_the_later_revision():
    frame = pd.DataFrame(
        [_document_row(title="first version"), _document_row(title="corrected version")]
    )
    out = deduplicate_documents(conform_documents(frame))

    assert len(out) == 1
    assert out.iloc[0]["title"] == "corrected version"


# --- provenance ------------------------------------------------------------


def test_the_committed_sample_passes_every_offline_provenance_check(documents, gold):
    report = provenance.check(documents, gold)

    assert report.ok, "\n".join(str(p) for p in report.problems)


def test_every_committed_document_has_reference_annotations(documents, gold):
    """A document with no annotations asserts the article reports nothing.

    That is almost never true, and it would silently penalise every extractor
    on that document.
    """
    assert provenance.check_gold_is_exhaustive(documents, gold) == []


def test_provenance_catches_a_reference_quotation_that_is_not_in_its_source():
    documents = pd.DataFrame([_document_row(text="The primary outcome was mortality.")])
    gold = pd.DataFrame(
        [
            {
                "document_id": "pmc:PMC1",
                "field": "primary_outcome",
                "value": "mortality",
                "quote": "This sentence appears nowhere in the article at all.",
            }
        ]
    )

    report = provenance.check(documents, gold)

    assert not report.ok
    assert "quote_not_in_document" in report.by_check()


def test_provenance_catches_a_claim_about_a_document_that_is_not_in_the_corpus():
    documents = pd.DataFrame([_document_row()])
    gold = pd.DataFrame(
        [
            {
                "document_id": "pmc:PMCmissing",
                "field": "design",
                "value": "crossover",
                "quote": "a quotation long enough to evidence something",
            }
        ]
    )

    assert "orphan_claim" in provenance.check(documents, gold).by_check()
