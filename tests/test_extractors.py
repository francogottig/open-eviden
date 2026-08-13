"""The deterministic baselines, and the failure that separates them."""

from __future__ import annotations

from open_eviden.synthesis.base import ground
from open_eviden.synthesis.rules import RegexExtractor, find_n_randomised, sentence_around
from open_eviden.synthesis.sections import SectionScopedExtractor, role_of


def _by_field(claims):
    return {c.field: c for c in claims}


def test_the_naive_baseline_takes_a_count_from_the_discussion(synthetic_document):
    """The error the section-scoped extractor exists to remove.

    "In the trial by Smith et al., 1204 patients were randomised" is a real
    sentence, correctly quoted, about somebody else's study. It is also the
    first randomisation sentence in the document that this pattern reaches
    after the abstract, and a pattern-matcher has no way to know the difference.
    """
    claims = _by_field(RegexExtractor().extract(synthetic_document))
    assert claims["n_randomised"].value == "240"  # abstract comes first

    # But the discussion sentence is findable, and would win if it came first.
    discussion = synthetic_document.text[synthetic_document.text.index("Discussion") :]
    found = find_n_randomised(discussion)
    assert found is not None and found[0] == 1204


def test_the_section_scoped_extractor_never_reads_the_discussion(synthetic_document):
    claims = SectionScopedExtractor().extract(synthetic_document)
    discussion = synthetic_document.text[synthetic_document.text.index("Discussion") :]

    for claim in claims:
        assert claim.quote not in discussion


def test_both_baselines_ground_everything_they_return(corpus):
    """Not virtue — they have no mechanism for inventing anything.

    Any method that beats them on recall while grounding worse is trading away
    something these give free.
    """
    for extractor in (RegexExtractor(), SectionScopedExtractor()):
        for document in corpus.as_documents():
            grounding = ground(extractor.extract(document), document)
            assert grounding.rejected == [], (
                f"{extractor.name} produced an ungrounded claim on {document.document_id}"
            )


def test_a_quotation_is_always_a_span_of_the_document(corpus):
    for document in corpus.as_documents():
        for claim in SectionScopedExtractor().extract(document):
            assert claim.quote in document.text


def test_quotations_clear_the_contract_length_floor(corpus):
    """A quotation short enough to appear in any article evidences nothing."""
    for document in corpus.as_documents():
        for claim in SectionScopedExtractor().extract(document):
            assert len(claim.quote) >= 20


def test_typographic_hyphens_are_matched(synthetic_document):
    """Publishers emit U+2010, which renders identically to a hyphen.

    This cost four of six blinding annotations before it was found, and it is
    invisible on screen — hence a test rather than a comment.
    """
    from dataclasses import replace

    document = replace(
        synthetic_document,
        text=synthetic_document.text.replace("double-blind", "double‐blind"),
        sections=(),
    )
    claims = _by_field(RegexExtractor().extract(document))

    assert claims["blinding"].value == "double-blind"


def test_spelled_out_participant_counts_are_read():
    """"Twelve trained cyclists" opens a trial report often enough to matter."""
    text = "Twelve trained cyclists took part in a randomized, placebo-controlled trial."
    found = find_n_randomised(text)

    assert found is not None and found[0] == 12


def test_an_implausibly_small_count_is_not_taken_as_a_participant_count():
    """"randomised into 2 groups" counts arms, not people."""
    text = "Participants were randomly assigned into 2 groups of equal size."
    assert find_n_randomised(text) is None


def test_section_roles_exclude_the_discussion():
    from open_eviden.synthesis.base import Section

    assert role_of(Section("Materials and methods", 0, 1)) == "methods"
    assert role_of(Section("Patients and Methods", 0, 1)) == "methods"
    assert role_of(Section("Results", 0, 1)) == "results"
    assert role_of(Section("Abstract", 0, 1)) == "abstract"
    assert role_of(Section("Discussion", 0, 1)) == "other"
    assert role_of(Section("Introduction", 0, 1)) == "other"
    assert role_of(Section("References", 0, 1)) == "other"


def test_the_section_extractor_falls_back_when_a_publisher_deposits_no_structure(
    synthetic_document,
):
    """A missing-sections penalty would belong to the publisher, not the method."""
    from dataclasses import replace

    document = replace(synthetic_document, sections=())
    claims = SectionScopedExtractor().extract(document)

    assert claims
    assert all("no sections found" in c.section for c in claims)


def test_sentence_around_returns_a_verbatim_span():
    text = "First sentence here. Second sentence with the match. Third one."
    index = text.index("match")
    quote = sentence_around(text, index, index + 5)

    assert quote in text
    assert quote == "Second sentence with the match."
