"""The deterministic baselines, and the failure that separates them."""

from __future__ import annotations

from open_eviden.synthesis.base import ground
from open_eviden.synthesis.rules import (
    RegexExtractor,
    extract_claims,
    find_n_randomised,
    sentence_around,
)
from open_eviden.synthesis.sections import SectionScopedExtractor, role_of


def _by_field(claims):
    return {c.field: c for c in claims}


def _primary_outcome(text):
    """The value the rules read out of `text`, or None if they read nothing."""
    return next((c.value for c in extract_claims(text) if c.field == "primary_outcome"), None)


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


def test_an_outcome_named_in_brackets_after_the_label_is_read():
    """"…for the primary outcome (premenstrual syndrome severity)…"

    Apposition rather than a copula. CONSORT 2010 item 6a asks for a completely
    defined primary outcome measure (https://doi.org/10.1136/bmj.c332) and says
    nothing about the construction, so this names the outcome as squarely as
    "the primary outcome was" does. Fragment from PMC13282192, reported in #1.
    """
    text = (
        "The calculation was based on the total PMSS score reported by Arıöz and Ege, "
        "with an effect size (Cohen's d) of 0.86 for the primary outcome "
        "(premenstrual syndrome severity)."
    )

    assert _primary_outcome(text) == "premenstrual syndrome severity"


def test_an_outcome_named_before_the_label_is_read():
    """"…on time trial performance (our primary outcome)…"

    The same apposition with the label second, so the outcome sits to the *left*
    of the words "primary outcome" and a rule that only looks rightward cannot
    reach it — per @nextbaires (#1), reading PMC13235231 by hand. The capture
    has to stop at "on", or it returns the clause the outcome is embedded in.
    """
    text = (
        "The sample size was determined a priori and informed by our previous work; "
        "assuming an effect size of the current intervention on time trial performance "
        "(our primary outcome) of dz = 0.9, 12 participants were deemed necessary."
    )

    assert _primary_outcome(text) == "time trial performance"


def test_a_qualifier_between_the_noun_and_the_copula_does_not_defeat_the_rule():
    """"The primary outcomes *of this feasibility trial* were …"

    Ordinary English, and it used to return nothing at all: the rule wanted
    "outcomes were" adjacent. Fragment from PMC13279985, reported in #1.

    Note for a domain reviewer: what this reads is the trial's own declaration,
    "intervention practicality and acceptability". The reference annotation for
    that article summarises the same outcomes as "feasibility: recruitment,
    retention, compliance", so recovering this sentence does not recover the
    score — see the pull request. Which of the two a reviewer would record is a
    domain question, not a pattern question.
    """
    text = (
        "These guidelines were not invoked during the trial. The primary outcomes of "
        "this feasibility trial were intervention practicality and acceptability, "
        "assessed through recruitment, retention, and compliance metrics."
    )

    value = _primary_outcome(text)
    assert value is not None
    assert value.startswith("intervention practicality and acceptability")


def test_a_sentence_that_merely_mentions_the_primary_outcome_names_none():
    """The condition #1 offers for its own proposal being wrong.

    If these shapes cannot be matched without also matching sample-size and
    discussion sentences that only mention the phrase, the narrow rule was the
    right trade. Each of these mentions it and defines nothing, and each must
    produce no claim rather than a boundary invented around a percentage.
    """
    mentions_only = (
        "The sample size was calculated to detect a difference in the primary outcome "
        "with 90% power at a two-sided alpha of 0.05.",
        "Assuming the primary outcome rate in the control group is 20%, 240 "
        "participants would be required.",
        "Our trial did not show a difference in the primary outcome, which was "
        "consistent with previous studies of this intervention.",
        "Participants who did not complete the primary outcome assessment were "
        "excluded from the per-protocol analysis.",
        "No changes were made after study initiation to the primary objective, primary "
        "end point, eligibility criteria, or core study design.",
    )

    for text in mentions_only:
        assert _primary_outcome(text) is None, text


def test_a_captured_outcome_drops_the_citation_it_was_padded_with():
    """The value loses the reference marker. The quotation keeps everything.

    "changes in the PI (according to Silness and Loe [30])" is a correct capture
    padded with the author's referencing, per @nextbaires (#1). Trimming it
    touches the value only — the quotation stays the sentence sliced out of the
    document, which is what grounding checks and what a reviewer reads.
    """
    text = (
        "Outcomes The primary outcome was depressive symptoms measured by the Chinese "
        "version of PHQ-9 [ 49 ]. Its Cronbach α was 0.91 in this study."
    )
    claim = _by_field(extract_claims(text))["primary_outcome"]

    assert claim.value == "depressive symptoms measured by the Chinese version of PHQ-9"
    assert claim.quote in text
    assert "[ 49 ]" in claim.quote


def test_a_capture_cut_off_inside_a_bracket_loses_the_fragment():
    """"reduction in liver fat-fraction (≥ 2" — the cap landed mid-bracket.

    The value is trimmed back to the last balanced point rather than reported
    with half a parenthetical hanging off it.
    """
    text = "The primary outcome was a reduction in liver fat-fraction (≥ 2.0 points) at 24 weeks."

    assert _primary_outcome(text) == "a reduction in liver fat-fraction"


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
