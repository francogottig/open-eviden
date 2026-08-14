"""The deterministic baselines, and the failure that separates them."""

from __future__ import annotations

from open_eviden.synthesis.base import ground
from open_eviden.synthesis.rules import (
    RegexExtractor,
    clean_outcome,
    extract_claims,
    find_n_randomised,
    sentence_around,
)
from open_eviden.synthesis.sections import SectionScopedExtractor, role_of


def _by_field(claims):
    return {c.field: c for c in claims}


def _outcome(text):
    """The primary outcome read out of `text`, or None if the rules decline to name one."""
    claim = _by_field(extract_claims(text)).get("primary_outcome")
    return claim.value if claim else None


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


def test_an_outcome_named_in_a_bracket_after_the_label_is_read():
    """"…for the primary outcome (premenstrual syndrome severity)…"

    Apposition rather than a copula. CONSORT 2010 item 6a requires the outcome
    to be defined and says nothing about the construction
    (https://doi.org/10.1136/bmj.c332); this is one of the shapes @nextbaires
    found the copula rules returning nothing for (#1).
    """
    text = (
        "The calculation was based on the total PMSS score reported by Arıöz and Ege, "
        "with an effect size (Cohen's d) of 0.86 for the primary outcome "
        "(premenstrual syndrome severity)."
    )

    assert _outcome(text) == "premenstrual syndrome severity"


def test_an_outcome_named_before_the_label_is_read():
    """"…on time trial performance (our primary outcome)…"

    The outcome is named *before* the words "primary outcome", so a rule that
    only ever looks rightward cannot reach it — the second shape from #1.
    """
    text = (
        "The sample size was informed by our previous work; assuming an effect size of "
        "the current intervention on time trial performance (our primary outcome) of "
        "d z = 0.9, alpha = 0.05, and power of 80%, 12 participants were deemed necessary."
    )

    assert _outcome(text) == "time trial performance"


def test_reading_backwards_stops_at_the_start_of_the_noun_phrase():
    """The bound that keeps the backwards read from swallowing the clause.

    Without a stop at the first preposition the capture runs back through "the
    effect of" and takes half the sample-size calculation with it, which is the
    "invented boundaries" failure the proposal asks to be checked for.
    """
    value = _outcome(
        "We powered the study on the effect of the intervention on time trial "
        "performance (our primary outcome) at 80%."
    )

    assert value == "time trial performance"
    assert "effect" not in value


def test_a_qualifier_between_the_noun_and_the_verb_does_not_defeat_the_rule():
    """"The primary outcomes *of this feasibility trial* were …"

    Completely ordinary English, and the adjacency the copula rule requires is
    all that was defeating it (#1).
    """
    text = (
        "The primary outcomes of this feasibility trial were intervention practicality "
        "and acceptability."
    )

    assert _outcome(text) == "intervention practicality and acceptability"


def test_a_sentence_that_merely_mentions_the_primary_outcome_names_none():
    """The proposer's own falsification condition, encoded.

    From #1: "If it turns out that these three sentence shapes cannot be matched
    without also matching sample-size and discussion sentences that merely
    mention the phrase 'primary outcome', then the current narrow rule is the
    right trade and I am wrong."

    Sample-size, discussion, protocol-change and eligibility sentences all use
    the phrase without defining anything. Each of these is either taken from, or
    modelled on, the articles in the committed sample.
    """
    mentions_only = (
        # Sample size.
        "The sample size calculation was based on the primary outcome [ 12 ].",
        "Assuming a 20% difference in the primary outcome, 240 participants were "
        "required for 90% power.",
        "The primary outcome rate in the control group is expected to be 20%.",
        # Discussion, pointing at a table rather than naming anything.
        "Our result for the primary outcome (see Table 2) is consistent with earlier work.",
        # Protocol changes.
        "No changes were made after study initiation to the primary objective, primary "
        "end point, eligibility criteria, or core study design.",
        # Eligibility.
        "Participants were excluded if the primary outcome could not be assessed at baseline.",
    )

    for sentence in mentions_only:
        assert _outcome(sentence) is None, sentence


def test_a_citation_is_trimmed_from_the_captured_outcome():
    """"changes in the PI (according to Silness and Loe [ 30 ])" — the reference is padding."""
    text = (
        "The primary outcomes were changes in the PI (according to Silness and Loe [ 30 ]) "
        "and GI (according to Loe and Silness [ 31 ])."
    )

    assert _outcome(text) == (
        "changes in the PI (according to Silness and Loe) and GI (according to Loe and Silness)"
    )


def test_a_bracket_the_capture_cut_in_half_is_trimmed():
    """The full stop inside "2.8%" ends the capture between "(" and ")"."""
    text = (
        "The primary outcome was reduction in liver fat-fraction (≥ 2.8%), assessed by "
        "magnetic resonance imaging."
    )

    assert _outcome(text) == "reduction in liver fat-fraction"


def test_trimming_the_value_leaves_the_quotation_verbatim():
    """Trimming touches the value only — the quote is still a span of the document.

    Grounding is 100% by construction and a change that scored better while
    grounding worse should be rejected outright (#1), so the two have to be kept
    apart: `clean_outcome` rewrites what is reported, never what is quoted.
    """
    text = (
        "Outcomes The primary outcomes were changes in the PI (according to Silness and "
        "Loe [ 30 ]) at 12 weeks."
    )
    claim = _by_field(extract_claims(text))["primary_outcome"]

    assert "[ 30 ]" not in claim.value  # trimmed out of what is reported
    assert "[ 30 ]" in claim.quote  # still there in what is quoted
    assert claim.quote in text  # and the quotation is a span of the document


def test_clean_outcome_leaves_an_already_clean_capture_alone():
    assert clean_outcome("change in HAM-D score at 12 weeks") == "change in HAM-D score at 12 weeks"
    assert clean_outcome("feasibility (recruitment, retention and adherence) and safety") == (
        "feasibility (recruitment, retention and adherence) and safety"
    )


def test_sentence_around_returns_a_verbatim_span():
    text = "First sentence here. Second sentence with the match. Third one."
    index = text.index("match")
    quote = sentence_around(text, index, index + 5)

    assert quote in text
    assert quote == "Second sentence with the match."
