"""Shared fixtures.

Everything here works offline from the committed sample, so the whole suite runs
on a fresh clone with no network and no API key. That is a deliberate
constraint: a benchmark whose tests need the internet is a benchmark whose CI
fails for reasons unrelated to the code, and one that needs a key is a benchmark
most people cannot verify at all.
"""

from __future__ import annotations

import pandas as pd
import pytest

from open_eviden.bench.corpus import Corpus
from open_eviden.lake import read_corpus, read_gold
from open_eviden.synthesis.base import Claim, Document


@pytest.fixture(scope="session")
def documents() -> pd.DataFrame:
    """The committed corpus: real CC BY trial reports from Europe PMC."""
    return read_corpus()


@pytest.fixture(scope="session")
def gold() -> pd.DataFrame:
    """The committed reference annotations, hand-built and hand-checked."""
    return read_gold()


@pytest.fixture(scope="session")
def corpus(documents, gold) -> Corpus:
    return Corpus(documents=documents, gold=gold)


@pytest.fixture(scope="session")
def sample_document(corpus) -> Document:
    """One real article, for tests that need genuine publisher text."""
    return corpus.as_documents()[0]


@pytest.fixture
def synthetic_document() -> Document:
    """A tiny article with hand-checkable content.

    Written to contain one of each thing the extractors look for, plus the trap
    that motivates `sections.py`: a Discussion sentence describing a *different*
    trial's participant count.
    """
    text = (
        "Abstract A total of 240 patients were randomly assigned to drug or placebo "
        "in this double-blind, parallel-group trial. The primary outcome was change "
        "in HAM-D score at 12 weeks. Registered as NCT01234567. "
        "Methods Participants attended weekly for 12 weeks of follow-up. "
        "Discussion In the trial by Smith et al., 1204 patients were randomised, a "
        "much larger sample than ours."
    )
    from open_eviden.synthesis.base import Section

    discussion_start = text.index("Discussion")
    return Document(
        document_id="test:synthetic",
        doi="10.1234/synthetic",
        title="A synthetic trial report",
        journal="Journal of Test Fixtures",
        year=2026,
        text=text,
        sections=(
            Section(title="Abstract", start=0, end=text.index("Methods") - 1),
            Section(title="Methods", start=text.index("Methods"), end=discussion_start - 1),
            Section(title="Discussion", start=discussion_start, end=len(text)),
        ),
    )


@pytest.fixture
def claim() -> Claim:
    return Claim(field="n_randomised", value="240", quote="", section="")
