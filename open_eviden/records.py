"""The canonical schemas, and the fields that get extracted.

Two tables. `DOCUMENT_SCHEMA` is one row per article: its identifiers, its
licence, and its text. `CLAIM_SCHEMA` is one row per extracted statement about an
article, and every row of it carries a quotation from that article.

Every source connector, whatever database it reads from, produces the document
shape and only that shape. An extractor written against `open-eviden` never
learns the quirks of Europe PMC's JSON or PubMed's XML — it sees one table, and
swapping the source out is not a code change for anyone downstream.

On the field set
----------------
The fields below are the ones a systematic reviewer fills in for every included
study before anything interesting can be computed: how many people, what design,
how blinded, how long, registered where. They are deliberately not the
*conclusions* of a study. This repository extracts what a paper says about how it
was conducted; it does not extract, rank, or pool what a paper claims to have
shown, and nothing here should be read as evidence that a treatment works. See
DISCLAIMER.md.

`primary_outcome` is the exception and the weak spot, because it is free text
with no single right phrasing. It is scored by token overlap rather than by
equality, and that scoring is the least trustworthy number this benchmark
produces. It is included anyway because leaving out the field reviewers care
most about would make the benchmark easy and useless.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

DOCUMENT_SCHEMA: dict[str, str] = {
    "document_id": "string",  # source-qualified, e.g. "pmc:PMC13180848"
    "doi": "string",
    "pmcid": "string",
    "pmid": "string",
    "title": "string",
    "journal": "string",
    "year": "Int64",  # nullable: a few OA records genuinely have no year
    "license": "string",  # normalized, e.g. "cc by"
    "source": "string",  # connector key: "europepmc", ...
    "text": "string",  # whitespace-normalized full text
    # JSON list of {"title", "start", "end"}: character offsets into `text`.
    #
    # Stored rather than re-detected, because `text` is normalized to a single
    # line and the structure would otherwise have to be guessed from heading
    # words in flowing prose — which mis-splits on any sentence containing the
    # word "Methods". The publisher's own XML knows where the sections are; this
    # keeps that knowledge instead of throwing it away and approximating it back.
    "sections": "string",
}

DOCUMENT_COLUMNS = list(DOCUMENT_SCHEMA)

CLAIM_SCHEMA: dict[str, str] = {
    "document_id": "string",
    "field": "string",
    "value": "string",  # always stored as text; typed comparison happens at scoring
    "quote": "string",  # verbatim span from the document, the whole point
    "section": "string",  # where the extractor believes it found it; advisory only
}

CLAIM_COLUMNS = list(CLAIM_SCHEMA)


class FieldKind(StrEnum):
    """How a field's value is compared to the reference value."""

    INTEGER = "integer"
    NUMBER = "number"
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"
    TEXT = "text"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: FieldKind
    description: str
    vocabulary: tuple[str, ...] = ()


#: The extractable fields. A new field here changes what every published score
#: means, so adding one is a protocol change — see .github/expert-loop/CONTRIBUTING-DRAFT.md.
FIELDS: dict[str, FieldSpec] = {
    spec.name: spec
    for spec in (
        FieldSpec(
            "n_randomised",
            FieldKind.INTEGER,
            "Number of participants randomised, as reported. Not the number "
            "screened, enrolled, or analysed — those are different numbers and "
            "papers report all four.",
        ),
        FieldSpec(
            "design",
            FieldKind.CATEGORICAL,
            "Allocation structure of the trial.",
            vocabulary=(
                "parallel-group",
                "crossover",
                "cluster-randomised",
                "factorial",
                "single-arm",
            ),
        ),
        FieldSpec(
            "blinding",
            FieldKind.CATEGORICAL,
            "Masking as described by the authors. There is deliberately no "
            "'unclear' term: a paper that never states its masking has no "
            "reference row for this field, and the correct behaviour is to say "
            "nothing rather than to record a guess. See the note on absence in "
            "the README — it is a real limitation of quotation-grounded "
            "extraction, not an oversight.",
            vocabulary=(
                "open-label",
                "single-blind",
                "double-blind",
                "triple-blind",
            ),
        ),
        FieldSpec(
            "follow_up_weeks",
            FieldKind.NUMBER,
            "Planned duration of follow-up for the primary outcome, in weeks. "
            "Papers state this in weeks, months, days and 'at discharge'; the "
            "reference value is whatever the stated period converts to, and the "
            "quote must contain the stated period.",
        ),
        FieldSpec(
            "registration_id",
            FieldKind.IDENTIFIER,
            "Trial registry identifier, e.g. NCT01234567, ISRCTN12345678, "
            "ChiCTR2000012345. Compared case-insensitively with internal "
            "whitespace and hyphens removed.",
        ),
        FieldSpec(
            "primary_outcome",
            FieldKind.TEXT,
            "The primary outcome measure as the authors define it. Scored by "
            "token overlap, not equality.",
        ),
    )
}

FIELD_NAMES = tuple(FIELDS)


def empty_documents() -> pd.DataFrame:
    """A zero-row document frame with the right dtypes, for the no-results path."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in DOCUMENT_SCHEMA.items()})


def empty_claims() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in CLAIM_SCHEMA.items()})


def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace to a single space, and strip.

    Applied identically to document text at ingest and to a quotation at
    grounding time. Without it, a quotation that is correct in every way except
    that the PDF broke a line in the middle of it would fail to match, and
    extractors would be scored on typesetting.

    It is applied *once*, at ingest, so the stored text and the offsets computed
    against it agree. Normalizing again at scoring time would be harmless here
    but would stop being harmless the moment anyone stored an offset.
    """
    return " ".join(text.split())


def conform_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a source-shaped frame to the canonical document schema.

    Missing columns are an error, not a fill — a source that cannot supply a DOI
    should say so explicitly with a null column rather than have one invented
    here. In a repository whose whole claim is that provenance is checkable, a
    silently invented identifier is the worst possible bug.
    """
    missing = [c for c in DOCUMENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"cannot conform, missing columns: {missing}")

    out = df.loc[:, DOCUMENT_COLUMNS].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["sections"] = out["sections"].astype("string").fillna("[]")
    for col in ("document_id", "doi", "pmcid", "pmid", "title", "journal", "license", "source"):
        out[col] = out[col].astype("string")
    out["text"] = out["text"].astype("string").map(
        lambda t: normalize_whitespace(t) if isinstance(t, str) else t
    )
    # DOIs are case-insensitive by specification and are conventionally written
    # lowercase; registries return both. Lowercasing here means a join on DOI
    # does not silently miss.
    out["doi"] = out["doi"].str.lower().str.removeprefix("https://doi.org/")
    out["license"] = out["license"].str.lower().str.strip()

    return out.sort_values("document_id", kind="stable").reset_index(drop=True)


def deduplicate_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated document_ids, keeping the last.

    Articles are revised, and Europe PMC will hand back a corrected full text for
    a record ingested months earlier. Last-wins means a re-run improves the
    corpus rather than duplicating it.
    """
    return (
        df.sort_values("document_id", kind="stable")
        .drop_duplicates(subset="document_id", keep="last")
        .reset_index(drop=True)
    )
