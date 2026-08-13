"""Row-level validation with quarantine, at the ingest boundary.

The same contract/dead-letter pattern `seismic-bench` uses on earthquake
catalogs, for the same reason: a literature corpus assembled from a public API is
messy in a small number of predictable ways, and both obvious responses to that
are wrong. Failing the whole load on a few bad records means one malformed
article blocks a year of ingest. Silently letting them through means a record
with no DOI quietly enters a table whose entire purpose is that every row has a
resolvable source.

So the batch is split: clean rows continue, bad rows go to `_quarantine/` tagged
with which rule they broke, and the run reports the count. The failure stays
visible without being fatal, and the quarantine file is a browsable record of
what your source actually sends you.

There is one rule here that is not a data-quality rule at all. `license_not_cc_by`
is a legal boundary wearing a contract's clothes: an article that is open access
but not CC BY cannot be redistributed in this repository, and quarantining it
means the mistake is recorded in a file rather than committed to git.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .lake import quarantine_dir
from .records import FIELD_NAMES

#: DOI syntax, per the Crossref-recommended pattern. This checks *shape*, not
#: existence — whether a DOI resolves is a network question and lives in
#: open_eviden/qa/doi.py, deliberately separated so that ingest stays offline.
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/[-._;()/:a-z0-9]+$", re.IGNORECASE)

#: Licences we are free to redistribute. CC BY only, and not by oversight:
#: the Europe PMC OA subset also contains CC BY-NC and CC BY-NC-ND articles,
#: which are open to read and not open to republish. See LICENSE-DATA.
REDISTRIBUTABLE_LICENSES = frozenset({"cc by", "cc-by", "cc0", "public domain"})

#: Below this, a "full text" record is an abstract or a stub, and an extractor
#: scored on it is being scored on a different task. Chosen from the observed
#: distribution: real OA research articles run tens of thousands of characters,
#: and the records that fall under this are consistently abstract-only.
MIN_TEXT_CHARS = 2_000


@dataclass(frozen=True)
class Rule:
    """A named predicate. `ok` returns a row-wise mask of *valid* rows."""

    name: str
    ok: Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Contract:
    table: str
    rules: tuple[Rule, ...]

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split into (clean, quarantined).

        Quarantined rows carry a `_reasons` column listing every rule they broke,
        semicolon-separated — not just the first, because a record with no DOI
        and no licence is a different kind of broken from one missing only a DOI,
        and you want to see that in the file.
        """
        if df.empty:
            return df.copy(), df.copy()

        reasons = pd.Series([""] * len(df), index=df.index, dtype="object")
        for rule in self.rules:
            broke = ~rule.ok(df).fillna(False).astype(bool)
            reasons.loc[broke] = (reasons.loc[broke] + ";" + rule.name).str.lstrip(";")

        is_bad = reasons != ""
        clean = df.loc[~is_bad].copy()
        quarantined = df.loc[is_bad].copy()
        if not quarantined.empty:
            quarantined["_reasons"] = reasons.loc[is_bad]

        return clean.reset_index(drop=True), quarantined.reset_index(drop=True)

    def write_quarantine(self, quarantined: pd.DataFrame, root: Path | None = None) -> Path | None:
        if quarantined.empty:
            return None
        out_dir = quarantine_dir(root)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.table}.parquet"
        quarantined.to_parquet(path, index=False)
        return path


def _nonempty(df: pd.DataFrame, col: str) -> pd.Series:
    value = df[col].astype("string")
    return value.notna() & (value.str.strip() != "")


#: The document contract.
#:
#: Bounds are deliberately generous where the question is quality and strict
#: where the question is provenance or licensing. A short article is a judgement
#: call; a missing DOI is not.
DOCUMENT_CONTRACT = Contract(
    table="documents",
    rules=(
        Rule("document_id_missing", lambda d: _nonempty(d, "document_id")),
        Rule("title_missing", lambda d: _nonempty(d, "title")),
        # No DOI means no way to check the source resolves, which is the one
        # guarantee this repository makes. It is not a recoverable record.
        Rule("doi_missing", lambda d: _nonempty(d, "doi")),
        Rule(
            "doi_malformed",
            lambda d: d["doi"].astype("string").fillna("").str.match(DOI_PATTERN),
        ),
        Rule(
            "license_not_redistributable",
            lambda d: d["license"].astype("string").str.lower().isin(REDISTRIBUTABLE_LICENSES),
        ),
        Rule("text_missing", lambda d: _nonempty(d, "text")),
        Rule(
            "text_too_short",
            lambda d: d["text"].astype("string").fillna("").str.len() >= MIN_TEXT_CHARS,
        ),
        # Publication years outside this range are parse failures, not articles.
        # The lower bound predates any journal this corpus draws from; the upper
        # bound catches an epoch-seconds value landing in a year column.
        Rule(
            "year_implausible",
            lambda d: pd.to_numeric(d["year"], errors="coerce").between(1800, 2100)
            | d["year"].isna(),
        ),
    ),
)

#: The claim contract, applied to *reference annotations* at load.
#:
#: Extractor output does not go through this. It goes through
#: `open_eviden.synthesis.base.ground`, which is stricter in the way that
#: matters: it checks the quotation against the actual document text, which a
#: row-wise contract cannot do because it has no access to the corpus.
CLAIM_CONTRACT = Contract(
    table="claims",
    rules=(
        Rule("document_id_missing", lambda d: _nonempty(d, "document_id")),
        Rule("field_unknown", lambda d: d["field"].isin(FIELD_NAMES)),
        Rule("value_missing", lambda d: _nonempty(d, "value")),
        Rule("quote_missing", lambda d: _nonempty(d, "quote")),
        # A three-word quotation does not evidence anything — it is a fragment
        # that will match somewhere in almost any article. The floor is low
        # enough to admit "randomised 240 participants" and high enough to
        # exclude a bare number, which would otherwise ground trivially.
        Rule("quote_too_short", lambda d: d["quote"].astype("string").fillna("").str.len() >= 20),
    ),
)
