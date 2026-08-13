"""The corpus, the frozen split, and the only door an extractor gets.

`Corpus` holds the documents and the reference annotations side by side because
the benchmark needs both. An *extractor* is never handed a `Corpus`. It is handed
one `Document` at a time, built here, carrying the article and nothing else — no
annotations, no split label, no way back to this object.

The split
---------
Which documents are development and which are test is decided by hashing the
document identifier, not by position, not by shuffling with a seed, and not by
date. The consequence is the point: adding an article to the corpus never moves
an existing article from one side to the other. A seeded shuffle would reassign
everything on every ingest, and published scores from before the ingest would
quietly stop being comparable to scores from after it.

Thirty percent development, seventy test. Development is what a method may be
tuned on and read by eye; test is what gets published. Nothing in the code
prevents you from tuning on test — nothing can, since you have both files in your
clone — so this is a protocol you keep rather than a wall you cannot climb, and
that is stated plainly rather than dressed up as a guarantee. What *is*
structural is that a method cannot tell at runtime which side it is on, so it
cannot behave differently on them by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b

import pandas as pd

from ..lake import read_corpus, read_gold
from ..records import CLAIM_COLUMNS
from ..synthesis.base import Claim, Document

#: Fraction of the corpus assigned to development, as an integer percentage.
#: Frozen. Changing it repartitions the benchmark and invalidates every published
#: score, which is why it is a module constant with this comment on it and not a
#: parameter someone can pass in a hurry.
DEV_PERCENT = 30

#: Domain separation for the split hash. Prevents this hash from ever colliding
#: with a hash computed for another purpose over the same identifiers.
_SPLIT_SALT = b"open-eviden/split/v1"

SPLITS = ("dev", "test")


def split_of(document_id: str) -> str:
    """Assign a document to a split, deterministically and independently.

    blake2b rather than Python's `hash`, which is randomized per process by
    default: with `hash` the split would silently change between two runs on the
    same machine, which is the kind of bug that takes a week to see.
    """
    digest = blake2b(document_id.encode("utf-8"), key=_SPLIT_SALT, digest_size=8).digest()
    return "dev" if int.from_bytes(digest, "big") % 100 < DEV_PERCENT else "test"


@dataclass(frozen=True)
class Corpus:
    documents: pd.DataFrame
    gold: pd.DataFrame

    @classmethod
    def load(cls, root=None) -> Corpus:
        return cls(documents=read_corpus(root), gold=read_gold(root))

    def split(self, split: str) -> Corpus:
        """A view containing only one side of the split."""
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
        keep = self.documents["document_id"].map(split_of) == split
        documents = self.documents.loc[keep].reset_index(drop=True)
        gold = self.gold.loc[self.gold["document_id"].isin(set(documents["document_id"]))]
        return Corpus(documents=documents, gold=gold.reset_index(drop=True))

    def as_documents(self) -> list[Document]:
        """Build the objects extractors actually receive.

        This is the only place a `Document` is constructed from corpus data, and
        it copies across exactly the declared fields. If a column is added to the
        corpus table — a funding statement, a risk-of-bias assessment — it does
        not reach extractors until someone adds it to `Document` on purpose.
        """
        return [Document.from_row(row) for _, row in self.documents.iterrows()]

    def gold_claims(self, document_id: str) -> list[Claim]:
        rows = self.gold.loc[self.gold["document_id"] == document_id]
        return [
            Claim(
                field=str(row["field"]),
                value=str(row["value"]),
                quote=str(row["quote"]),
                section=str(row.get("section", "") or ""),
            )
            for _, row in rows.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.documents)


def split_summary(documents: pd.DataFrame) -> dict[str, int]:
    """Document counts per split, for run records and for sanity."""
    assigned = documents["document_id"].map(split_of)
    return {split: int((assigned == split).sum()) for split in SPLITS}


def check_gold_covers_corpus(corpus: Corpus) -> list[str]:
    """Return document ids that have no reference annotations at all.

    A document nobody annotated is not a hard failure — it is scored as though
    the article reports none of the fields, which is almost never true, so it
    would quietly punish every extractor on it. The benchmark refuses to run on
    one rather than producing a number that looks fine and is wrong.
    """
    annotated = set(corpus.gold["document_id"])
    return [d for d in corpus.documents["document_id"] if d not in annotated]


def gold_frame_is_wellformed(gold: pd.DataFrame) -> None:
    """Raise if the reference table is missing columns the scorer depends on."""
    missing = [c for c in CLAIM_COLUMNS if c not in gold.columns]
    if missing:
        raise ValueError(f"reference annotations are missing columns: {missing}")
