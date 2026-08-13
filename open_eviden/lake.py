"""Where things live on disk.

A flat medallion layout: raw source pulls land in `bronze/`, the conformed and
validated corpus in `silver/`, and anything a benchmark run consumes in `gold/`.
Only the small committed sample under `data/sample/` is in version control;
everything else is generated and gitignored, so a clone is small and a
`git status` after a full ingest is still clean.

`restricted/` is the exception that matters in this repository. Data obtained
under a data use agreement goes there and nowhere else, and nothing in this
package ever reads it into anything that gets committed. See
`open_eviden/sources/physionet.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def data_root(root: Path | None = None) -> Path:
    """Resolve the data directory.

    Order: explicit argument, then `OPEN_EVIDEN_DATA`, then `<repo>/data`. The
    environment variable exists so CI and a laptop can point at different disks
    without a flag on every command.
    """
    if root is not None:
        return Path(root)
    env = os.environ.get("OPEN_EVIDEN_DATA")
    return Path(env) if env else REPO_ROOT / "data"


def bronze_dir(root: Path | None = None) -> Path:
    return data_root(root) / "bronze"


def silver_dir(root: Path | None = None) -> Path:
    return data_root(root) / "silver"


def gold_dir(root: Path | None = None) -> Path:
    return data_root(root) / "gold"


def quarantine_dir(root: Path | None = None) -> Path:
    return data_root(root) / "_quarantine"


def restricted_dir(root: Path | None = None) -> Path:
    """Credentialed data, obtained under the user's own agreement.

    Separate from the medallion layers on purpose. Nothing here is ever read by
    the benchmark, published to the leaderboard, or committed — a directory that
    is merely gitignored is one `git add -f` away from a disclosure, so keeping
    it out of the paths every other function reaches is the actual control.
    """
    return data_root(root) / "restricted"


def sample_corpus_path() -> Path:
    """The committed corpus sample. Small enough for git, real enough to test on."""
    return REPO_ROOT / "data" / "sample" / "corpus.sample.parquet"


def sample_gold_path() -> Path:
    """The committed reference annotations, hand-built and hand-checked."""
    return REPO_ROOT / "data" / "sample" / "gold.sample.parquet"


def corpus_path(root: Path | None = None) -> Path:
    return silver_dir(root) / "corpus.parquet"


def write_corpus(df: pd.DataFrame, root: Path | None = None) -> Path:
    path = corpus_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_corpus(root: Path | None = None) -> pd.DataFrame:
    """Read the full corpus, falling back to the committed sample.

    The fallback is what lets CI, and a newcomer who has just cloned, run the
    whole benchmark end to end without network access or an ingest step. It is
    also why every test in this repository can be run offline.
    """
    path = corpus_path(root)
    if path.exists():
        return pd.read_parquet(path)
    sample = sample_corpus_path()
    if sample.exists():
        return pd.read_parquet(sample)
    raise FileNotFoundError(
        f"No corpus at {path} and no sample at {sample}. "
        "Run `python -m open_eviden.sources.cli --help` to fetch one."
    )


def read_gold(root: Path | None = None) -> pd.DataFrame:
    """Read the reference annotations.

    There is deliberately no `write_gold`. The reference set is hand-built and
    hand-checked; a function that regenerates it from a model's output would
    turn the benchmark into a mirror, and it would be used by accident within a
    month of existing.
    """
    path = gold_dir(root) / "gold.parquet"
    if path.exists():
        return pd.read_parquet(path)
    sample = sample_gold_path()
    if sample.exists():
        return pd.read_parquet(sample)
    raise FileNotFoundError(f"No reference annotations at {path} and no sample at {sample}.")
