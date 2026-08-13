"""PhysioNet: identifiers and agreements, never data.

This module downloads nothing. That is its whole design.

PhysioNet hosts datasets under a range of access levels, and the useful clinical
ones — MIMIC-IV among them — are *credentialed*: you complete human-subjects
research training, sign a data use agreement, and are granted access as an
individual. That agreement is between you and the data's custodians. Code cannot
accept it on your behalf, and a convenience function that fetched the data with
a token pasted into an environment variable would be a tool for violating it.

So this resolves an identifier to the landing page, states the access level and
the agreement you would be signing, and tells you where the data goes if you
obtain it: `data/restricted/`, which git ignores and which nothing else in this
package reads.

The one function that touches the filesystem, `restricted_path`, refuses to
return a path outside that directory. That is the actual control — a
`.gitignore` entry is one `git add -f` away from a disclosure, whereas code that
cannot name a path outside `data/restricted/` cannot write one there by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..lake import restricted_dir

BASE_URL = "https://physionet.org/content/"


@dataclass(frozen=True)
class Dataset:
    slug: str
    name: str
    access: str  # "open" | "restricted" | "credentialed"
    note: str

    @property
    def url(self) -> str:
        return f"{BASE_URL}{self.slug}/"

    @property
    def requires_agreement(self) -> bool:
        return self.access != "open"


#: Datasets this repository knows how to talk about. Deliberately short: it
#: exists to answer "what am I agreeing to" for the datasets people actually ask
#: about, not to mirror PhysioNet's catalogue.
KNOWN: dict[str, Dataset] = {
    d.slug: d
    for d in (
        Dataset(
            "mimiciv",
            "MIMIC-IV",
            "credentialed",
            "Requires CITI human-subjects training and a signed data use "
            "agreement, granted to you as a named individual. It is not "
            "transferable, and derived tables built from it stay bound by it.",
        ),
        Dataset(
            "mimic-iv-demo",
            "MIMIC-IV Clinical Database Demo",
            "open",
            "A 100-patient demo under the Open Data Commons ODbL. Open to "
            "download — and still not committed here: it is patient-level data, "
            "and a repository that never contains any is easier to trust than "
            "one that contains only the allowed kind.",
        ),
        Dataset(
            "eicu-crd",
            "eICU Collaborative Research Database",
            "credentialed",
            "Same credentialing route as MIMIC-IV.",
        ),
    )
}


def describe(slug: str) -> Dataset:
    """Look up a dataset, or fail with the list of the ones we know."""
    if slug not in KNOWN:
        raise KeyError(f"unknown dataset {slug!r}. Known: {sorted(KNOWN)}")
    return KNOWN[slug]


def instructions(slug: str) -> str:
    """What you would have to do to obtain this dataset yourself."""
    dataset = describe(slug)
    lines = [
        f"{dataset.name} ({dataset.slug})",
        f"  Access:   {dataset.access}",
        f"  Landing:  {dataset.url}",
        f"  Note:     {dataset.note}",
        "",
    ]
    if dataset.requires_agreement:
        lines += [
            "  This repository will not fetch it for you, and you should be wary",
            "  of anything that offers to. Complete the credentialing process on",
            "  PhysioNet, download it yourself, and place it under:",
            f"    {restricted_dir()}",
            "",
            "  Nothing in open_eviden reads that directory. It is not an input to",
            "  the benchmark, it is never published to the leaderboard, and it must",
            "  never be committed.",
        ]
    else:
        lines += [
            "  Openly downloadable. Even so, put it under:",
            f"    {restricted_dir()}",
            "  and do not commit it — see DISCLAIMER.md on why this repository holds",
            "  no patient-level data of any access level.",
        ]
    return "\n".join(lines)


def restricted_path(*parts: str, root: Path | None = None) -> Path:
    """Resolve a path inside `data/restricted/`, refusing anything outside it.

    The traversal check is not paranoia about a hostile caller — it is about a
    dataset slug interpolated from a config file, or a `..` that arrives through
    three layers of helper. The consequence of getting it wrong here is writing
    credentialed clinical data into a tracked directory, so the function that
    resolves the path is the right place to make that impossible.
    """
    base = restricted_dir(root).resolve()
    candidate = base.joinpath(*parts).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(
            f"refusing to resolve {candidate} — credentialed data is confined to {base}"
        )
    return candidate
