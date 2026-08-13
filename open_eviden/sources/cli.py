"""Build a corpus from Europe PMC.

    python -m open_eviden.sources.cli --query "randomized controlled trial" --limit 40
    python -m open_eviden.sources.cli --dataset mimiciv    # what you'd be agreeing to

The result lands in `data/silver/corpus.parquet`, which git ignores. The
committed sample under `data/sample/` was built with this same command and is
what everything runs on by default, so a clone reproduces every published number
without ever calling this.
"""

from __future__ import annotations

import argparse
import sys

from ..contracts import DOCUMENT_CONTRACT
from ..lake import write_corpus
from . import europepmc, physionet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m open_eviden.sources.cli",
        description="Fetch open-access articles and conform them to the corpus schema.",
    )
    parser.add_argument(
        "--query",
        default="randomized controlled trial",
        help="Europe PMC query. CC BY and open-access filters are always added.",
    )
    parser.add_argument("--limit", type=int, default=25, help="maximum articles to fetch")
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "instead of fetching, print what obtaining a PhysioNet dataset would "
            f"require ({sorted(physionet.KNOWN)})"
        ),
    )
    args = parser.parse_args(argv)

    if args.dataset:
        print(physionet.instructions(args.dataset))
        return 0

    print(f"Searching Europe PMC: {args.query!r} (limit {args.limit})", file=sys.stderr)
    documents = europepmc.fetch(args.query, limit=args.limit)

    if documents.empty:
        print("No articles matched, or none had deposited full text.", file=sys.stderr)
        return 1

    clean, quarantined = DOCUMENT_CONTRACT.apply(documents)
    path = DOCUMENT_CONTRACT.write_quarantine(quarantined)

    if not quarantined.empty:
        print(f"\nQuarantined {len(quarantined)} record(s) -> {path}", file=sys.stderr)
        for reason, count in quarantined["_reasons"].value_counts().items():
            print(f"  {reason}: {count}", file=sys.stderr)

    if clean.empty:
        print("\nEvery record was quarantined. Nothing written.", file=sys.stderr)
        return 1

    out = write_corpus(clean)
    print(f"\n{len(clean)} documents -> {out}", file=sys.stderr)
    print(
        "\nThis corpus has no reference annotations, so the benchmark cannot score "
        "against it yet. The committed sample under data/sample/ is the annotated one.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
