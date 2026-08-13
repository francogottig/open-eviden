"""Run the quality checks.

    python -m open_eviden.qa.cli              # offline provenance checks
    python -m open_eviden.qa.cli --doi        # also resolve every DOI (network)

Two modes because the checks have two different failure characters, and mixing
them costs you the offline one. `provenance` fails only when this repository is
wrong; `--doi` can fail because a publisher's server is having a bad afternoon.
CI runs them as separate jobs for the same reason.
"""

from __future__ import annotations

import argparse
import sys

from ..lake import read_corpus, read_gold
from . import doi as doi_check
from . import provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m open_eviden.qa.cli",
        description="Check that every extracted claim still points at a real source.",
    )
    parser.add_argument(
        "--doi",
        action="store_true",
        help="also resolve every DOI against doi.org (requires network)",
    )
    parser.add_argument(
        "--allow-network-errors",
        action="store_true",
        help=(
            "treat a timeout or connection failure as inconclusive rather than a "
            "failure. Use in CI; a DOI check that fails on a flaky network gets "
            "switched off within a month."
        ),
    )
    args = parser.parse_args(argv)

    documents, gold = read_corpus(), read_gold()

    print(f"Checking {len(documents)} documents and {len(gold)} reference claims")
    report = provenance.check(documents, gold)

    unannotated = provenance.check_gold_is_exhaustive(documents, gold)
    for document_id in unannotated:
        report.problems.append(
            provenance.Problem(
                "document_unannotated",
                document_id,
                "no reference annotations; scoring would treat the article as "
                "reporting nothing",
            )
        )

    if report.ok:
        print("Provenance: ok")
    else:
        print(f"\nProvenance: {len(report.problems)} problem(s)", file=sys.stderr)
        for name, count in report.by_check().items():
            print(f"  {name}: {count}", file=sys.stderr)
        print(file=sys.stderr)
        for problem in report.problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    if not args.doi:
        print("\nDOI resolution skipped. Run with --doi to check it (needs network).")
        return 0

    print(f"\nResolving {len(documents)} DOIs")
    resolutions = doi_check.resolve_all(
        zip(documents["document_id"], documents["doi"], strict=True)
    )

    dead = [r for r in resolutions if r.status == doi_check.UNRESOLVED_STATUS]
    inconclusive = [r for r in resolutions if r.status is None]

    if dead:
        print(f"\n{len(dead)} DOI(s) do not resolve:", file=sys.stderr)
        for r in dead:
            print(f"  {r.document_id}: {r.doi}", file=sys.stderr)
        return 1

    if inconclusive:
        message = f"{len(inconclusive)} DOI(s) could not be checked (network)"
        if args.allow_network_errors:
            print(f"\n{message} — treated as inconclusive, not a failure.")
        else:
            print(f"\n{message}:", file=sys.stderr)
            for r in inconclusive:
                print(f"  {r.document_id}: {r.doi} — {r.error}", file=sys.stderr)
            return 1

    print(f"\nDOI resolution: {len(resolutions) - len(inconclusive)} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
