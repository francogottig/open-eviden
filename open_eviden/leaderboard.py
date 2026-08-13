"""Aggregate individual runs into the published leaderboard.

    python -m open_eviden.leaderboard

Reads every `leaderboard/runs/*.json` and writes `leaderboard/results.json`,
which is what the website reads and what CI regenerates on merge.

Runs are only comparable if they used the same protocol, so entries are grouped
by a protocol fingerprint and ranked within a group. Two methods scored on
different splits, or against different matching thresholds, do not share a
ranking — quietly ranking them together is how a benchmark starts lying.

Non-deterministic entries are ranked alongside the rest but carry
`deterministic: false`, and CI never regenerates them. A number a reviewer
cannot reproduce is still worth publishing here — the alternative is excluding
every language model from a benchmark about extraction — but it has to be
labelled as what it is.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "leaderboard" / "runs"
RESULTS_PATH = REPO_ROOT / "leaderboard" / "results.json"


def protocol_key(run: dict) -> str:
    """A human-readable fingerprint of the scoring protocol."""
    protocol = run["protocol"]
    return (
        f"{protocol['split']}-split "
        f"{protocol['n_documents']}docs "
        f"dev{protocol['dev_percent']}% "
        f"text-f1>={protocol['text_match_threshold']}"
    )


#: Significant figures kept in the published file.
#:
#: Precision, recall and F1 are ratios of small integers, so they are exact in
#: binary far more often than a summed likelihood would be — but `mean_token_f1`
#: is an average of averages and does drift in the last places between
#: platforms. The committed file is diffed against a fresh run in CI, so
#: unrounded values would make the check fail on machines other than the one
#: that generated it, and the check would get switched off within a week.
#:
#: Nine figures sits clear of that drift and clear of anything meaningful: two
#: methods separated at the ninth significant figure of their F1 are not
#: distinguishable by a corpus this size.
SIGNIFICANT_FIGURES = 9


def _round(value):
    """Round floats to SIGNIFICANT_FIGURES, recursing into containers."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return (
            value
            if value != value or value in (float("inf"), float("-inf"))
            else float(f"{value:.{SIGNIFICANT_FIGURES}g}")
        )
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v) for v in value]
    return value


def collect(runs_dir: Path = RUNS_DIR) -> list[dict]:
    if not runs_dir.exists():
        return []
    return [json.loads(path.read_text()) for path in sorted(runs_dir.glob("*.json"))]


def build(runs: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for run in runs:
        totals = run["totals"]
        entry = {
            "extractor": run["extractor"],
            "precision": totals["precision"],
            "recall": totals["recall"],
            "f1": totals["f1"],
            "grounding_rate": totals["grounding_rate"],
            "n_claims_returned": totals["n_claims_returned"],
            "n_claims_grounded": totals["n_claims_grounded"],
            "n_documents": totals["n_documents"],
            "rejection_reasons": totals["rejection_reasons"],
            "per_field": {
                name: {"f1": s["f1"], "precision": s["precision"], "recall": s["recall"]}
                for name, s in totals["per_field"].items()
            },
            "deterministic": run["description"].get("deterministic"),
            "can_hallucinate": run["description"].get("can_hallucinate"),
            "description": run["description"],
        }
        groups.setdefault(protocol_key(run), []).append(_round(entry))

    # Ranked by F1. Precision and recall are both reported because the ranking
    # hides which one a method traded away, and that is usually the thing a
    # reader actually wants to know.
    for entries in groups.values():
        entries.sort(key=lambda e: (e["f1"], e["precision"]), reverse=True)
        for rank, entry in enumerate(entries, start=1):
            entry["rank"] = rank

    # Deliberately no timestamp and no library versions. This file is committed
    # and CI fails if it drifts from what the code produces, so anything that
    # varies between two identical runs would break that check on every push and
    # train everyone to ignore it. Provenance that does vary — when a run
    # happened, on which Python — stays in the per-run records under
    # leaderboard/runs/, which git does not track.
    return {
        "ranked_by": "f1, higher is better; ties broken by precision",
        "note": (
            "Entries are only comparable within a protocol group. The grounding "
            "rate matters independently of rank: a method can rank first on F1 "
            "while grounding poorly, which means it is producing correct values "
            "it cannot point at — and a value nobody can check against the source "
            "is the thing this benchmark exists to discourage."
        ),
        "protocols": [
            {"protocol": key, "entries": entries} for key, entries in sorted(groups.items())
        ],
    }


def main() -> int:
    runs = collect()
    if not runs:
        print(f"No runs found in {RUNS_DIR}. Run the benchmark first.")
        return 1

    results = build(runs)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")

    for group in results["protocols"]:
        print(f"\n{group['protocol']}")
        print(f"  {'rank':<5} {'extractor':<12} {'P':>6} {'R':>6} {'F1':>6} {'grounded':>9}")
        for entry in group["entries"]:
            print(
                f"  {entry['rank']:<5} {entry['extractor']:<12} "
                f"{entry['precision']:>6.2f} {entry['recall']:>6.2f} "
                f"{entry['f1']:>6.2f} {entry['grounding_rate']:>8.0%}"
            )
    print(f"\n-> {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
