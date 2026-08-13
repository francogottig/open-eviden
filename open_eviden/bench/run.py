"""Run an extractor across the corpus and write the result.

The protocol is fixed here rather than configurable per extractor, which is the
point: two entries on the leaderboard differ only in the method, never in how
they were scored.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..synthesis.base import ground, validate_claims
from . import metrics
from .corpus import (
    DEV_PERCENT,
    Corpus,
    check_gold_covers_corpus,
    gold_frame_is_wellformed,
    split_summary,
)

#: Built-in extractors. A submission adds a line here, or passes
#: `--extractor package.module:ClassName` to run without touching the registry.
REGISTRY: dict[str, str] = {
    "regex": "open_eviden.synthesis.rules:RegexExtractor",
    "sections": "open_eviden.synthesis.sections:SectionScopedExtractor",
    # Not in the committed leaderboard: needs a key and a network, and does not
    # return the same thing twice. Runnable, and its results are recorded with
    # deterministic=false so a re-run is never mistaken for a reproduction.
    "llm": "open_eviden.synthesis.llm:LLMExtractor",
}


def load_extractor(spec: str):
    """Resolve a registry key or a `module:ClassName` path to an instance."""
    target = REGISTRY.get(spec, spec)
    if ":" not in target:
        raise ValueError(
            f"unknown extractor {spec!r}. Use one of {sorted(REGISTRY)} "
            "or a 'package.module:ClassName' path."
        )
    module_name, class_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


@dataclass
class RunResult:
    extractor: str
    description: dict
    protocol: dict
    per_document: list[dict] = field(default_factory=list)
    totals: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "extractor": self.extractor,
            "description": self.description,
            "protocol": self.protocol,
            "totals": self.totals,
            "per_document": self.per_document,
            "environment": self.environment,
        }


def run(
    extractor,
    corpus: Corpus,
    *,
    split: str = "test",
    verbose: bool = True,
) -> RunResult:
    """Score `extractor` on one side of the split.

    The extractor is handed one `Document` at a time and never sees the
    reference annotations — `corpus.gold_claims` is called here, after
    `extract` has already returned. That ordering is the whole guarantee, and
    `tests/test_grounding.py` asserts it holds even for an extractor written to
    look for a way around it.
    """
    gold_frame_is_wellformed(corpus.gold)
    view = corpus.split(split)

    unannotated = check_gold_covers_corpus(view)
    if unannotated:
        raise ValueError(
            f"{len(unannotated)} document(s) in the {split} split have no reference "
            f"annotations: {unannotated[:3]}. Scoring against an empty reference "
            "would silently penalise every extractor on those documents."
        )

    name = getattr(extractor, "name", type(extractor).__name__)
    document_scores: list[metrics.DocumentScore] = []

    for i, document in enumerate(view.as_documents(), start=1):
        returned = validate_claims(extractor.extract(document), name)
        grounding = ground(returned, document)
        reference = view.gold_claims(document.document_id)

        score = metrics.score_document(document.document_id, grounding, reference)
        document_scores.append(score)

        if verbose:
            print(
                f"  [{i}/{len(view)}] {document.document_id:<24} "
                f"returned={grounding.n_returned:2d} "
                f"grounded={len(grounding.claims):2d} "
                f"reference={len(reference):2d}",
                file=sys.stderr,
            )

    totals = metrics.aggregate(document_scores)

    return RunResult(
        extractor=name,
        description=extractor.describe(),
        protocol={
            "split": split,
            "dev_percent": DEV_PERCENT,
            "n_documents": len(view),
            "corpus_documents": len(corpus),
            "split_sizes": split_summary(corpus.documents),
            "text_match_threshold": metrics.TEXT_MATCH_THRESHOLD,
            "number_abs_tol": metrics.NUMBER_ABS_TOL,
            "number_rel_tol": metrics.NUMBER_REL_TOL,
            "grounding": "verbatim quotation required, whitespace-normalized",
        },
        per_document=[s.as_dict() for s in document_scores],
        totals=totals,
        environment={
            "python": platform.python_version(),
            "run_at": datetime.now(UTC).isoformat(),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m open_eviden.bench.run",
        description="Score an extraction method on the benchmark protocol.",
    )
    parser.add_argument(
        "--extractor",
        default="sections",
        help=f"registry key {sorted(REGISTRY)} or 'module:ClassName'",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["dev", "test"],
        help="dev is for tuning and reading by eye; test is what gets published",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the result JSON here (default: leaderboard/runs/<extractor>.json)",
    )
    args = parser.parse_args(argv)

    extractor = load_extractor(args.extractor)
    corpus = Corpus.load()

    print(
        f"Running {args.extractor} on the {args.split} split "
        f"of {len(corpus)} documents",
        file=sys.stderr,
    )
    result = run(extractor, corpus, split=args.split)

    out = args.out or Path("leaderboard/runs") / f"{result.extractor}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2) + "\n")

    t = result.totals
    print(
        f"\n{result.extractor}: {t['n_documents']} documents  "
        f"P={t['precision']:.2f} R={t['recall']:.2f} F1={t['f1']:.2f}  "
        f"grounding={t['grounding_rate']:.0%}\n"
        f"-> {out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
