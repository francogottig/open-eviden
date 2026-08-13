# Repository-specific drafting instructions

Read this before implementing a proposal here. It takes precedence over the
generic expert-loop prompt.

## The two rules that outrank everything

**Never let a value count without a verbatim quotation.** Every claim an
extractor returns passes through `synthesis/base.py:ground`, which searches the
document for the quotation and discards the claim if it is not there. Do not add
a fuzzy-match mode, a similarity threshold, a "close enough" fallback, or a flag
that skips the check for one extractor. If a proposal seems to need one, what it
actually needs is a better quotation, and `tests/test_grounding.py` is where that
gets enforced. Any change that makes those tests pass by weakening them is wrong.

**Never add patient-facing output or a diagnostic claim.** This repository
extracts what papers say about how trials were conducted. It does not take a
patient's characteristics and return a suggested action, it does not rank
treatments, and it does not tell anyone what to do. A proposal asking for any of
those is out of scope no matter how it is framed — say so on the issue and stop.
See `DISCLAIMER.md`, which is not decoration.

## Never commit data that is not ours to publish

The corpus is CC BY only, filtered at ingest in `sources/europepmc.py` and
enforced by `DOCUMENT_CONTRACT`. "Open access" is not one licence: the Europe PMC
OA subset also contains CC BY-NC and CC BY-NC-ND articles, which are open to read
and not open to republish. Do not widen `REDISTRIBUTABLE_LICENSES`.

Nothing credentialed goes in this repository under any circumstances — no
MIMIC-IV, no PhysioNet restricted dataset, no clinical extract, not even the open
demo. `sources/physionet.py` resolves identifiers and downloads nothing, and that
is deliberate rather than unfinished.

## Scientific and clinical claims need sources

Any assertion about medicine, methodology, or reporting practice — as opposed to
an assertion about this code — carries a DOI or URL in the comment or docstring
where it appears. "Assessor-only masking is single-blinding under CONSORT" needs a
citation. "This loop iterates over sections" does not.

If the proposer gave firsthand experience rather than a paper, attribute it to
them explicitly: `# Per @username (#42): the cluster/individual mix-up is the
most common extraction error in second-reviewer checks.` That is legitimate
evidence and should be visible as what it is.

## Extractors

A new extractor implements the `Extractor` protocol in `synthesis/base.py` and
nothing more: `name`, `extract(document) -> Sequence[Claim]`, `describe() -> dict`.
Register it in `REGISTRY` in `bench/run.py`.

Return claims whose `quote` is copied out of `document.text` — slice it, do not
rebuild it. A rebuilt string is where grounding failures come from, and
`sentence_around` in `rules.py` exists so you do not have to write that yourself.

An extractor that needs the network or an API key must be optional, must not be
imported at module scope by anything the tests touch, and must not appear in the
committed leaderboard. Everything CI checks runs offline from a clone.

## The reference set is not yours to edit

`data/sample/gold.sample.parquet` is the measuring instrument. Changing it moves
every published score without changing a single line of method, and doing it
correctly means reading the articles. It is outside `allowed-paths` for this
reason.

If a proposal is *about* the reference set — an annotation you believe is wrong,
a field that should be defined differently, an article that should not be in the
corpus — that is a genuinely valuable contribution and it needs a human. Say so
on the issue, quote the specific annotation, and stop.

## Tests

A change to extraction behaviour needs a test on the behaviour, not on the
plumbing. A test asserting that `extract` returns a list does not demonstrate
that a new pattern is right; a test asserting that a spelled-out participant
count is read, or that a Discussion sentence is not, does.

Do not assert exact leaderboard scores in tests. That freezes the baselines
against improvement, which is the opposite of the point. `tests/test_extractors.py`
asserts properties — everything grounds, quotations are spans, the discussion is
never read — and those hold no matter how good the extractor gets.

## What not to touch

`bench/split.py`'s hashing, `bench/metrics.py`, and the field definitions in
`records.py` define the protocol every published result depends on. Changing them
silently invalidates the leaderboard. If a proposal genuinely requires a protocol
change, say so on the issue and let a maintainer decide — do not implement it.
