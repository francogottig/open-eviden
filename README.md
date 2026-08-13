# open-eviden

**A benchmark for extracting study characteristics from clinical trial reports,
where a value only counts if the method can point at the sentence it came from —
and a way for clinicians and reviewers to contribute without writing code.**

> ⚠️ **Research tooling. Not a medical device.** No patient-facing output, no
> diagnostic claims, not validated for clinical use, and no patient data of any
> kind in this repository. Read [DISCLAIMER.md](DISCLAIMER.md) before using or
> citing anything here.

Everything runs offline from a committed sample of real CC BY articles, so a
clone reproduces every published number on a laptop with no API key and no
downloads.

---

## What this is honest about

**The reference set is small and one person built it.** Thirteen articles, 64
hand-made annotations, a single annotator who is a data engineer rather than a
clinician. That is enough to separate a working method from a broken one and not
enough to rank two good methods with any confidence. A difference of a few points
of F1 on ten test documents is noise. **Expanding and double-annotating the
reference set is the single most valuable contribution this repository could
receive**, and it is the one thing the automated contribution loop is deliberately
not allowed to touch.

**Some annotations are judgement calls, and they are not marked.** One article
says assessor blinding "could not be guaranteed"; it is recorded as `open-label`,
which is a reading rather than a quotation. Another states two named groups
without ever writing "parallel group"; it is recorded as `parallel-group`, which
is the inference a reviewer makes and a pattern-matcher has to be taught. A second
annotator would disagree somewhere, and there is currently no way to find out
where.

**Grounding proves the quotation is real. It does not prove the quotation
supports the value.** A method can quote a genuine sentence from the Methods and
attach a number that sentence does not contain, and the check will pass it. That
is why every extracted quote is carried through to the output: so a human can
read it. This is the limitation to keep in mind if you are tempted to trust a row
without opening the paper.

**Absence cannot be quoted, so it cannot be scored properly.** "The paper never
says whether outcome assessors were blinded" is a real and important finding in
risk-of-bias assessment, and it has no sentence attached to it. This benchmark
handles absence by treating it as an abstention — say nothing and you neither
gain nor lose. That is a workaround, not a solution, and finding a better one is
an open problem worth having.

**The corpus is open-access articles, which are not a random sample of the
literature.** They skew towards particular journals, funders, and fields. Scores
here are a signal about relative method quality, not an estimate of accuracy on
your corpus.

## The protocol

A method receives one article and returns claims. Each claim is a field, a value,
and a quotation. That is the entire interface.

| | |
|---|---|
| **Corpus** | 13 primary randomised-trial reports, Europe PMC open-access subset, CC BY only |
| **Fields** | participants randomised, design, blinding, follow-up, registration id, primary outcome |
| **Split** | 30% development / 70% test, assigned by a keyed hash of the document id |
| **Grounding** | A claim counts only if its quotation is verbatim in the article |
| **Scoring** | Precision, recall, F1 against the reference set; grounding rate reported separately |

**Grounding is structurally enforced, not requested.**
[`synthesis/base.py`](open_eviden/synthesis/base.py) searches the document for
every quotation an extractor returns and discards the claim if it is not there.
The failure this exists to close is specific: a language model has read a great
deal of clinical literature and can often say how many participants a well-known
trial randomised *from memory*, without reading the document in front of it. On a
benchmark that scores values alone, recalling and extracting earn identical
marks. Requiring a verbatim span collapses that — a remembered value has no
quotation, a fabricated quotation is not in the text, and either way the claim is
dropped before it can score.

The `Document` an extractor receives carries the article and nothing else: no
handle on the reference set, no handle on the corpus, no split label.
[`tests/test_grounding.py`](tests/test_grounding.py) asserts all of it, including
against extractors written specifically to cheat.

**The split is a hash, not a shuffle.** Adding an article never moves an existing
one between development and test, so scores from before an ingest stay comparable
to scores from after it. Nothing stops you tuning on the test split — you have
both files — so that is a protocol you keep rather than a wall you cannot climb,
and saying otherwise would be dressing up a convention as a guarantee.

## Quick start

```bash
git clone https://github.com/nextbaires/open-eviden
cd open-eviden
uv venv && uv pip install -e ".[dev]"

# Runs on the committed sample — no network, no API key.
python -m open_eviden.bench.run --extractor regex
python -m open_eviden.bench.run --extractor sections
python -m open_eviden.qa.cli          # offline provenance checks
pytest
```

To check that every source still resolves, and to build your own corpus:

```bash
python -m open_eviden.qa.cli --doi
python -m open_eviden.sources.cli --query "trial" --limit 40
```

## Submitting a method

Implement three members:

```python
class MyExtractor:
    name = "my-extractor"

    def extract(self, document):
        """Return claims. Each carries a value AND a verbatim quotation."""
        ...

    def describe(self):
        """Parameters and provenance, for the leaderboard record."""
        return {"extractor": "MyExtractor", "deterministic": True, ...}
```

Then add it to `REGISTRY` in
[`bench/run.py`](open_eviden/bench/run.py) or run it directly:

```bash
python -m open_eviden.bench.run --extractor mypackage.models:MyExtractor
```

Slice quotations out of `document.text` rather than rebuilding them — a rebuilt
string is where grounding failures come from.

## Leaderboard

Current standings on the test split — 10 articles, 51 reference claims.
Reproduce with:

```bash
pytest
python -m open_eviden.bench.run --extractor regex
python -m open_eviden.bench.run --extractor sections
python -m open_eviden.leaderboard
```

| Rank | Method | Precision | Recall | F1 | Grounded |
|---|---|---:|---:|---:|---:|
| 1 | `sections` | 0.79 | 0.73 | **0.76** | 100% |
| 2 | `regex` | 0.77 | 0.71 | 0.73 | 100% |

**Both baselines ground everything they return, and they are not being
virtuous — they have no mechanism for inventing anything.** Every value is read
out of a span they matched, and the quotation is that span's sentence sliced out
of the document. Any method that beats them on recall while grounding worse is
trading away something these give free.

The gap between them is the whole reason the second one exists. `regex` matches
anywhere in the article, including the Discussion, where authors write things like
"in the trial by Smith et al., 1,204 patients were randomised" — a real number,
correctly quoted, belonging to somebody else's study. `sections` runs the same
patterns over the Abstract, Methods and Results only. The patterns were never
wrong; the scope was.

Per field, on the test split:

| Field | `sections` F1 | What is going on |
|---|---:|---|
| `registration_id` | 1.00 | Registry identifiers have a fixed shape. This is what a solved field looks like. |
| `design` | 1.00 | Solved only because the baseline is allowed to infer "parallel-group" from two named arms, the way a reviewer does. |
| `follow_up_weeks` | 0.75 | Papers state duration in weeks, months, "at one year", and as an intervention length that is not always the follow-up length. |
| `blinding` | 0.67 | Better than it looks: four of six were being lost to a typographic hyphen, not to a hard judgement. |
| `n_randomised` | 0.56 | Stated in unbounded ways — "Twelve trained cyclists", "a total of 214 patients underwent…", "( n = 28 ) were randomised". |
| `primary_outcome` | 0.33 | Free text, scored by token overlap at an arbitrary threshold. Trust this number least. |

**`primary_outcome` is the weakest field and the least trustworthy number on the
page.** It is free text with no single right phrasing, so it is scored by token
overlap against a threshold picked by looking at the development set — 0.6,
which accepts "change in HAM-D score at 12 weeks" against "change from baseline
in HAM-D total score" and rejects "adverse events". Every number involving this
field inherits that arbitrariness. A better way to score a free-text field would
improve this benchmark more than a better extractor would.

**`blinding` is where the last real bug was.** Four of the six blinding
annotations were being missed because publishers emit U+2010 HYPHEN where a
reader sees a dash, and `double[- ]blind` does not match `double‐blind`. The two
render identically in a terminal. Finding it moved this field from 0.29 to 0.67
and the overall score from 0.72 to 0.76 — which is worth stating because it is
the kind of defect that looks like a hard domain problem right up until it turns
out to be a character class.

Full records — every document, every field, every rejected claim and why — are in
[`leaderboard/results.json`](leaderboard/results.json), rebuilt by CI on every
merge. CI fails if the committed results drift from what the code produces, so a
number nobody can reproduce cannot appear here.

The language-model extractor
([`synthesis/llm.py`](open_eviden/synthesis/llm.py)) is runnable and deliberately
absent from this table: it needs a key and a network, and it does not return the
same thing twice. Its entries, when submitted, are recorded with the model id and
marked `deterministic: false`, and CI never regenerates them.

### One API decision worth knowing about

The plan for this repository called for JSON-schema-constrained output *and* the
Claude API's document citations. Those two cannot be combined in one request —
`citations: {enabled: true}` together with `output_config.format` returns a 400.

This repository uses the schema and verifies quotations itself. Citations would
give offsets computed by the API, which is a stronger guarantee about the span —
but one available *only* to that extractor. The rule-based baselines cannot
produce API citations and neither can a submission built on another vendor's
model, and a benchmark whose central guarantee is stronger for one class of
entrant is not measuring the same thing across the leaderboard. `--citations`
runs the other configuration for comparison, because "the two mechanisms agree"
should be checkable rather than asserted.

## Contributing without writing code

**If you do systematic reviews, screen trials, or extract data by hand, you can
contribute — and your contribution is the kind this project most needs.**

Open an issue using the *Domain proposal* template. Describe what you know: what
the extractor gets wrong, what the reporting guidelines say, what you correct
when you second-check somebody else's extraction. You do not need to read the
code or know Python.

A maintainer reads it. If it is in scope, an assistant implements it against this
repository, writes a test, cites your sources, and opens a pull request with your
proposal quoted verbatim so you can check it says what you meant. You are credited
as the proposer. The domain judgement stays yours; if the implementation is wrong,
that is our bug to fix, not yours.

This runs on [`nextbaires/expert-loop`](https://github.com/nextbaires/expert-loop).
The loop cannot edit the corpus or the reference annotations — those need a human
who has read the articles.

Things that would genuinely help, if you are looking for somewhere to start:

- **More annotated articles, and a second annotator on the ones already here.**
  The biggest gap by a wide margin. Thirteen articles annotated by one person is
  a starting point, not a benchmark.
- **A better way to score free-text fields.** `primary_outcome` is scored by
  token overlap against a threshold chosen by eye. Reviewers compare outcomes by
  meaning, not by shared words, and that gap is the weakest link in the scoring.
- **Blinding, done properly.** What should be recorded when a paper describes
  masking partially, or only admits in the limitations that it failed?
- **Cluster and crossover trials.** The participant count means something
  different in each, and this repository currently pretends it does not.
- **A way to score absence.** How should "the paper never reports this" be
  evidenced and rewarded, given that there is no sentence to quote?
- **Fields that are missing.** Risk-of-bias domains, funding source, conflicts of
  interest — all extractable, all things reviewers record, none implemented.
- **Telling us the field definitions are wrong.** If `follow_up_weeks` is a
  badly-posed question because trials have several follow-up points, that is a
  finding, and it is one an expert can report in two paragraphs.

## Data

Articles come from the Europe PMC open-access subset, filtered to CC BY at ingest
because "open access" is not one licence — the subset also contains CC BY-NC and
CC BY-NC-ND articles that we are not free to redistribute here. Code is MIT;
annotations are CC BY 4.0; each article keeps its own licence. See
[LICENSE-DATA](LICENSE-DATA).

**No patient-level data is in this repository, at any access level.** Datasets
that require an agreement — MIMIC-IV among them — are referenced by identifier
only; [`sources/physionet.py`](open_eviden/sources/physionet.py) resolves the
identifier and the agreement you would be signing, and downloads nothing on your
behalf. It cannot even name a path outside the gitignored `data/restricted/`.

The provenance of the reference set is in
[`tools/build_reference_set.py`](tools/build_reference_set.py): the values were
decided by reading each article, and the quotations were located and verified
against the source rather than typed. Candidate sentences were surfaced by a
broad keyword search deliberately wider than the extractors' own patterns, so the
reference set is not limited to what the baselines can already find.

## License

MIT for code, CC BY 4.0 for annotations. See [LICENSE](LICENSE) and
[LICENSE-DATA](LICENSE-DATA).
