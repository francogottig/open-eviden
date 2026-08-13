# Disclaimer

**This software is research tooling. It is not a medical device, and nothing it
produces is medical advice.**

Read this before using anything in this repository, and before citing anything it
outputs.

## What this repository does

It reads published clinical literature and extracts structured statements about
studies — how many people were randomised, how long they were followed, what the
primary outcome was — each one tied to a verbatim quotation from the source and
that source's DOI. It then scores how well an extraction method does that,
against a small hand-checked reference set.

That is a *literature-processing* task. Its output is a table about papers.

## What it does not do

- **It does not produce patient-facing output.** There is no interface here for a
  patient, and nothing in it should be put in front of one.
- **It does not diagnose, treat, or recommend.** No function in this repository
  takes a patient's characteristics and returns a suggested action, and none will
  be accepted. A pull request that adds one will be closed.
- **It does not tell you whether a treatment works.** Extracting "the primary
  outcome was a 4.2-point reduction on the HAM-D" is a claim about what a paper
  says. Whether that paper is right, whether it generalises to anyone in
  particular, and whether the effect is clinically meaningful are separate
  questions this repository does not touch.
- **It is not validated for clinical use, and no validation is planned.** It has
  not been reviewed by any regulator, and it is not intended to support a
  regulatory submission.

## The extraction is imperfect, in ways that matter

Every method here — including the deterministic rule-based ones — misses
statements, and can attach a correct-looking value to the wrong quotation. The
grounding check in `open_eviden/synthesis/base.py` guarantees that a reported
value points at text that genuinely appears in the source document. It does not,
and cannot, guarantee that the pointed-at text actually supports the value.

Read the quote. That is what it is there for.

The reference set is small, hand-built, and drawn from open-access articles,
which are not a random sample of the literature. Scores on it are a signal about
relative method quality, not an estimate of accuracy on your corpus.

## If you are a clinician or a researcher using this

Treat every extracted row as a pointer back to the paper, not as a substitute for
reading it. The value of this tool is that it makes the pointer explicit and
checkable — the DOI resolves, the quote is verbatim, and you can go and look.

Any decision affecting a patient stays with a qualified clinician, exercising
their own judgement, working from the primary sources.

## Data

Nothing credentialed, restricted, or identifiable is committed to this
repository, and no such data should ever be added to it. Datasets that require a
data use agreement — MIMIC-IV among them — are referenced by identifier only and
fetched by you, under your own agreement, into a directory git does not track.
See [`open_eviden/sources/physionet.py`](open_eviden/sources/physionet.py).

If you believe something in this repository contains patient data, treat it as an
incident: open a security advisory rather than a public issue. See
[SECURITY.md](SECURITY.md).
