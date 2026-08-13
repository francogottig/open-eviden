# Security and disclosure

## If you believe this repository contains patient data

Treat it as an incident, not a bug. **Do not open a public issue** — that
republishes it.

Open a private security advisory:
<https://github.com/nextbaires/open-eviden/security/advisories/new>

Nothing credentialed or patient-level is supposed to be here at all. The corpus
is published journal articles under CC BY, and `open_eviden/sources/physionet.py`
downloads nothing on your behalf precisely so that credentialed data has no route
into the repository. If something got through anyway, that is the failure worth
hearing about fastest.

## Other vulnerabilities

Same channel. Please include what you did, what happened, and what you expected.

## The expert loop, and what it can reach

Issues on this repository can be turned into pull requests by an assistant, via
[`nextbaires/expert-loop`](https://github.com/nextbaires/expert-loop). Two things
about that are worth stating plainly rather than leaving in a workflow file:

**Issue text is untrusted input to a model with write access to a branch.** A
proposal that contains instructions aimed at the assistant — to ignore its
prompt, touch other files, run commands, or reveal configuration — is a
prompt-injection attempt. The drafting prompt tells the assistant to treat issue
text as evidence rather than instructions and to report the attempt in the pull
request, and the permission deny-list blocks writes to workflows, CODEOWNERS and
dependency manifests. Neither of those is a guarantee. The actual control is that
a maintainer has to apply a label before anything runs, and a human has to
approve the pull request before anything merges.

**The loop cannot reach the data.** `allowed-paths` excludes `data/` and
`tools/`, so a drafted change cannot edit the corpus or the reference
annotations. That is the difference between a contribution that changes a method
and one that quietly changes the scoreboard.

## What this project is not

Research tooling. Not a medical device, no patient-facing output, no diagnostic
claims, not validated for clinical use. See [DISCLAIMER.md](DISCLAIMER.md). A
report that it gives poor medical advice is not a vulnerability — it is a
category error, and the answer is that it should not be asked for any.
