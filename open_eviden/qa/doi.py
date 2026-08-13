"""Does every DOI in the corpus still resolve.

This is the check the project promises: a claim in this repository can be
followed back to a source that exists. It needs the network, which is why it is
its own module, its own CI job, and its own failure — the offline checks in
`provenance.py` must never fail because a publisher's server was down.

It runs on every push and on a schedule, because link rot is not a one-off event
and a DOI that resolved when it was committed can stop resolving a year later
without anything in the repository changing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

RESOLVER = "https://doi.org/"

#: Polite pacing. doi.org publishes no rate limit; this keeps a full corpus
#: check from looking like an attack on a service everyone depends on.
REQUEST_DELAY_SECONDS = 0.2

#: A DOI resolves to a publisher's page, and publishers do all sorts of things:
#: 302 to a landing page, 403 a bot, 503 under load. Only a 404 from the
#: resolver itself is evidence the identifier is wrong — everything else is
#: evidence about the publisher's server, which is not what this checks.
UNRESOLVED_STATUS = 404


@dataclass(frozen=True)
class Resolution:
    doi: str
    document_id: str
    status: int | None
    error: str | None = None

    @property
    def resolved(self) -> bool:
        """True unless the resolver said this identifier does not exist.

        A network error is `resolved = False` only in the sense that we could
        not tell; `main` distinguishes the two, because failing CI on a
        transient timeout is how a check gets disabled.
        """
        return self.status is not None and self.status != UNRESOLVED_STATUS


def resolve(doi: str, document_id: str, *, session=None, timeout: int = 20) -> Resolution:
    """HEAD the DOI resolver, following redirects to the publisher."""
    session = session or requests.Session()
    url = RESOLVER + doi
    try:
        response = session.head(url, allow_redirects=True, timeout=timeout)
        # Some publishers reject HEAD outright but serve GET. Retrying with GET
        # on a 405 avoids reporting a live DOI as dead because of a server's
        # method handling.
        if response.status_code == 405:
            response = session.get(url, allow_redirects=True, timeout=timeout, stream=True)
            response.close()
        return Resolution(doi=doi, document_id=document_id, status=response.status_code)
    except requests.RequestException as exc:
        return Resolution(doi=doi, document_id=document_id, status=None, error=str(exc))


def resolve_all(pairs, *, verbose: bool = True) -> list[Resolution]:
    """Resolve every (document_id, doi) pair, pacing between requests."""
    session = requests.Session()
    session.headers["User-Agent"] = "open-eviden (+https://github.com/nextbaires/open-eviden)"

    results = []
    for document_id, doi in pairs:
        result = resolve(str(doi), str(document_id), session=session)
        results.append(result)
        if verbose:
            mark = "ok" if result.resolved else "FAIL"
            print(f"  {mark:<4} {result.status or result.error}  {doi}")
        time.sleep(REQUEST_DELAY_SECONDS)
    return results
