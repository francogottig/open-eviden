"""PubMed E-utilities: metadata only.

Europe PMC is the primary source because it deposits structured full text.
PubMed is here for the case Europe PMC cannot serve: confirming that an article
exists, and recovering its DOI and publication type when a record is thin.

It returns metadata, never full text — PubMed does not hold full text, and an
abstract passed off as an article would quietly change what the benchmark
measures.

E-utilities terms: https://www.ncbi.nlm.nih.gov/books/NBK25497/
Without an API key the service allows 3 requests/second; with one, 10. Set
`NCBI_API_KEY` if you have one and are fetching more than a handful.
"""

from __future__ import annotations

import os
import time

import requests

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

#: 3 requests/second unkeyed. Staying a hair under it rather than at it, because
#: the penalty for exceeding it is a block, not a 429.
REQUEST_DELAY_SECONDS = 0.35
KEYED_DELAY_SECONDS = 0.11


def _params(extra: dict) -> dict:
    params = {"db": "pubmed", "retmode": "json", "tool": "open-eviden", **extra}
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


def _delay() -> float:
    return KEYED_DELAY_SECONDS if os.environ.get("NCBI_API_KEY") else REQUEST_DELAY_SECONDS


def search(query: str, *, limit: int = 20, timeout: int = 30) -> list[str]:
    """Return PMIDs matching `query`."""
    response = requests.get(
        ESEARCH, params=_params({"term": query, "retmax": limit}), timeout=timeout
    )
    response.raise_for_status()
    time.sleep(_delay())
    return response.json().get("esearchresult", {}).get("idlist", [])


def summarize(pmids: list[str], *, timeout: int = 30) -> list[dict]:
    """Fetch record summaries for a list of PMIDs.

    Returns the flat fields this repository cares about — identifiers, title,
    journal, year, publication types — rather than E-utilities' full nested
    structure, so that a caller never has to learn its shape to answer "is this
    a randomised trial and what is its DOI".
    """
    if not pmids:
        return []

    response = requests.get(
        ESUMMARY, params=_params({"id": ",".join(pmids)}), timeout=timeout
    )
    response.raise_for_status()
    time.sleep(_delay())

    result = response.json().get("result", {})
    records = []
    for pmid in result.get("uids", []):
        entry = result.get(pmid, {})
        doi = next(
            (a.get("value") for a in entry.get("articleids", []) if a.get("idtype") == "doi"),
            "",
        )
        records.append(
            {
                "pmid": pmid,
                "doi": doi,
                "title": entry.get("title", ""),
                "journal": entry.get("fulljournalname") or entry.get("source", ""),
                # `pubdate` is free text — "2023 Mar 15", "2023 Spring", "2023".
                # Only the leading year is reliably parseable, so only the
                # leading year is taken.
                "year": (entry.get("pubdate") or "")[:4],
                "publication_types": entry.get("pubtype", []),
            }
        )
    return records
