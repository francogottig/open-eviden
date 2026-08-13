"""Europe PMC ingestion — search, full text, and the licence filter.

Europe PMC exposes the Open Access subset as JATS XML, which is the reason this
is the primary source: the publisher's own markup tells us where the Methods
section ends, and that structure is what `synthesis/sections.py` needs. Scraping
it back out of flowing prose would work most of the time and fail exactly on the
articles with unusual headings, which are the ones worth having.

API docs: https://europepmc.org/RestfulWebService

The licence filter is the part to read before changing anything. "Open access"
is not one licence: the OA subset contains CC BY, CC BY-NC and CC BY-NC-ND
articles, and only the first is ours to redistribute. `fetch` requests CC BY at
the query level *and* checks the returned licence field per record, because a
filter applied only in the query is one API change away from silently
committing articles we have no right to publish.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from .. import records
from ..records import normalize_whitespace

SEARCH_ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT_ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

#: The service caps a page at 1,000 results and paginates by cursor.
MAX_PAGE_SIZE = 1_000

#: Polite pacing between requests. Europe PMC publishes no hard rate limit;
#: this keeps a corpus build from looking like an attack on a free service that
#: a great deal of research infrastructure depends on.
REQUEST_DELAY_SECONDS = 0.34

#: Licence strings Europe PMC returns that we are free to redistribute. Compared
#: after lowercasing and stripping. Anything else is dropped at ingest with a
#: reason, not filtered silently — see `fetch`.
ALLOWED_LICENSES = frozenset({"cc by", "cc-by", "cc0"})

#: JATS elements that are not article prose. Dropping them at parse time keeps
#: reference-list author names and table cell fragments out of the text an
#: extractor searches, which otherwise produce quotations that ground correctly
#: and mean nothing.
_SKIP_TAGS = frozenset(
    {
        "ref-list",
        "back",
        "fn-group",
        "table-wrap",
        "table",
        "fig",
        "graphic",
        "media",
        "supplementary-material",
        "author-notes",
        "history",
        "permissions",
        "funding-group",
        "contrib-group",
        "aff",
        "journal-meta",
    }
)


def _element_text(element: ET.Element) -> str:
    """All descendant text of `element`, minus the elements we skip.

    `itertext` would be shorter, but it has no way to exclude a subtree, and
    the subtrees excluded here are exactly the ones that pollute the corpus.
    """
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        if node.tag in _SKIP_TAGS:
            return
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(element)
    return normalize_whitespace(" ".join(parts))


def _section_title(section: ET.Element) -> str:
    title = section.find("title")
    return normalize_whitespace(title.text or "") if title is not None and title.text else ""


def parse_fulltext(xml_text: str) -> tuple[str, list[dict]]:
    """JATS XML to (text, sections).

    Section offsets are computed against the *already normalized* text, and each
    piece is normalized before being joined with a single space, so that the
    normalization `records.conform_documents` applies afterwards is a no-op and
    the offsets stay valid. Normalizing once at the end instead would shift
    every offset by the amount of whitespace collapsed before it — a bug that
    only shows up as slightly-wrong section boundaries, which is the kind that
    survives review.
    """
    root = ET.fromstring(xml_text)

    pieces: list[tuple[str, str]] = []  # (title, text)

    abstract = root.find(".//front//abstract")
    if abstract is not None:
        text = _element_text(abstract)
        if text:
            pieces.append(("Abstract", text))

    body = root.find("body")
    if body is not None:
        sections = body.findall("sec")
        if sections:
            for section in sections:
                title = _section_title(section) or "Untitled section"
                text = _element_text(section)
                # Strip the heading off the front of its own section text, so a
                # quotation is never half-heading.
                if title != "Untitled section" and text.startswith(title):
                    text = text[len(title) :].strip()
                if text:
                    pieces.append((title, text))
        else:
            # Some publishers deposit the body as one untitled block. Keeping it
            # as a single section is honest: `sections.py` then falls back to
            # searching the whole article and records that it did.
            text = _element_text(body)
            if text:
                pieces.append(("Body", text))

    assembled: list[str] = []
    section_records: list[dict] = []
    cursor = 0
    for title, text in pieces:
        if cursor:
            cursor += 1  # the joining space
        section_records.append({"title": title, "start": cursor, "end": cursor + len(text)})
        assembled.append(text)
        cursor += len(text)

    return " ".join(assembled), section_records


def _record_from_result(result: dict, text: str, sections: list[dict]) -> dict:
    import json

    return {
        "document_id": f"pmc:{result.get('pmcid')}",
        "doi": result.get("doi") or "",
        "pmcid": result.get("pmcid") or "",
        "pmid": result.get("pmid") or "",
        "title": result.get("title") or "",
        "journal": (result.get("journalInfo") or {}).get("journal", {}).get("title")
        or result.get("journalTitle")
        or "",
        "year": result.get("pubYear"),
        "license": result.get("license") or "",
        "source": "europepmc",
        "text": text,
        "sections": json.dumps(sections),
    }


#: Publication type the corpus is restricted to.
#:
#: This is a scientific decision, not a plumbing detail, so it is a named
#: constant rather than a string buried in a query.
#:
#: A free-text search for "randomised controlled trial" returns mostly articles
#: *about* trials — systematic reviews, meta-analyses, published protocols,
#: narrative reviews. Those are not the population this benchmark is for. A
#: review reports how many trials it included and how many participants they
#: enrolled between them; asking an extractor for "the number randomised" there
#: has no correct answer, and a reference annotation would be inventing one.
#: MEDLINE's indexed publication type distinguishes a primary trial report from
#: an article discussing trials, and that is the distinction that matters.
#:
#: Published protocols are excluded by the same reasoning: they describe a trial
#: that has not happened yet, so their participant count is a target rather than
#: a result. They are worth benchmarking on eventually — the fields differ in
#: meaning rather than in kind — and doing that properly means a separate
#: protocol group, not quietly mixing them in here.
PUBLICATION_TYPE = "Randomized Controlled Trial"


def search(
    query: str,
    *,
    page_size: int = 25,
    publication_type: str | None = PUBLICATION_TYPE,
    timeout: int = 60,
    session: requests.Session | None = None,
) -> list[dict]:
    """Search the OA subset. Returns raw result dicts, unfiltered."""
    session = session or requests.Session()
    session.headers["User-Agent"] = "open-eviden (+https://github.com/nextbaires/open-eviden)"

    # The licence term must be double-quoted. Europe PMC accepts a
    # single-quoted value without complaint and matches nothing, so the wrong
    # quoting looks like "no open-access trials exist" rather than a syntax error.
    terms = [f"({query})", "OPEN_ACCESS:y", "IN_EPMC:y", 'LICENSE:"cc by"']
    if publication_type:
        terms.append(f'PUB_TYPE:"{publication_type}"')

    response = session.get(
        SEARCH_ENDPOINT,
        params={
            "query": " AND ".join(terms),
            "format": "json",
            "resultType": "core",
            "pageSize": min(page_size, MAX_PAGE_SIZE),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("resultList", {}).get("result", [])


def fetch_fulltext(
    pmcid: str, *, timeout: int = 60, session: requests.Session | None = None
) -> str | None:
    """Fetch one article's JATS XML. None when the full text is not available.

    A record can appear in the OA search and still have no deposited full text —
    the subset is defined by licence, not by what the publisher actually sent.
    Returning None rather than raising keeps one missing article from ending a
    corpus build.
    """
    session = session or requests.Session()
    response = session.get(FULLTEXT_ENDPOINT.format(pmcid=pmcid), timeout=timeout)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.text


def fetch(
    query: str,
    *,
    limit: int = 25,
    timeout: int = 60,
    verbose: bool = True,
) -> pd.DataFrame:
    """Search, fetch full text, and conform to the canonical document schema.

    Licence is checked here per record even though the query already filters on
    it. Two filters for one rule looks redundant until the day the query syntax
    changes meaning and every article in the corpus becomes one we are not
    allowed to publish.
    """
    session = requests.Session()
    session.headers["User-Agent"] = "open-eviden (+https://github.com/nextbaires/open-eviden)"

    results = search(query, page_size=limit, timeout=timeout, session=session)
    rows: list[dict] = []

    for result in results:
        pmcid = result.get("pmcid")
        license_name = str(result.get("license") or "").lower().strip()

        if not pmcid:
            continue
        if license_name not in ALLOWED_LICENSES:
            if verbose:
                print(f"  skip {pmcid}: licence {license_name!r} is not redistributable here")
            continue

        xml_text = fetch_fulltext(pmcid, timeout=timeout, session=session)
        time.sleep(REQUEST_DELAY_SECONDS)
        if not xml_text:
            if verbose:
                print(f"  skip {pmcid}: no full text deposited")
            continue

        try:
            text, sections = parse_fulltext(xml_text)
        except ET.ParseError as exc:
            if verbose:
                print(f"  skip {pmcid}: unparseable XML ({exc})")
            continue

        if not text:
            continue

        rows.append(_record_from_result(result, text, sections))
        if verbose:
            print(f"  {pmcid}: {len(text):,} chars, {len(sections)} sections")

    if not rows:
        return records.empty_documents()

    return records.deduplicate_documents(records.conform_documents(pd.DataFrame(rows)))
