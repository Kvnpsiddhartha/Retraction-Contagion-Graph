"""
data/external_apis.py — Prompt 02b: External API clients (Crossref +
Semantic Scholar).

Two thin, well-behaved HTTP clients:
    - fetch_crossref_metadata            : DOI -> Crossref work metadata
    - fetch_semantic_scholar_citations   : DOI -> list of citing-paper records
    - parse_citations_to_models          : pure transform, raw S2 citations
                                            -> (CitingPaper, CitationContext)

Only `shared/` is imported — this module knows nothing about
`data/retraction_watch.py`, `memory/`, `pipeline/`, or `api/`.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import date, datetime
from typing import Any, Optional

import requests
from pydantic import ValidationError

from shared.config import settings
from shared.exceptions import ExternalAPIError
from shared.schemas import CitationContext, CitationIntent, CitingPaper, normalize_doi

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Module-level session (connection reuse across all calls made by this
# process — both clients share it since they hit different hosts anyway).
# --------------------------------------------------------------------------

_session = requests.Session()

_S2_CITATION_FIELDS = "title,abstract,publicationDate,contexts,intents,externalIds"
_S2_PAGE_SIZE_CAP = 100  # keep individual pages small & fast for the demo

_INTENT_MAP: dict[str, CitationIntent] = {
    "background": CitationIntent.BACKGROUND,
    "methodology": CitationIntent.METHOD,
    "method": CitationIntent.METHOD,
    "result": CitationIntent.RESULT,
    "results": CitationIntent.RESULT,
}


# --------------------------------------------------------------------------
# Shared retry/backoff helper
# --------------------------------------------------------------------------


def _sleep_with_backoff(attempt: int) -> None:
    """Exponential backoff with jitter, attempt is 0-indexed."""
    base = settings.retry_backoff_base_seconds * (2**attempt)
    jitter = random.uniform(0, settings.retry_backoff_base_seconds)
    time.sleep(base + jitter)


def _request_with_retry(
    method: str,
    url: str,
    *,
    source: str,
    treat_404_as_terminal: bool = True,
    **kwargs: Any,
) -> requests.Response:
    """Issue an HTTP request with bounded retries and exponential backoff.

    Retry policy:
        - Network-level failures (timeout, connection error) -> retry.
        - HTTP 429 (rate limited) -> retry with backoff (this is the
          expected steady-state behavior for both Crossref and Semantic
          Scholar under light abuse, not an exceptional case).
        - Other non-2xx, non-404 statuses (5xx etc.) -> retry.
        - HTTP 404 -> returned immediately to the caller (not retried,
          not an error) when `treat_404_as_terminal` is True, since a 404
          here means "resource does not exist," which retrying cannot fix.
          The caller decides how to handle it (empty dict / empty list).

    Raises:
        ExternalAPIError: once `settings.max_retries` attempts have all
            failed.
    """
    last_exc: Exception | None = None
    last_status: Optional[int] = None

    for attempt in range(settings.max_retries + 1):
        try:
            response = _session.request(
                method,
                url,
                timeout=settings.request_timeout_seconds,
                **kwargs,
            )
        except (requests.Timeout, requests.ConnectionError, requests.RequestException) as exc:
            last_exc = exc
            logger.warning(
                "[%s] request to %s failed on attempt %d/%d: %s",
                source,
                url,
                attempt + 1,
                settings.max_retries + 1,
                exc,
            )
            if attempt < settings.max_retries:
                _sleep_with_backoff(attempt)
                continue
            raise ExternalAPIError(
                f"Request to {url} failed after {settings.max_retries + 1} attempts",
                cause=exc,
                source=source,
            ) from exc

        if response.status_code == 404 and treat_404_as_terminal:
            return response

        if response.ok:
            return response

        last_status = response.status_code
        if response.status_code == 429:
            logger.warning(
                "[%s] rate limited (429) on attempt %d/%d for %s; backing off.",
                source,
                attempt + 1,
                settings.max_retries + 1,
                url,
            )
        else:
            logger.warning(
                "[%s] non-2xx status %d on attempt %d/%d for %s.",
                source,
                response.status_code,
                attempt + 1,
                settings.max_retries + 1,
                url,
            )

        if attempt < settings.max_retries:
            _sleep_with_backoff(attempt)
            continue

        raise ExternalAPIError(
            f"Request to {url} failed after {settings.max_retries + 1} attempts",
            source=source,
            status_code=last_status,
        )

    # Unreachable in practice (loop always returns or raises), but keeps
    # type-checkers happy and guards against future edits to the loop above.
    raise ExternalAPIError(
        f"Request to {url} exhausted retries without a definitive outcome",
        source=source,
        status_code=last_status,
    )


# --------------------------------------------------------------------------
# Crossref
# --------------------------------------------------------------------------


def fetch_crossref_metadata(doi: str) -> dict:
    """Fetch Crossref metadata for a DOI.

    Args:
        doi: a DOI, possibly prefixed / mixed case (normalized internally).

    Returns:
        The raw `message` dict from Crossref's JSON response, or `{}` if
        the DOI is not found in Crossref (HTTP 404) — this is treated as a
        normal, expected outcome (not every DOI resolvable elsewhere is in
        Crossref), not an error.

    Raises:
        ExternalAPIError: on network failure, timeout, or a non-2xx,
            non-404 status after exhausting retries.
    """
    norm_doi = normalize_doi(doi)
    url = f"{settings.crossref_base_url}/works/{norm_doi}"
    params = {"mailto": settings.crossref_mailto}

    response = _request_with_retry("GET", url, source="crossref", params=params)

    if response.status_code == 404:
        logger.warning("Crossref: DOI %s not found (404); returning empty metadata.", norm_doi)
        return {}

    try:
        payload = response.json()
    except ValueError as exc:
        raise ExternalAPIError(
            f"Crossref returned non-JSON response for DOI {norm_doi}",
            cause=exc,
            source="crossref",
            status_code=response.status_code,
        ) from exc

    message = payload.get("message")
    if not isinstance(message, dict):
        logger.warning(
            "Crossref: response for DOI %s had no usable 'message' field; returning empty dict.",
            norm_doi,
        )
        return {}

    return message


# --------------------------------------------------------------------------
# Semantic Scholar
# --------------------------------------------------------------------------


def fetch_semantic_scholar_citations(doi: str, limit: int = 100) -> list[dict]:
    """Fetch papers that cite `doi`, with citation context/intent where available.

    Paginates through Semantic Scholar's `/paper/DOI:{doi}/citations`
    endpoint until either the API reports no further pages (no `next`
    offset) or `limit` results have been collected, whichever comes first.

    Args:
        doi: the cited paper's DOI (normalized internally).
        limit: maximum number of citation records to return. Must be > 0;
            a non-positive value is treated as "no results requested" and
            short-circuits to `[]` without any HTTP call.

    Returns:
        The raw list of citation objects as given by the API (each
        containing `contexts`, `intents`, and the citing paper's fields).
        Returns `[]` if the paper itself is not found in the Semantic
        Scholar graph (HTTP 404) — logged as a warning, not raised.

    Raises:
        ExternalAPIError: on network failure, timeout, or a non-2xx,
            non-404, non-429-exhausted status, after exhausting retries.
            The error message includes how many pages were successfully
            fetched before the failure, since partial results are
            deliberately NOT returned in place of raising (silently
            treating partial data as complete would violate the "no
            silent failures" invariant).
    """
    if limit <= 0:
        logger.warning("fetch_semantic_scholar_citations called with non-positive limit=%d; returning [].", limit)
        return []

    norm_doi = normalize_doi(doi)
    url = f"{settings.semantic_scholar_base_url}/paper/DOI:{norm_doi}/citations"

    results: list[dict] = []
    offset = 0
    pages_fetched = 0

    while len(results) < limit:
        page_size = min(_S2_PAGE_SIZE_CAP, limit - len(results))
        params = {
            "fields": _S2_CITATION_FIELDS,
            "limit": page_size,
            "offset": offset,
        }

        try:
            response = _request_with_retry(
                "GET",
                url,
                source="semantic_scholar",
                params=params,
            )
        except ExternalAPIError as exc:
            raise ExternalAPIError(
                f"Semantic Scholar citation fetch for DOI {norm_doi} failed after "
                f"{pages_fetched} page(s) successfully fetched ({len(results)} records so far)",
                cause=exc.cause,
                source="semantic_scholar",
                status_code=exc.status_code,
            ) from exc

        if response.status_code == 404:
            if pages_fetched == 0:
                logger.warning(
                    "Semantic Scholar: DOI %s not found in citation graph (404); returning [].",
                    norm_doi,
                )
                return []
            logger.warning(
                "Semantic Scholar: DOI %s returned 404 mid-pagination after %d page(s); "
                "stopping and returning results collected so far.",
                norm_doi,
                pages_fetched,
            )
            break

        try:
            payload = response.json()
        except ValueError as exc:
            raise ExternalAPIError(
                f"Semantic Scholar returned non-JSON response for DOI {norm_doi} "
                f"after {pages_fetched} page(s) successfully fetched",
                cause=exc,
                source="semantic_scholar",
                status_code=response.status_code,
            ) from exc

        pages_fetched += 1
        page_data = payload.get("data") or []
        results.extend(page_data)

        next_offset = payload.get("next")
        if next_offset is None or not page_data:
            break
        offset = next_offset

    logger.info(
        "Semantic Scholar: fetched %d citation record(s) for DOI %s across %d page(s).",
        len(results),
        norm_doi,
        pages_fetched,
    )

    return results[:limit]


# --------------------------------------------------------------------------
# Pure transform: raw Semantic Scholar citations -> typed models
# --------------------------------------------------------------------------


def _extract_paper_object(raw_citation: dict) -> dict:
    """Return the sub-object that actually carries the citing paper's
    fields (title/abstract/publicationDate/externalIds).

    Real Semantic Scholar responses nest these under a `citingPaper` key;
    some simplified/mocked payloads (and our own test fixtures) may place
    them at the top level instead. Support both shapes rather than
    assuming one, since the wire format is outside our control.
    """
    nested = raw_citation.get("citingPaper")
    if isinstance(nested, dict):
        return nested
    return raw_citation


def _first_non_empty_context(contexts: Any) -> Optional[str]:
    if not contexts:
        return None
    for entry in contexts:
        if isinstance(entry, str) and entry.strip():
            return entry.strip()
    return None


def _map_intent(intents: Any) -> CitationIntent:
    if not intents:
        return CitationIntent.UNKNOWN
    first = intents[0]
    if not isinstance(first, str):
        return CitationIntent.UNKNOWN
    return _INTENT_MAP.get(first.strip().lower(), CitationIntent.UNKNOWN)


def _parse_publication_date(raw: Any) -> Optional[date]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = raw.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue
    return None


def parse_citations_to_models(
    cited_doi: str,
    raw_citations: list[dict],
) -> tuple[list[CitingPaper], list[CitationContext]]:
    """Transform raw Semantic Scholar citation objects into typed models.

    Pure function — no I/O, no retries, no logging side effects beyond
    `logger` calls. Never raises on malformed input: any single citation
    record that can't be turned into valid models is skipped and counted,
    with a summary logged at the end (per architecture invariant: degrade,
    don't crash on one bad record).

    Args:
        cited_doi: the DOI all these citations point *to* (normalized
            internally).
        raw_citations: the raw list returned by
            `fetch_semantic_scholar_citations`.

    Returns:
        A `(citing_papers, citation_contexts)` tuple. Both lists are
        index-aligned with each other (not necessarily with
        `raw_citations`, since skipped records are simply omitted).
    """
    cited_norm = normalize_doi(cited_doi)

    citing_papers: list[CitingPaper] = []
    citation_contexts: list[CitationContext] = []
    n_skipped_no_doi = 0
    n_skipped_invalid = 0

    for raw in raw_citations:
        if not isinstance(raw, dict):
            n_skipped_invalid += 1
            continue

        paper_obj = _extract_paper_object(raw)
        external_ids = paper_obj.get("externalIds") or {}
        if not isinstance(external_ids, dict):
            external_ids = {}

        raw_citing_doi = external_ids.get("DOI") or external_ids.get("doi")
        if not raw_citing_doi:
            n_skipped_no_doi += 1
            logger.debug(
                "Skipping citation of %s with no DOI in externalIds (title=%r).",
                cited_norm,
                paper_obj.get("title"),
            )
            continue

        try:
            citing_norm = normalize_doi(raw_citing_doi)
        except ValueError:
            n_skipped_invalid += 1
            logger.warning(
                "Skipping citation of %s with unparsable DOI %r.", cited_norm, raw_citing_doi
            )
            continue

        if citing_norm == cited_norm:
            # A paper "citing itself" is not a meaningful edge for our
            # dependency graph (and would fail CitationContext's own
            # self-citation guard) — degrade by skipping, don't crash.
            n_skipped_invalid += 1
            logger.debug("Skipping self-referential citation for doi=%s.", cited_norm)
            continue

        title = (paper_obj.get("title") or "").strip() or "Untitled citing paper"
        abstract = paper_obj.get("abstract")
        pub_date = _parse_publication_date(paper_obj.get("publicationDate"))

        try:
            citing_paper = CitingPaper(
                doi=citing_norm,
                title=title,
                abstract=abstract,
                pub_date=pub_date,
            )
        except (ValidationError, ValueError) as exc:
            n_skipped_invalid += 1
            logger.warning("Skipping citing paper %s: failed schema validation: %s", citing_norm, exc)
            continue

        context_sentence = _first_non_empty_context(raw.get("contexts"))
        intent = _map_intent(raw.get("intents"))

        try:
            citation_context = CitationContext(
                citing_doi=citing_norm,
                cited_doi=cited_norm,
                context_sentence=context_sentence,
                intent=intent,
                source="semantic_scholar",
            )
        except (ValidationError, ValueError) as exc:
            n_skipped_invalid += 1
            logger.warning(
                "Skipping citation context %s -> %s: failed schema validation: %s",
                citing_norm,
                cited_norm,
                exc,
            )
            continue

        citing_papers.append(citing_paper)
        citation_contexts.append(citation_context)

    if n_skipped_no_doi or n_skipped_invalid:
        logger.info(
            "parse_citations_to_models(%s): kept %d, skipped %d (no DOI), skipped %d (invalid), "
            "out of %d raw record(s).",
            cited_norm,
            len(citing_papers),
            n_skipped_no_doi,
            n_skipped_invalid,
            len(raw_citations),
        )

    return citing_papers, citation_contexts
