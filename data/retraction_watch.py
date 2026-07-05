"""
data/retraction_watch.py — Prompt 02a: Retraction Watch ingestion.

Downloads the Retraction Watch CSV (mirrored by Crossref on GitLab), parses
it defensively (the CSV's schema has changed before and is not guaranteed),
and selects a small, demo-ready seed set of retracted papers.

Only `shared/` is imported — this module knows nothing about
`data/external_apis.py`, `memory/`, `pipeline/`, or `api/`.
"""

from __future__ import annotations

import csv
import logging
import random
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests
from pydantic import ValidationError

from shared.config import settings
from shared.exceptions import DataValidationError, ExternalAPIError
from shared.schemas import RetractedPaper, RetractionReason, normalize_doi

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

_DEFAULT_CACHE_PATH = Path(".cache") / "retraction_watch.csv"

# Case-insensitive column aliases. Retraction Watch has renamed / reformatted
# columns before, so we resolve by a small alias list rather than assuming
# one exact header name. Keys are our canonical field names.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "doi": ("originalpaperdoi", "original paper doi", "doi", "original_paper_doi"),
    "title": ("title",),
    "journal": ("journal",),
    "original_pub_date": (
        "originalpaperdate",
        "original paper date",
        "original_paper_date",
    ),
    "retraction_date": ("retractiondate", "retraction date", "retraction_date"),
    "reason": ("reason", "reason(s)", "reasons"),
    "citation_count": (
        "citationcount",
        "citation count",
        "citation_count",
        "citations",
        "timescited",
        "times cited",
    ),
}

# Required canonical fields: without these, we cannot build a usable
# RetractedPaper at all, so we fail fast rather than guessing row-by-row.
_REQUIRED_CANONICAL_FIELDS = ("doi", "title")

# datetime.strptime formats seen in real-world Retraction Watch exports,
# tried in order. Falls back to None (never crashes a row) if none match.
_DATE_FORMATS = (
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%d-%m-%Y",
    "%B %d, %Y",
)

_FALLBACK_REASON_RAW = "Unspecified"

_REASON_KEYWORDS: tuple[tuple[str, RetractionReason], ...] = (
    ("fabricat", RetractionReason.FABRICATION),
    ("falsif", RetractionReason.FALSIFIED_DATA),
    ("plagiar", RetractionReason.PLAGIARISM),
    ("ethic", RetractionReason.ETHICAL_VIOLATION),
    ("error", RetractionReason.ERROR),
    ("mistake", RetractionReason.ERROR),
)


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def _sleep_with_backoff(attempt: int) -> None:
    """Exponential backoff with jitter, attempt is 0-indexed."""
    base = settings.retry_backoff_base_seconds * (2**attempt)
    jitter = random.uniform(0, settings.retry_backoff_base_seconds)
    time.sleep(base + jitter)


def fetch_retraction_watch_csv(
    dest_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> Path:
    """Download the Retraction Watch CSV to a local cache path.

    Args:
        dest_path: local destination. Defaults to ``./.cache/retraction_watch.csv``.
        force_refresh: if True, re-download even if `dest_path` already exists.

    Returns:
        The local path to the CSV (guaranteed to exist on success).

    Raises:
        ExternalAPIError: if all retry attempts fail (network error, timeout,
            or non-2xx HTTP status).
    """
    path = dest_path or _DEFAULT_CACHE_PATH

    if path.exists() and not force_refresh:
        logger.info("Retraction Watch CSV already cached at %s; skipping download.", path)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = settings.retraction_watch_csv_url

    last_exc: Exception | None = None
    for attempt in range(settings.max_retries + 1):
        try:
            logger.info(
                "Downloading Retraction Watch CSV (attempt %d/%d) from %s",
                attempt + 1,
                settings.max_retries + 1,
                url,
            )
            response = requests.get(
                url,
                timeout=settings.request_timeout_seconds,
                stream=True,
            )
            response.raise_for_status()

            tmp_path = path.with_suffix(path.suffix + ".part")
            with open(tmp_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        fh.write(chunk)
            tmp_path.replace(path)

            logger.info("Retraction Watch CSV downloaded successfully to %s", path)
            return path

        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            logger.warning(
                "Retraction Watch CSV download attempt %d/%d failed: %s",
                attempt + 1,
                settings.max_retries + 1,
                exc,
            )
            if attempt < settings.max_retries:
                _sleep_with_backoff(attempt)

    raise ExternalAPIError(
        "Failed to download Retraction Watch CSV after all retries",
        cause=last_exc,
        source="retraction_watch_csv",
    )


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------


def _resolve_columns(fieldnames: list[str]) -> dict[str, str]:
    """Map canonical field names -> actual CSV header names, case-insensitively.

    Raises:
        DataValidationError: if a required canonical field has no matching
            column in the CSV header.
    """
    normalized_to_actual = {name.strip().lower(): name for name in fieldnames}

    resolved: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized_to_actual:
                resolved[canonical] = normalized_to_actual[alias]
                break

    missing_required = [f for f in _REQUIRED_CANONICAL_FIELDS if f not in resolved]
    if missing_required:
        raise DataValidationError(
            "Retraction Watch CSV is missing required column(s) "
            f"{missing_required}; columns found: {fieldnames}",
            field=",".join(missing_required),
        )

    return resolved


def _parse_date(raw: str | None) -> Optional[date]:
    if raw is None:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue
    logger.debug("Could not parse date value %r with any known format.", raw)
    return None


def _parse_citation_count(raw: str | None) -> int:
    if raw is None:
        return 0
    candidate = raw.strip().replace(",", "")
    if not candidate:
        return 0
    try:
        value = int(float(candidate))
    except ValueError:
        return 0
    return max(value, 0)


def classify_reason(raw_reason: str) -> RetractionReason:
    """Best-effort keyword classification of a free-text retraction reason.

    Pure function, no I/O. Case-insensitive substring match against a small
    keyword table, checked in a fixed priority order (fabrication before
    error, etc.) so that a combined reason string like
    "Fabrication of data; Author error" resolves to the higher-severity
    category. Falls back to OTHER when nothing matches or the input is
    blank.
    """
    if not raw_reason or not raw_reason.strip():
        return RetractionReason.OTHER

    haystack = raw_reason.lower()
    for keyword, reason in _REASON_KEYWORDS:
        if keyword in haystack:
            return reason
    return RetractionReason.OTHER


def _read_rows_with_fallback_encoding(csv_path: Path) -> tuple[csv.DictReader, object]:
    """Open the CSV with a DictReader, retrying with a permissive encoding
    if the primary (utf-8-sig) decode fails partway through.

    Returns the DictReader and the open file handle (caller is responsible
    for closing it) so we can stream large files without loading everything
    into memory.
    """
    fh = open(csv_path, newline="", encoding="utf-8-sig", errors="strict")
    try:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise DataValidationError(
                f"Retraction Watch CSV at {csv_path} has no header row / is empty."
            )
        return reader, fh
    except UnicodeDecodeError:
        fh.close()
        logger.warning(
            "utf-8-sig decode failed for %s; retrying with latin-1 (lossless byte mapping).",
            csv_path,
        )
        fh = open(csv_path, newline="", encoding="latin-1")
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            fh.close()
            raise DataValidationError(
                f"Retraction Watch CSV at {csv_path} has no header row / is empty."
            )
        return reader, fh


def parse_retraction_watch_csv(csv_path: Path) -> list[RetractedPaper]:
    """Stream-parse the Retraction Watch CSV into `RetractedPaper` objects.

    Never raises on a single malformed row (missing DOI, unparsable date,
    schema-violating field combination, etc.) — such rows are logged and
    skipped. Only raises if the CSV itself is unusable (missing required
    columns, empty file).

    Returns:
        A de-duplicated (by normalized DOI, first occurrence wins) list of
        `RetractedPaper`.

    Raises:
        DataValidationError: if required columns (`doi`, `title`) cannot be
            located in the header.
    """
    reader, fh = _read_rows_with_fallback_encoding(csv_path)

    papers: list[RetractedPaper] = []
    seen_dois: set[str] = set()

    n_rows = 0
    n_missing_doi = 0
    n_malformed = 0
    n_duplicates = 0

    try:
        columns = _resolve_columns(list(reader.fieldnames or []))

        for row in reader:
            n_rows += 1

            raw_doi = (row.get(columns.get("doi", ""), "") or "").strip()
            if not raw_doi:
                n_missing_doi += 1
                continue

            try:
                norm_doi = normalize_doi(raw_doi)
            except ValueError:
                logger.warning("Row %d: DOI %r does not look valid; skipping.", n_rows, raw_doi)
                n_malformed += 1
                continue

            if norm_doi in seen_dois:
                n_duplicates += 1
                continue

            raw_title = (row.get(columns.get("title", ""), "") or "").strip()
            if not raw_title:
                logger.warning("Row %d (doi=%s): missing title; skipping.", n_rows, norm_doi)
                n_malformed += 1
                continue

            raw_journal = (row.get(columns.get("journal", ""), "") or "").strip() or None

            raw_reason = (row.get(columns.get("reason", ""), "") or "").strip()
            reason_enum = classify_reason(raw_reason)
            reason_raw = raw_reason or _FALLBACK_REASON_RAW

            original_pub_date = _parse_date(row.get(columns.get("original_pub_date", "")))
            retraction_date = _parse_date(row.get(columns.get("retraction_date", "")))
            citation_count_hint = _parse_citation_count(row.get(columns.get("citation_count", "")))

            try:
                paper = RetractedPaper(
                    doi=norm_doi,
                    title=raw_title,
                    journal=raw_journal,
                    original_pub_date=original_pub_date,
                    retraction_date=retraction_date,
                    retraction_reason=reason_enum,
                    retraction_reason_raw=reason_raw,
                    citation_count_hint=citation_count_hint,
                )
            except (ValidationError, ValueError) as exc:
                logger.warning("Row %d (doi=%s): failed schema validation: %s", n_rows, norm_doi, exc)
                n_malformed += 1
                continue

            seen_dois.add(norm_doi)
            papers.append(paper)

    finally:
        fh.close()

    logger.info(
        "Parsed Retraction Watch CSV: %d rows read, %d kept, %d missing DOI, "
        "%d malformed/skipped, %d duplicate DOIs dropped.",
        n_rows,
        len(papers),
        n_missing_doi,
        n_malformed,
        n_duplicates,
    )

    return papers


# --------------------------------------------------------------------------
# Seed selection
# --------------------------------------------------------------------------

_HIGH_SEVERITY_REASONS = {RetractionReason.FABRICATION, RetractionReason.FALSIFIED_DATA}
_MIN_DATE = date.min


def select_seed_retractions(
    papers: list[RetractedPaper],
    count: Optional[int] = None,
) -> list[RetractedPaper]:
    """Select a small, demo-ready seed set of high-severity retractions.

    Filters to fabrication/falsified-data retractions, applies a citation
    count band when that data is actually present, and returns the most
    recent `count` papers.

    Args:
        papers: full parsed Retraction Watch dataset.
        count: how many to return. Defaults to `settings.seed_retraction_count`.

    Returns:
        Up to `count` `RetractedPaper`, sorted most-recently-retracted first,
        with no duplicate DOIs.

    Raises:
        DataValidationError: if fewer than 1 paper survives filtering.
    """
    resolved_count = count if count is not None else settings.seed_retraction_count

    high_severity = [p for p in papers if p.retraction_reason in _HIGH_SEVERITY_REASONS]

    citation_data_present = any(p.citation_count_hint > 0 for p in high_severity)
    if not citation_data_present:
        logger.warning(
            "No usable citation-count data found in Retraction Watch CSV "
            "(all citation_count_hint == 0); citation-count filtering disabled."
        )
        candidates = high_severity
    else:
        candidates = [
            p
            for p in high_severity
            if p.citation_count_hint == 0
            or settings.min_citation_count <= p.citation_count_hint <= settings.max_citation_count
        ]

    if len(candidates) < 1:
        raise DataValidationError(
            "No retracted papers survived filtering "
            f"(started with {len(papers)}, {len(high_severity)} high-severity, "
            f"0 within citation band [{settings.min_citation_count}, {settings.max_citation_count}])."
        )

    candidates.sort(
        key=lambda p: p.retraction_date or _MIN_DATE,
        reverse=True,
    )

    selected = candidates[:resolved_count]

    logger.info(
        "Selected %d seed retraction(s) out of %d high-severity candidates "
        "(from %d total parsed papers).",
        len(selected),
        len(candidates),
        len(papers),
    )

    return selected
