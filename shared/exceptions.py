"""
shared/exceptions.py — typed exception hierarchy for the whole project.

Every module that can fail (HTTP clients, Cognee wrapper, pipeline builders,
API handlers) must raise one of these instead of a bare Exception, so that
callers (and FastAPI exception handlers) can pattern-match on failure mode.

Design notes:
    - All exceptions carry an optional `detail` string and optional `cause`
      (the original exception, if any) so that logging/handling code can
      surface *why* something failed without needing to re-parse `str(e)`.
    - `__str__` includes the cause when present, which makes these safe to
      log directly (`logger.error(str(exc))`) without losing root-cause info.
    - Exceptions are intentionally shallow (no deep inheritance) — a flat
      hierarchy off a single base is enough for this project's size and
      keeps `except (ExternalAPIError, MemoryServiceError)` style catches
      simple to read.
"""

from __future__ import annotations


class RetractionGraphError(Exception):
    """Base exception for the whole project.

    Every other exception in this project subclasses this, so callers that
    want a catch-all boundary (e.g. a FastAPI global exception handler) can
    do `except RetractionGraphError` and be confident they've caught every
    domain-specific failure, while unrelated bugs (KeyError, TypeError from
    a real code defect) still propagate and are not accidentally swallowed.
    """

    def __init__(self, detail: str | None = None, *, cause: Exception | None = None) -> None:
        self.detail = detail
        self.cause = cause
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        base = self.detail or self.__class__.__doc__ or self.__class__.__name__
        if self.cause is not None:
            return f"{base} (caused by {type(self.cause).__name__}: {self.cause})"
        return base

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self._build_message()


class ExternalAPIError(RetractionGraphError):
    """Raised when an upstream HTTP API (Crossref, Semantic Scholar, GitLab CSV) fails after retries."""

    def __init__(
        self,
        detail: str | None = None,
        *,
        cause: Exception | None = None,
        source: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.source = source
        self.status_code = status_code
        enriched = detail or "External API call failed after retries"
        if source:
            enriched = f"[{source}] {enriched}"
        if status_code is not None:
            enriched = f"{enriched} (status={status_code})"
        super().__init__(enriched, cause=cause)


class DataValidationError(RetractionGraphError):
    """Raised when ingested data fails to map onto our schemas."""

    def __init__(
        self,
        detail: str | None = None,
        *,
        cause: Exception | None = None,
        field: str | None = None,
    ) -> None:
        self.field = field
        enriched = detail or "Ingested data does not match expected schema"
        if field:
            enriched = f"{enriched} (field={field})"
        super().__init__(enriched, cause=cause)


class MemoryServiceError(RetractionGraphError):
    """Raised when Cognee remember/recall/improve/forget fails."""

    def __init__(
        self,
        detail: str | None = None,
        *,
        cause: Exception | None = None,
        operation: str | None = None,
    ) -> None:
        self.operation = operation
        enriched = detail or "Cognee memory operation failed"
        if operation:
            enriched = f"[{operation}] {enriched}"
        super().__init__(enriched, cause=cause)


class NotFoundError(RetractionGraphError):
    """Raised when a requested DOI/edge does not exist in memory."""

    def __init__(
        self,
        detail: str | None = None,
        *,
        cause: Exception | None = None,
        identifier: str | None = None,
    ) -> None:
        self.identifier = identifier
        enriched = detail or "Requested resource was not found in memory"
        if identifier:
            enriched = f"{enriched} (id={identifier})"
        super().__init__(enriched, cause=cause)
