"""
shared/config.py — centralized runtime configuration.

Every module reads configuration exclusively through the `settings` singleton
exported here. Nothing outside this file should call `os.environ` directly,
so that all tunables stay discoverable in one place and `.env.example`
stays the single source of truth for what can be configured.

`configure_logging()` is a separate, explicit call (not a module-level side
effect) so that importing `shared.config` — e.g. from a unit test — never
mutates global logging state as a side effect of import. Each entrypoint
(api/main.py, pipeline/ingest_runner.py, scripts/seed_demo.py, ...) calls
`configure_logging()` exactly once at startup.
"""

from __future__ import annotations

import logging
import sys

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"
_VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


class Settings(BaseSettings):
    """Application configuration, loaded from environment variables / `.env`.

    Field values below are defaults used when no environment variable /
    `.env` entry overrides them. See `.env.example` for the corresponding
    `UPPER_SNAKE_CASE` environment variable names (pydantic-settings maps
    `crossref_mailto` -> `CROSSREF_MAILTO` automatically, case-insensitively).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- External API endpoints ---
    crossref_mailto: str = "demo@example.com"
    crossref_base_url: str = "https://api.crossref.org"
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"
    retraction_watch_csv_url: str = (
        "https://gitlab.com/crossref/retraction-watch-data/-/raw/main/retraction_watch.csv"
    )

    # --- HTTP resilience ---
    request_timeout_seconds: float = Field(default=10.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0)
    retry_backoff_base_seconds: float = Field(default=0.5, gt=0.0)

    # --- Ingestion scope ---
    seed_retraction_count: int = Field(default=8, gt=0)
    min_citation_count: int = Field(default=10, ge=0)
    max_citation_count: int = Field(default=50, gt=0)

    # --- Logging ---
    log_level: str = "INFO"

    @field_validator("crossref_base_url", "semantic_scholar_base_url", "retraction_watch_csv_url")
    @classmethod
    def _no_trailing_slash(cls, v: str) -> str:
        """Normalize base URLs so callers can safely do f"{base_url}/path" without '//'."""
        v = v.strip()
        if not v:
            raise ValueError("URL setting must not be empty")
        return v.rstrip("/")

    @field_validator("crossref_mailto")
    @classmethod
    def _mailto_looks_like_email(cls, v: str) -> str:
        v = v.strip()
        if "@" not in v or v.startswith("@") or v.endswith("@"):
            raise ValueError(
                "crossref_mailto must look like an email address (used for Crossref's polite pool)"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, v: str) -> str:
        normalized = v.strip().upper()
        if normalized not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, got {v!r}"
            )
        return normalized

    @model_validator(mode="after")
    def _citation_bounds_are_sane(self) -> "Settings":
        if self.min_citation_count > self.max_citation_count:
            raise ValueError(
                "min_citation_count "
                f"({self.min_citation_count}) must be <= max_citation_count "
                f"({self.max_citation_count})"
            )
        return self


settings = Settings()


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once, using `settings.log_level` by default.

    Idempotent: safe to call more than once (e.g. in tests that import
    several entrypoints) — it clears any handlers this function previously
    attached before re-adding one, instead of stacking duplicate handlers
    that would otherwise cause every log line to print multiple times.

    Args:
        level: optional override, e.g. "DEBUG". Falls back to
            `settings.log_level` when omitted.
    """
    resolved_level = (level or settings.log_level).strip().upper()
    if resolved_level not in _VALID_LOG_LEVELS:
        resolved_level = "INFO"

    root = logging.getLogger()
    root.setLevel(resolved_level)

    # Remove only handlers we previously installed (tagged below), so we
    # don't clobber handlers other libraries/tests may have configured.
    for existing in list(root.handlers):
        if getattr(existing, "_retraction_graph_handler", False):
            root.removeHandler(existing)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    handler._retraction_graph_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)
