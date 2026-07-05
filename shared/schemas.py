"""
shared/schemas.py — the single source of truth for every data shape in the
Retraction Contagion Graph project.

Field names, types, and enum values here are load-bearing: `data/`,
`memory/`, `pipeline/`, and `api/` all import these models verbatim and
assume this exact shape. This module contains schemas and pure validation
logic only — no HTTP calls, no Cognee imports, no I/O of any kind.
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# --------------------------------------------------------------------------
# DOI normalization
# --------------------------------------------------------------------------

# Matches, case-insensitively, any of the common DOI resolver prefixes:
#   https://doi.org/10.1000/xyz
#   http://doi.org/10.1000/xyz
#   https://dx.doi.org/10.1000/xyz
#   http://dx.doi.org/10.1000/xyz
#   doi.org/10.1000/xyz            (scheme omitted)
#   doi:10.1000/xyz                (bare "doi:" scheme, sometimes seen in CSV exports)
_DOI_PREFIX_RE = re.compile(
    r"^\s*(?:(?:https?://)?(?:dx\.)?doi\.org/|doi:\s*)",
    re.IGNORECASE,
)


def normalize_doi(doi: str) -> str:
    """Normalize a DOI string to a canonical, comparable form.

    Canonical form is: lowercase, no surrounding whitespace, no resolver
    prefix (``https://doi.org/``, ``http://dx.doi.org/``, ``doi:``, etc).

    This is the *only* place DOI normalization is implemented in the
    project (per architecture invariant #1) — every schema that carries a
    DOI field routes through this function via a `field_validator`.

    Args:
        doi: a raw DOI string, possibly prefixed, whitespace-padded, or
            mixed-case.

    Returns:
        The normalized DOI, e.g. ``"10.1000/xyz"``.

    Raises:
        ValueError: if `doi` is empty/whitespace-only, or if after
            stripping known prefixes nothing resembling a DOI remains
            (a DOI must contain a "/" separating registrant code and
            suffix, per the DOI spec).

    Examples:
        >>> normalize_doi(" HTTPS://DOI.ORG/10.1000/ABC ")
        '10.1000/abc'
        >>> normalize_doi("http://dx.doi.org/10.1000/XyZ")
        '10.1000/xyz'
        >>> normalize_doi("10.1000/xyz")
        '10.1000/xyz'
        >>> normalize_doi("doi:10.1000/xyz")
        '10.1000/xyz'
    """
    if doi is None:
        raise ValueError("DOI must not be None")

    candidate = doi.strip()
    if not candidate:
        raise ValueError("DOI must not be empty or whitespace-only")

    candidate = _DOI_PREFIX_RE.sub("", candidate)
    candidate = candidate.strip().strip("/")
    candidate = candidate.lower()

    if not candidate or "/" not in candidate:
        raise ValueError(
            f"Value does not look like a valid DOI after normalization: {doi!r} -> {candidate!r}"
        )

    return candidate


class RetractionReason(str, Enum):
    FABRICATION = "fabrication"
    FALSIFIED_DATA = "falsified_data"
    PLAGIARISM = "plagiarism"
    ETHICAL_VIOLATION = "ethical_violation"
    ERROR = "error"
    OTHER = "other"


class RetractedPaper(BaseModel):
    """A paper that has been retracted, as sourced from Retraction Watch / Crossref."""

    doi: str
    title: str
    journal: Optional[str] = None
    original_pub_date: Optional[date] = None
    retraction_date: Optional[date] = None
    retraction_reason: RetractionReason
    retraction_reason_raw: str
    citation_count_hint: int = Field(default=0, ge=0)

    @field_validator("doi")
    @classmethod
    def _norm(cls, v: str) -> str:
        return normalize_doi(v)

    @field_validator("title", "retraction_reason_raw")
    @classmethod
    def _non_empty_str(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be empty or whitespace-only")
        return v

    @model_validator(mode="after")
    def _retraction_after_publication(self) -> "RetractedPaper":
        if (
            self.original_pub_date is not None
            and self.retraction_date is not None
            and self.retraction_date < self.original_pub_date
        ):
            raise ValueError(
                "retraction_date "
                f"({self.retraction_date}) cannot be earlier than "
                f"original_pub_date ({self.original_pub_date})"
            )
        return self


class CitationIntent(str, Enum):
    BACKGROUND = "background"
    METHOD = "method"
    RESULT = "result"
    UNKNOWN = "unknown"


class CitingPaper(BaseModel):
    """A paper that cites a retracted paper (directly or transitively)."""

    doi: str
    title: str
    abstract: Optional[str] = None
    pub_date: Optional[date] = None

    @field_validator("doi")
    @classmethod
    def _norm(cls, v: str) -> str:
        return normalize_doi(v)

    @field_validator("title")
    @classmethod
    def _non_empty_title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title must not be empty or whitespace-only")
        return v

    @field_validator("abstract")
    @classmethod
    def _blank_abstract_to_none(cls, v: Optional[str]) -> Optional[str]:
        # Semantic Scholar frequently returns "" rather than omitting the
        # field entirely; normalize that to None so downstream code can do
        # a single `if abstract is None` check (degrade, don't crash).
        if v is None:
            return None
        v = v.strip()
        return v or None


class CitationContext(BaseModel):
    """The specific context in which one paper cites another."""

    citing_doi: str
    cited_doi: str
    context_sentence: Optional[str] = None
    intent: CitationIntent = CitationIntent.UNKNOWN
    source: Literal["semantic_scholar", "manual_fallback"] = "semantic_scholar"

    @field_validator("citing_doi", "cited_doi")
    @classmethod
    def _norm(cls, v: str) -> str:
        return normalize_doi(v)

    @field_validator("context_sentence")
    @classmethod
    def _blank_sentence_to_none(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None

    @model_validator(mode="after")
    def _no_self_citation(self) -> "CitationContext":
        if self.citing_doi == self.cited_doi:
            raise ValueError(
                f"citing_doi and cited_doi must differ (got {self.citing_doi!r} for both)"
            )
        return self


class EdgeStatus(str, Enum):
    FLAGGED = "flagged"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    SELF_CORRECTED = "self_corrected"


class RelationType(str, Enum):
    RELIES_ON = "relies_on"
    CITES_IN_PASSING = "cites_in_passing"
    CONTRADICTS = "contradicts"


class DependencyEdge(BaseModel):
    """A typed, directed edge from one paper's claim to another's.

    ID contract (enforced, not just documented):
        `id` MUST be the deterministic string ``f"{from_doi}__{to_doi}"``
        computed from the *normalized* DOIs. The ID is constructed by the
        caller (`pipeline/document_builder.py`), not derived automatically
        by this model, because the builder is the layer that owns
        deduplication decisions during re-ingestion. This model validates
        the contract on construction so a caller mistake (e.g. swapping
        from/to, or hand-typing a stale id) fails fast at the schema
        boundary instead of silently corrupting the graph with duplicate
        or mismatched edges.

        Re-ingesting the same (from_doi, to_doi) pair must always produce
        the same `id`, so `memory/cognee_service.py` can use it as a
        natural dedupe key and never create duplicate edges.
    """

    id: str
    from_doi: str
    to_doi: str
    relation: RelationType
    confidence: float = Field(ge=0.0, le=1.0)
    hop_depth: int = Field(ge=1)
    status: EdgeStatus = EdgeStatus.FLAGGED
    evidence_text: Optional[str] = None

    @field_validator("from_doi", "to_doi")
    @classmethod
    def _norm(cls, v: str) -> str:
        return normalize_doi(v)

    @field_validator("id")
    @classmethod
    def _non_empty_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("id must not be empty or whitespace-only")
        return v

    @field_validator("evidence_text")
    @classmethod
    def _blank_evidence_to_none(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None

    @model_validator(mode="after")
    def _id_matches_deterministic_contract(self) -> "DependencyEdge":
        if self.from_doi == self.to_doi:
            raise ValueError(
                f"from_doi and to_doi must differ (got {self.from_doi!r} for both)"
            )
        expected_id = f"{self.from_doi}__{self.to_doi}"
        if self.id != expected_id:
            raise ValueError(
                f"DependencyEdge.id must be deterministic: expected {expected_id!r} "
                f"from (from_doi={self.from_doi!r}, to_doi={self.to_doi!r}), got {self.id!r}"
            )
        return self


class MemoryDocType(str, Enum):
    RETRACTED_PAPER = "retracted_paper"
    CITING_PAPER = "citing_paper"
    DEPENDENCY_EDGE = "dependency_edge"


class MemoryDocument(BaseModel):
    """The flattened text+metadata unit actually handed to Cognee's `remember()`."""

    doc_id: str
    doc_type: MemoryDocType
    content: str
    metadata: dict = Field(default_factory=dict)

    @field_validator("doc_id")
    @classmethod
    def _non_empty_doc_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("doc_id must not be empty or whitespace-only")
        return v

    @field_validator("content")
    @classmethod
    def _non_empty_content(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("content must not be empty or whitespace-only")
        return v


# --------------------------------------------------------------------------
# API contract (used by api/main.py and frontend/index.html)
# --------------------------------------------------------------------------


class RecallRequest(BaseModel):
    query: str
    max_results: int = Field(default=10, gt=0, le=100)

    @field_validator("query")
    @classmethod
    def _non_empty_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty or whitespace-only")
        return v


class RecallResultItem(BaseModel):
    doc_id: str
    summary: str
    related_edges: list[DependencyEdge] = Field(default_factory=list)


class RecallResponse(BaseModel):
    query: str
    results: list[RecallResultItem]


class ImproveRequest(BaseModel):
    edge_id: str
    verdict: Literal["confirmed", "false_positive"]
    reviewer_note: Optional[str] = None

    @field_validator("edge_id")
    @classmethod
    def _non_empty_edge_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("edge_id must not be empty or whitespace-only")
        return v

    @field_validator("reviewer_note")
    @classmethod
    def _blank_note_to_none(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None


class ForgetRequest(BaseModel):
    edge_id: str
    reason: str

    @field_validator("edge_id", "reason")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be empty or whitespace-only")
        return v


class GraphNode(BaseModel):
    doi: str
    title: str
    node_type: Literal["retracted", "citing"]
    status: Optional[EdgeStatus] = None

    @field_validator("doi")
    @classmethod
    def _norm(cls, v: str) -> str:
        return normalize_doi(v)


class GraphResponse(BaseModel):
    root_doi: str
    nodes: list[GraphNode]
    edges: list[DependencyEdge]

    @field_validator("root_doi")
    @classmethod
    def _norm(cls, v: str) -> str:
        return normalize_doi(v)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    cognee_ready: bool
