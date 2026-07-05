"""
shared — foundation package for the Retraction Contagion Graph project.

This package contains only:
    - exceptions.py : typed exception hierarchy used across the whole codebase
    - config.py     : pydantic-settings Settings object + configure_logging()
    - schemas.py    : Pydantic v2 data models (internal + API contract)

No business logic, no I/O, no third-party service calls live here. Every
other module in the codebase (data/, memory/, pipeline/, api/) imports from
this package and must not redefine these shapes.
"""

from __future__ import annotations

from shared.exceptions import (
    RetractionGraphError,
    ExternalAPIError,
    DataValidationError,
    MemoryServiceError,
    NotFoundError,
)
from shared.config import Settings, settings, configure_logging
from shared.schemas import (
    normalize_doi,
    RetractionReason,
    RetractedPaper,
    CitationIntent,
    CitingPaper,
    CitationContext,
    EdgeStatus,
    RelationType,
    DependencyEdge,
    MemoryDocType,
    MemoryDocument,
    RecallRequest,
    RecallResultItem,
    RecallResponse,
    ImproveRequest,
    ForgetRequest,
    GraphNode,
    GraphResponse,
    HealthResponse,
)

__all__ = [
    # exceptions
    "RetractionGraphError",
    "ExternalAPIError",
    "DataValidationError",
    "MemoryServiceError",
    "NotFoundError",
    # config
    "Settings",
    "settings",
    "configure_logging",
    # schemas
    "normalize_doi",
    "RetractionReason",
    "RetractedPaper",
    "CitationIntent",
    "CitingPaper",
    "CitationContext",
    "EdgeStatus",
    "RelationType",
    "DependencyEdge",
    "MemoryDocType",
    "MemoryDocument",
    "RecallRequest",
    "RecallResultItem",
    "RecallResponse",
    "ImproveRequest",
    "ForgetRequest",
    "GraphNode",
    "GraphResponse",
    "HealthResponse",
]

__version__ = "0.1.0"
