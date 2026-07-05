"""
memory — Cognee integration layer for the Retraction Contagion Graph project.

Exposes `CogneeService` (the single point of contact with the `cognee`
library plus local edge-review state) and `get_cognee_service()` (a
module-level singleton accessor suitable for use as a FastAPI dependency).

No other module should import `cognee` directly — everything routes
through `CogneeService` so the rest of the codebase is insulated from
Cognee's own API surface/version churn.
"""

from __future__ import annotations

from memory.cognee_service import CogneeService, get_cognee_service

__all__ = [
    "CogneeService",
    "get_cognee_service",
]
