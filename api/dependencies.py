"""
api/dependencies.py — FastAPI dependency wrappers.

Kept as a separate, tiny module (rather than inlining `get_cognee_service()`
calls directly in `api/main.py`) purely so tests can override it via
FastAPI's `app.dependency_overrides[cognee_dependency] = ...` without
touching the module-level singleton in `memory/cognee_service.py` at all.
"""

from __future__ import annotations

from memory.cognee_service import CogneeService, get_cognee_service


def cognee_dependency() -> CogneeService:
    """FastAPI dependency wrapping `get_cognee_service()`.

    A thin indirection layer: production code gets the same process-wide
    singleton every time, while tests can swap in a fake/mock `CogneeService`
    via `app.dependency_overrides` without needing to monkeypatch the
    singleton accessor itself.
    """
    return get_cognee_service()
