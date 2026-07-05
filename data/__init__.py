"""
data — ingestion layer for the Retraction Contagion Graph project.

This package contains:
    - retraction_watch.py : download/parse Retraction Watch CSV, select seed
                             retracted papers (Prompt 02a)
    - external_apis.py    : Crossref + Semantic Scholar HTTP clients
                             (Prompt 02b)

Both submodules depend only on `shared/` (schemas, config, exceptions) and
are independent of each other and of `memory/`, `pipeline/`, `api/`.
"""

from __future__ import annotations

__all__: list[str] = []
