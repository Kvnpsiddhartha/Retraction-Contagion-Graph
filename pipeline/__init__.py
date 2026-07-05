"""pipeline — orchestration layer that turns raw data/ + memory/ primitives
into the artifacts the memory layer ingests, and that runs the full
ingestion flow end-to-end.

Exposes:
    * `document_builder` (Prompt 03) — builds MemoryDocument/DependencyEdge
      artifacts from raw retracted/citing paper data.
    * `ingest_runner` (Prompt 04) — the single entrypoint (`run_ingestion`)
      that wires `data/`, `pipeline/document_builder.py`, and
      `memory/cognee_service.py` together into one CSV -> Cognee run, also
      usable as a CLI via `python -m pipeline.ingest_runner`.
"""
