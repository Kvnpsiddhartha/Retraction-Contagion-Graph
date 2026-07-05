"""
api/main.py — FastAPI backend for the Retraction Contagion Graph project.

Implements exactly the HTTP contract `frontend/index.html` (Prompt 02d) was
built against, on top of `memory/cognee_service.py` (Prompt 02c) and
`pipeline/ingest_runner.py` (Prompt 04). This module owns routing, request/
response validation, and HTTP-level error mapping only — it holds no
business logic of its own.

Run with:

    uvicorn api.main:app --reload --port 8000

Then visit http://127.0.0.1:8000/docs for interactive Swagger UI.

Routes:
    GET  /health            -> HealthResponse
    POST /api/ingest         -> IngestionResult
    POST /api/recall         -> RecallResponse
    POST /api/improve        -> DependencyEdge
    POST /api/forget         -> DependencyEdge
    GET  /api/graph/{doi}    -> GraphResponse
"""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import cognee_dependency
from memory.cognee_service import CogneeService
from pipeline.ingest_runner import IngestionResult, run_ingestion
from shared.config import configure_logging
from shared.exceptions import DataValidationError, MemoryServiceError, NotFoundError
from shared.schemas import (
    DependencyEdge,
    EdgeStatus,
    ForgetRequest,
    GraphNode,
    GraphResponse,
    HealthResponse,
    ImproveRequest,
    RecallRequest,
    RecallResponse,
    normalize_doi,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Retraction Contagion Graph API")

# CORS: this is a hackathon-scale demo where `frontend/index.html` may be
# opened directly via `file://` or served from a throwaway local static
# server on an arbitrary port, so we allow every origin here. This is
# NOT safe for a real deployment — before deploying anywhere reachable
# outside a demo/dev machine, replace `allow_origins=["*"]` with an
# explicit allowlist (and re-evaluate `allow_credentials`).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


# Note: `@app.on_event` is deprecated in favor of lifespan context managers
# in newer FastAPI/Starlette, but it's used here deliberately to match this
# prompt's required interface exactly; it still works correctly (just emits
# a DeprecationWarning) on current FastAPI versions.
@app.on_event("startup")
async def on_startup() -> None:
    """Configure logging once at process startup.

    `configure_logging()` is itself idempotent (see `shared/config.py`), so
    this is safe even under `--reload`'s multiple import passes.
    """
    configure_logging()


# --------------------------------------------------------------------------
# Global exception handler
# --------------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for anything not already mapped to a specific HTTP status
    by a route handler below. Never leaks internals to the client — the
    full exception (with traceback) is logged server-side only.
    """
    logger.exception(
        "Unhandled exception while processing %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again or contact support."},
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health(service: CogneeService = Depends(cognee_dependency)) -> HealthResponse:
    """Liveness/readiness probe. Always returns 200 — `is_ready()` never
    raises (see its docstring), and cognee being unready is a normal,
    reportable state, not a health-check failure in itself.
    """
    cognee_ready = await service.is_ready()
    return HealthResponse(status="ok", cognee_ready=cognee_ready)


@app.post("/api/ingest", response_model=IngestionResult)
async def ingest(
    seed_count: int | None = None,
    service: CogneeService = Depends(cognee_dependency),
) -> IngestionResult:
    """Run the ingestion pipeline (Prompt 04) end-to-end.

    A totally unusable seed set (`DataValidationError`, or a bad
    `seed_count` that `run_ingestion` rejects with `ValueError`) is a client
    error -> 422, since it means "there is nothing to ingest with these
    parameters", not a server fault.

    A *partial* failure (some documents failed to embed into Cognee) is
    NOT an HTTP error: `run_ingestion` already degrades gracefully and
    reports that in `IngestionResult.errors`, so we return 200 with the
    errors visible in the body. Anything else (e.g. `ExternalAPIError` from
    a CSV download that failed after retries) is not specifically mapped
    here and falls through to the global exception handler as a 500,
    since it reflects an infrastructure problem rather than bad input.
    """
    try:
        return await run_ingestion(service=service, seed_count=seed_count)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/recall", response_model=RecallResponse)
async def recall(
    req: RecallRequest, service: CogneeService = Depends(cognee_dependency)
) -> RecallResponse:
    """Run a natural-language recall query against Cognee.

    Note: `req.max_results` out-of-range (<=0 or >100) is already rejected
    with HTTP 422 by pydantic's own `Field(gt=0, le=100)` constraint on
    `RecallRequest` (shared/schemas.py) before this function body ever
    runs, via FastAPI's automatic request validation — no extra check is
    needed here.
    """
    try:
        results = await service.recall(req.query, max_results=req.max_results)
    except MemoryServiceError as exc:
        raise HTTPException(
            status_code=503, detail=f"memory service unavailable: {exc}"
        ) from exc
    return RecallResponse(query=req.query, results=results)


@app.post("/api/improve", response_model=DependencyEdge)
async def improve(
    req: ImproveRequest, service: CogneeService = Depends(cognee_dependency)
) -> DependencyEdge:
    """Record a reviewer's verdict ('confirmed' / 'false_positive') on a
    flagged edge.
    """
    try:
        return await service.improve(req.edge_id, req.verdict, req.reviewer_note)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MemoryServiceError as exc:
        raise HTTPException(
            status_code=503, detail=f"memory service unavailable: {exc}"
        ) from exc


@app.post("/api/forget", response_model=DependencyEdge)
async def forget(
    req: ForgetRequest, service: CogneeService = Depends(cognee_dependency)
) -> DependencyEdge:
    """Mark an edge as self-corrected. Same error mapping as `/api/improve`."""
    try:
        return await service.forget(req.edge_id, req.reason)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MemoryServiceError as exc:
        raise HTTPException(
            status_code=503, detail=f"memory service unavailable: {exc}"
        ) from exc


@app.get("/api/graph/{doi:path}", response_model=GraphResponse)
async def graph(
    doi: str, service: CogneeService = Depends(cognee_dependency)
) -> GraphResponse:
    """Return the local dependency subgraph anchored at `doi`.

    Integration note: real DOIs contain a `/` (e.g. `10.1000/xyz`). ASGI
    servers (uvicorn) percent-decode the request path *before* Starlette
    ever sees it, so a plain `{doi}` (str) path converter would see the
    decoded slash as an extra path segment and 404. We use the `:path`
    converter instead, which matches the rest of the URL including
    slashes, so this route works whether the caller percent-encodes the
    DOI (`encodeURIComponent(doi)` in JS, the recommended approach) or not.
    `frontend/index.html`'s `fetchGraph` should still `encodeURIComponent`
    the DOI for URL-safety in general (e.g. DOIs occasionally contain `?`
    or `#`), but a literal `/` specifically will resolve correctly either
    way given `:path`.

    An unringested (or malformed-but-otherwise-well-formed) DOI with zero
    edges is a valid, if uninteresting, answer -> 200 with just the root
    node and no edges. A DOI that fails to normalize at all (doesn't look
    like a DOI, per `normalize_doi`) is a client error -> 422.
    """
    try:
        root_doi = normalize_doi(doi)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid DOI {doi!r}: {exc}") from exc

    edges = service.list_edges_for_doi(root_doi)

    # For every doi that appears as an edge's `to_doi`, remember the status
    # of the first such edge encountered, so each node can show "the status
    # of the edge pointing into it" without an O(n^2) scan per node.
    status_by_doi: dict[str, EdgeStatus] = {}
    for edge in edges:
        status_by_doi.setdefault(edge.to_doi, edge.status)

    # Root node type: 'retracted' if some hop-1 edge points *into* the root
    # (i.e. the root is the original retracted paper being cited), else
    # 'citing' (the root itself is a citing paper we're inspecting).
    root_is_retracted = any(
        edge.to_doi == root_doi and edge.hop_depth == 1 for edge in edges
    )

    # TODO: CogneeService doesn't expose paper titles directly in this MVP;
    # a real implementation would join back to the ingested MemoryDocument
    # metadata (see pipeline/document_builder.py) for a human-readable
    # title. Using the DOI itself as a placeholder title in the meantime.
    nodes: dict[str, GraphNode] = {
        root_doi: GraphNode(
            doi=root_doi,
            title=root_doi,
            node_type="retracted" if root_is_retracted else "citing",
            status=status_by_doi.get(root_doi),
        )
    }

    for edge in edges:
        for candidate_doi in (edge.from_doi, edge.to_doi):
            if candidate_doi in nodes:
                continue
            nodes[candidate_doi] = GraphNode(
                doi=candidate_doi,
                title=candidate_doi,
                node_type="citing",
                status=status_by_doi.get(candidate_doi),
            )

    return GraphResponse(root_doi=root_doi, nodes=list(nodes.values()), edges=edges)
