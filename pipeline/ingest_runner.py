"""
pipeline/ingest_runner.py — Prompt 04: Ingestion Runner.

The single entrypoint that runs the full pipeline end-to-end:

    CSV download -> parse -> seed selection -> contagion graph build
    -> Cognee ingestion (documents + edges)

This module is deliberately a thin orchestrator: every real decision
(HTTP retries, CSV parsing/column-aliasing, seed-selection filters,
citation-relation classification, second-hop bounding, Cognee version
negotiation) already lives in the modules it imports. `run_ingestion`
just calls them in the documented order and turns the result into a
single `IngestionResult` that both the CLI and the (Phase 5) FastAPI
`/api/ingest` endpoint can consume.

Used as:
    * A standalone CLI: `python -m pipeline.ingest_runner [--seed-count N] [--force-refresh]`
    * An importable coroutine: `await run_ingestion(service=<CogneeService>)`,
      called by `api/main.py`'s `/api/ingest` endpoint (Phase 5) and by
      `scripts/seed_demo.py` (Phase 6) as the first step of the demo script.

Partial-failure trade-off (see `run_ingestion` step 6/7 below): a
`MemoryServiceError` raised while remembering documents does not abort the
run. We still attempt to register every edge we already built, because the
structured edge graph is independently useful (e.g. to the `/api/graph`
endpoint) even if the free-text documents failed to embed into Cognee. The
failure itself is never swallowed — it's logged at ERROR and surfaced in
`IngestionResult.errors`, so callers can tell the run was incomplete.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Optional

from pydantic import BaseModel, Field

from data.retraction_watch import (
    fetch_retraction_watch_csv,
    parse_retraction_watch_csv,
    select_seed_retractions,
)
from memory.cognee_service import CogneeService, get_cognee_service
from pipeline.document_builder import build_contagion_graph
from shared.config import configure_logging, settings
from shared.exceptions import DataValidationError, MemoryServiceError

logger = logging.getLogger(__name__)


class IngestionResult(BaseModel):
    """Summary of one end-to-end ingestion run.

    `errors` is populated (never raised past this boundary) for failure
    modes that still leave the run in a partially-useful state — currently
    just a `MemoryServiceError` from `remember_documents`. A completely
    unusable run (e.g. an empty seed set) is instead a hard failure that
    propagates out of `run_ingestion` as an exception; see the module
    docstring and the `DataValidationError` handling in `main()`.
    """

    seed_paper_count: int
    document_count: int
    edge_count: int
    documents_ingested: int
    errors: list[str] = Field(default_factory=list)


async def run_ingestion(
    service: Optional[CogneeService] = None,
    seed_count: Optional[int] = None,
    force_refresh_csv: bool = False,
) -> IngestionResult:
    """Run the full ingestion pipeline end-to-end.

    Args:
        service: `CogneeService` to ingest into. Defaults to the module-
            level singleton (`get_cognee_service()`) — callers that want a
            fresh/mocked instance (tests, or a FastAPI-injected instance)
            pass one explicitly instead.
        seed_count: how many seed retractions to select. Defaults to
            `settings.seed_retraction_count`.
        force_refresh_csv: if True, re-download the Retraction Watch CSV
            even if a cached copy already exists.

    Returns:
        An `IngestionResult` with all counts populated. `errors` is
        non-empty if remembering documents into Cognee failed, but the run
        otherwise completed (edges are still registered in that case).

    Raises:
        ValueError: if `seed_count` is zero or negative. Raised before any
            network call happens.
        DataValidationError: propagated as-is from `select_seed_retractions`
            if no papers survive filtering — a run that can't even select a
            seed set can't proceed, so this is a hard failure rather than
            an entry in `errors`.
        ExternalAPIError: propagated as-is from `fetch_retraction_watch_csv`
            if the CSV can't be downloaded after all retries.
    """
    if seed_count is not None and seed_count <= 0:
        raise ValueError(f"seed_count must be a positive integer, got {seed_count!r}")

    resolved_service = service or get_cognee_service()
    resolved_seed_count = seed_count if seed_count is not None else settings.seed_retraction_count

    logger.info(
        "Starting ingestion run (seed_count=%d, force_refresh_csv=%s)",
        resolved_seed_count,
        force_refresh_csv,
    )

    # Steps 2-5: sync CSV fetch/parse/select + graph build. No awaiting
    # needed here — these are plain function calls, safe to run directly
    # inside this coroutine per the non-functional requirements.
    csv_path = fetch_retraction_watch_csv(force_refresh=force_refresh_csv)
    all_papers = parse_retraction_watch_csv(csv_path)
    seed_papers = select_seed_retractions(all_papers, count=resolved_seed_count)
    documents, edges = build_contagion_graph(seed_papers)

    logger.info(
        "Graph build complete: %d seed paper(s), %d document(s), %d edge(s).",
        len(seed_papers),
        len(documents),
        len(edges),
    )

    # Step 6: remember documents. A MemoryServiceError here is caught (not
    # re-raised) so that step 7 (edge registration) still runs — see the
    # module docstring for the reasoning behind this trade-off.
    errors: list[str] = []
    documents_ingested = 0
    try:
        documents_ingested = await resolved_service.remember_documents(documents)
    except MemoryServiceError as exc:
        logger.error("remember_documents failed during ingestion run: %s", exc)
        errors.append(str(exc))

    # Step 7: register edges regardless of whether step 6 succeeded. Each
    # edge is registered independently so that one bad edge can't prevent
    # the rest of the batch from being recorded.
    for edge in edges:
        try:
            await resolved_service.register_edge(edge)
        except Exception as exc:  # register_edge is a local in-memory write;
            # not expected to raise in normal CogneeService usage, but we
            # still guard it so a single unexpected failure degrades to a
            # logged/recorded error instead of crashing the whole run.
            logger.error("register_edge failed for edge id=%s: %s", edge.id, exc)
            errors.append(f"register_edge failed for edge id={edge.id}: {exc}")

    result = IngestionResult(
        seed_paper_count=len(seed_papers),
        document_count=len(documents),
        edge_count=len(edges),
        documents_ingested=documents_ingested,
        errors=errors,
    )

    logger.info(
        "Ingestion run finished: seeds=%d documents=%d edges=%d ingested=%d errors=%d",
        result.seed_paper_count,
        result.document_count,
        result.edge_count,
        result.documents_ingested,
        len(result.errors),
    )
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.ingest_runner",
        description=(
            "Run the retraction contagion graph ingestion pipeline end-to-end: "
            "download the Retraction Watch CSV, select a seed set of retracted "
            "papers, build the citation-contagion graph, and ingest it into Cognee."
        ),
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of seed retractions to select. Defaults to "
            "settings.seed_retraction_count (currently "
            f"{settings.seed_retraction_count}) when omitted. Must be a "
            "positive integer."
        ),
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download the Retraction Watch CSV even if a cached copy exists.",
    )
    return parser


def _print_summary(result: IngestionResult) -> None:
    """Print a human-readable summary of an IngestionResult to stdout."""
    print("Ingestion run summary")
    print("----------------------")
    print(f"Seed papers selected:  {result.seed_paper_count}")
    print(f"Documents built:       {result.document_count}")
    print(f"Documents ingested:    {result.documents_ingested}")
    print(f"Edges registered:      {result.edge_count}")
    if result.errors:
        print(f"Errors ({len(result.errors)}):")
        for err in result.errors:
            print(f"  - {err}")
    else:
        print("Errors: none")


def main() -> None:
    """CLI entrypoint. Safe to run with zero required arguments."""
    configure_logging()
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        result = asyncio.run(
            run_ingestion(seed_count=args.seed_count, force_refresh_csv=args.force_refresh)
        )
    except ValueError as exc:
        # seed_count <= 0 — caught here so the CLI reports a clean message
        # rather than a raw traceback, per the spec's edge cases.
        print(f"Invalid arguments: {exc}", file=sys.stderr)
        sys.exit(1)
    except DataValidationError as exc:
        # Empty seed set (or unusable CSV) after filtering — a hard failure
        # that run_ingestion deliberately lets propagate rather than
        # folding into IngestionResult.errors. The CLI is the one place
        # that must still turn it into a clear message + non-zero exit
        # instead of an unhandled traceback.
        print(f"Ingestion cannot proceed: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_summary(result)
    sys.exit(1 if result.errors else 0)


if __name__ == "__main__":
    main()
