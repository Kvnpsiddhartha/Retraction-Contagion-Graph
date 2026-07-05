"""
scripts/seed_demo.py — Prompt 06: Demo Seed Script.

Leaves the system in exactly the state the 4-minute demo script (see
`retraction-contagion-graph-context.md`, section 5) needs:

    * Real ingested data (a full `run_ingestion()` pass — no synthetic
      shortcuts).
    * One edge left `flagged`, untouched, for the presenter to click
      live during the `improve()` AND `forget()` demo beats, in that
      order (Steps 4 and 5 of the demo script) — the same edge walks
      flagged -> confirmed -> self_corrected live, on screen.
    * One edge pre-marked `confirmed` (a "before" example, already
      staged, so the presenter can point at an already-resolved case
      before triggering the live one).
    * One edge pre-marked `self_corrected` (if a third distinct edge is
      available) so the graph already shows a resolved "self-corrected"
      example without anyone needing to click it live.

This script deliberately reaches into `CogneeService._edges` (a private
attribute) to select demo edges. That is a documented, deliberate
exception granted specifically to this file by the Prompt 06 spec: this
script *is* the demo-state orchestration layer, and adding new public
`CogneeService` methods just to serve one script's edge-selection heuristic
would leak demo concerns into the memory layer. All access to the private
dict is funneled through the single `_snapshot_edges()` helper below, never
inlined at call sites, so the "we touched a private attribute here, on
purpose" fact stays visible in exactly one place.

Usage:
    python scripts/seed_demo.py
    python scripts/seed_demo.py --seed-count 10
    python scripts/seed_demo.py --force-refresh-csv
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# Make the project root importable.
#
# This file is meant to be run directly as `python scripts/seed_demo.py`
# (per DEMO.md), not as `python -m scripts.seed_demo`. When Python runs a
# script directly, it puts the script's own directory (scripts/) on
# sys.path[0], NOT the project root that shared/, memory/, pipeline/, and
# data/ live under — so a plain `from shared.config import settings` would
# raise ModuleNotFoundError. We fix that up front, before any first-party
# import, rather than asking the user to remember `python -m` or set
# PYTHONPATH themselves.
# --------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from memory.cognee_service import CogneeService, get_cognee_service  # noqa: E402
from pipeline.ingest_runner import IngestionResult, run_ingestion  # noqa: E402
from shared.config import configure_logging  # noqa: E402
from shared.exceptions import DataValidationError, MemoryServiceError, NotFoundError  # noqa: E402
from shared.schemas import DependencyEdge, EdgeStatus  # noqa: E402

logger = logging.getLogger(__name__)

# The project brief's non-negotiable minimum: without at least this many
# FLAGGED edges, there is no live improve()/forget() moment to demo, and
# seeding must fail loudly rather than produce a boring/broken demo.
_MIN_FLAGGED_EDGES_REQUIRED = 2

# How many distinct edges we *want* ideally (confirmed + flagged +
# self_corrected). Falling short of this (having exactly 2, not 3) is not
# a failure — see the edge case in the spec — just a smaller demo.
_IDEAL_DEMO_EDGE_COUNT = 3

_REVIEWER_NOTE_FOR_CONFIRM = "Pre-staged for demo: dependency confirmed on manual review."
_FORGET_REASON_FOR_DEMO = "paper issued erratum"

_BANNER = "=" * 70


class DemoSeedingError(RuntimeError):
    """Raised when the ingested graph cannot support the minimum viable demo.

    Kept as a distinct, local exception type (rather than reusing e.g.
    `DataValidationError` from `shared/exceptions.py`) because this failure
    mode is specific to *this script's* demo requirements, not to the
    shared data-validation contract the rest of the app relies on. `main()`
    catches this specifically to print a clean, actionable message instead
    of a raw traceback.
    """


# --------------------------------------------------------------------------
# Edge selection
# --------------------------------------------------------------------------


def _snapshot_edges(service: CogneeService) -> list[DependencyEdge]:
    """Return every edge currently registered on `service`.

    Deliberately reaches into the private `CogneeService._edges` dict —
    see the module docstring for why this is the one sanctioned place in
    the whole codebase that does this. `CogneeService` only exposes
    single-edge (`get_edge`) and per-DOI (`list_edges_for_doi`) public
    lookups, neither of which can answer "give me every edge so I can pick
    the best set to demo" without already knowing which DOIs to ask about
    — which is exactly the information this script is trying to discover.
    """
    return list(service._edges.values())  # noqa: SLF001 - sanctioned, see docstring


def _select_demo_edges(
    all_edges: list[DependencyEdge],
) -> tuple[DependencyEdge, DependencyEdge, Optional[DependencyEdge], str]:
    """Pick the demo edges (two, or three if available) plus the demo DOI.

    Selection strategy (fully deterministic — re-running against the same
    ingested graph always makes the same picks, which matters for the
    "running seed_demo.py twice in a row" idempotency requirement):

        1. Only `FLAGGED` edges are eligible. A fresh `run_ingestion()` call
           produces every edge with `status=EdgeStatus.FLAGGED`
           (`pipeline/document_builder.py:build_dependency_edge`) — picking
           from anything else would mean layering demo state on top of
           *prior* review state instead of a clean baseline.
        2. Prefer `hop_depth == 1` edges over `hop_depth == 2`. A hop-1
           edge's `to_doi` is the original retracted paper, which is a much
           cleaner thing for a presenter to call "the demo DOI" than a
           second-hop intermediate paper.
        3. Group the preferred pool by `to_doi` and prefer the group with
           the most members. Picking edges that share a `to_doi` means one
           `GET /api/graph/{demo_doi}` call surfaces the confirmed edge,
           the flagged edge, and the self-corrected edge all in the same
           view — much stronger for a live demo than three unrelated nodes.
        4. Within the chosen group, sort by edge `id` for a fully
           reproducible pick regardless of dict iteration order.

    Returns:
        `(edge_to_leave_flagged, edge_to_confirm, edge_to_forget_or_none, demo_doi)`

    Raises:
        DemoSeedingError: if fewer than `_MIN_FLAGGED_EDGES_REQUIRED` FLAGGED
            edges exist in `all_edges`.
    """
    flagged = [e for e in all_edges if e.status == EdgeStatus.FLAGGED]

    if len(flagged) < _MIN_FLAGGED_EDGES_REQUIRED:
        raise DemoSeedingError(
            f"Only {len(flagged)} FLAGGED edge(s) resulted from ingestion; "
            f"need at least {_MIN_FLAGGED_EDGES_REQUIRED} to stage a demo. "
            "Try increasing --seed-count (more seed retractions -> more "
            "citing papers -> more candidate edges), or check that "
            "data/external_apis.py can actually reach Crossref/Semantic "
            "Scholar from this network (see DEMO.md's Troubleshooting "
            "section)."
        )

    hop1 = [e for e in flagged if e.hop_depth == 1]
    pool = hop1 if len(hop1) >= _MIN_FLAGGED_EDGES_REQUIRED else flagged
    if pool is not hop1:
        logger.warning(
            "Fewer than %d hop-1 FLAGGED edges available (%d); falling back "
            "to selecting from all hop depths.",
            _MIN_FLAGGED_EDGES_REQUIRED,
            len(hop1),
        )

    groups: dict[str, list[DependencyEdge]] = {}
    for edge in pool:
        groups.setdefault(edge.to_doi, []).append(edge)

    # Prefer the largest group; break ties on to_doi string for full
    # determinism (dict iteration order is insertion order in modern
    # Python, but correctness shouldn't quietly depend on that).
    best_to_doi = max(groups, key=lambda doi: (len(groups[doi]), doi))
    group = sorted(groups[best_to_doi], key=lambda e: e.id)

    if len(group) < _MIN_FLAGGED_EDGES_REQUIRED:
        # The best single to_doi group still doesn't have enough members
        # (candidates spread thinly across many to_dois) — fall back to
        # the full pool. We lose the "one DOI shows everything" demo
        # nicety, but a working demo beats a picture-perfect one that
        # doesn't exist.
        logger.warning(
            "No single to_doi has >= %d candidate edges; falling back to "
            "selecting across multiple to_dois (the demo DOI will only "
            "show a subset of the picked edges).",
            _MIN_FLAGGED_EDGES_REQUIRED,
        )
        group = sorted(pool, key=lambda e: e.id)
        demo_doi = group[0].to_doi
    else:
        demo_doi = best_to_doi

    edge_to_confirm = group[0]
    edge_to_leave_flagged = group[1]
    edge_to_forget = group[2] if len(group) >= _IDEAL_DEMO_EDGE_COUNT else None

    return edge_to_leave_flagged, edge_to_confirm, edge_to_forget, demo_doi


def _find_second_hop_doi(all_edges: list[DependencyEdge], demo_doi: str) -> Optional[str]:
    """Find a DOI to paste into the graph view for the "second-hop case" beat.

    Looks for a hop-2 edge whose `to_doi` is a paper that itself has a
    hop-1 edge pointing at `demo_doi` — i.e. a paper that cites a citer of
    the original retracted paper, inheriting exposure without ever citing
    the retracted work directly. That paper's DOI is what the presenter
    pastes into the graph view: it shows both the hop-1 edge (into
    `demo_doi`) and the hop-2 edge (into it) on one screen, which is the
    whole point of the second-hop demo beat.

    Falls back to any hop-2 edge's `to_doi` at all (regardless of whether
    it chains back to `demo_doi` specifically) if no such chained example
    exists, since a second-hop example anywhere in the graph is still more
    useful for this demo beat than nothing. Returns None only if the
    ingested graph has zero hop-2 edges at all (e.g. every second-hop fetch
    failed, or the citer pool had no RELIES_ON hop-1 candidates to expand).
    """
    hop1_into_demo_doi = {
        edge.from_doi for edge in all_edges if edge.hop_depth == 1 and edge.to_doi == demo_doi
    }

    chained_hop2 = sorted(
        (edge for edge in all_edges if edge.hop_depth == 2 and edge.to_doi in hop1_into_demo_doi),
        key=lambda e: e.id,
    )
    if chained_hop2:
        return chained_hop2[0].to_doi

    any_hop2 = sorted((e for e in all_edges if e.hop_depth == 2), key=lambda e: e.id)
    if any_hop2:
        logger.warning(
            "No hop-2 edge chains back to demo_doi=%s specifically; using an "
            "unrelated hop-2 example instead (%s). Step 3 of the demo will "
            "be less visually connected to the main story.",
            demo_doi,
            any_hop2[0].to_doi,
        )
        return any_hop2[0].to_doi

    logger.warning(
        "No hop-2 edges exist anywhere in the ingested graph; the "
        "second-hop demo beat (Step 3) will have nothing to show. Consider "
        "increasing --seed-count to widen the citation pool."
    )
    return None


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


async def seed_for_demo(seed_count: int = 6, *, force_refresh_csv: bool = False) -> dict:
    """Run ingestion, then stage the demo-ready edge states described above.

    Args:
        seed_count: how many seed retractions to ingest. Defaults to 6
            (deliberately smaller than `settings.seed_retraction_count`'s
            default of 8) to keep the demo graph tight and fast to build
            within a live demo's setup window.
        force_refresh_csv: keyword-only, defaults to False. Re-downloads
            the Retraction Watch CSV even if a cached copy exists. Kept
            keyword-only and defaulted so `seed_for_demo(seed_count=...)`
            and `seed_for_demo()` — the calls the Prompt 06 interface is
            pinned to — are unaffected; only the CLI (`--force-refresh-csv`)
            needs this.

    Returns:
        A dict summarizing the run, safe to assert against in tests
        without parsing stdout:

            seed_paper_count, document_count, edge_count,
            documents_ingested, ingestion_errors,
            demo_doi, second_hop_doi,
            edge_id_confirmed, edge_id_flagged_for_live_demo,
            edge_id_self_corrected (may be None),
            self_corrected_skipped_reason (set iff the above is None)

    Raises:
        DemoSeedingError: if fewer than two FLAGGED edges result from
            ingestion (see `_select_demo_edges`), or if a pre-staging
            improve()/forget() call unexpectedly fails. Ingestion itself
            still ran and its edges are still registered on the service
            singleton when this is raised — only demo-staging failed.
        DataValidationError: propagated as-is from `run_ingestion` if the
            seed set is empty after filtering (e.g. no fabrication/
            falsified-data retractions survive the citation-count band).
    """
    configure_logging()

    service = get_cognee_service()

    logger.info(
        "Step 1/3: running full ingestion pipeline (seed_count=%d, force_refresh_csv=%s)...",
        seed_count,
        force_refresh_csv,
    )
    result: IngestionResult = await run_ingestion(
        service=service, seed_count=seed_count, force_refresh_csv=force_refresh_csv
    )

    if result.errors:
        print(_BANNER)
        print("WARNING: ingestion completed with errors — demo data is PARTIAL")
        print(_BANNER)
        for err in result.errors:
            print(f"  - {err}")
        print(
            "Continuing anyway: partial data is still demoable. The presenter "
            "should be aware some titles/context may be missing.\n"
        )
        logger.warning(
            "run_ingestion returned %d error(s); continuing with partial data.",
            len(result.errors),
        )

    logger.info(
        "Step 1/3 complete: %d seed paper(s), %d document(s) built, "
        "%d document(s) ingested, %d edge(s) registered.",
        result.seed_paper_count,
        result.document_count,
        result.documents_ingested,
        result.edge_count,
    )

    logger.info("Step 2/3: selecting demo edges from the ingested graph...")
    all_edges = _snapshot_edges(service)
    edge_to_leave_flagged, edge_to_confirm, edge_to_forget, demo_doi = _select_demo_edges(
        all_edges
    )
    second_hop_doi = _find_second_hop_doi(all_edges, demo_doi)

    logger.info(
        "Step 2/3 complete: demo_doi=%s, confirm=%s, leave_flagged=%s, forget=%s",
        demo_doi,
        edge_to_confirm.id,
        edge_to_leave_flagged.id,
        edge_to_forget.id if edge_to_forget else "<skipped - fewer than 3 candidates>",
    )

    logger.info("Step 3/3: staging edge states...")
    try:
        await service.improve(
            edge_to_confirm.id, "confirmed", reviewer_note=_REVIEWER_NOTE_FOR_CONFIRM
        )
    except (NotFoundError, MemoryServiceError) as exc:
        # Should not happen: edge_to_confirm was just read out of this same
        # service's own edge store. Guarded anyway per "no silent failures"
        # (00-architecture-overview.md invariant #3) — a demo script that
        # silently skips its own staging step is worse than one that fails
        # loudly and says why.
        raise DemoSeedingError(
            f"Failed to pre-stage confirmed edge {edge_to_confirm.id!r}: {exc}"
        ) from exc

    self_corrected_skipped_reason: Optional[str] = None
    if edge_to_forget is not None:
        try:
            await service.forget(edge_to_forget.id, _FORGET_REASON_FOR_DEMO)
        except (NotFoundError, MemoryServiceError) as exc:
            raise DemoSeedingError(
                f"Failed to pre-stage self-corrected edge {edge_to_forget.id!r}: {exc}"
            ) from exc
    else:
        self_corrected_skipped_reason = (
            "Only 2 distinct FLAGGED edges were available after ingestion "
            "(need 3 for a pre-staged self_corrected example); the live "
            "forget() demo beat will still work using the flagged edge, but "
            "there is no separate already-resolved example to point at."
        )
        logger.info(
            "Skipping pre-staged self_corrected edge: %s", self_corrected_skipped_reason
        )

    # edge_to_leave_flagged is intentionally left untouched here — it stays
    # FLAGGED so the presenter can click improve() AND forget() on it live,
    # in that order, during Steps 4 and 5 of the demo script.

    summary = {
        "seed_paper_count": result.seed_paper_count,
        "document_count": result.document_count,
        "edge_count": result.edge_count,
        "documents_ingested": result.documents_ingested,
        "ingestion_errors": list(result.errors),
        "demo_doi": demo_doi,
        "second_hop_doi": second_hop_doi,
        "edge_id_confirmed": edge_to_confirm.id,
        "edge_id_flagged_for_live_demo": edge_to_leave_flagged.id,
        "edge_id_self_corrected": edge_to_forget.id if edge_to_forget else None,
        "self_corrected_skipped_reason": self_corrected_skipped_reason,
    }

    _print_demo_summary(summary)
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _print_demo_summary(summary: dict) -> None:
    """Print the human-readable block a presenter copies values out of.

    Labels here (`DEMO_DOI`, `EDGE_ID_...`, etc.) are the exact tokens
    `DEMO.md` refers back to, so the runbook stays correct across CSV
    updates / different `--seed-count` values without hardcoding DOIs that
    only exist for one particular run.
    """
    print()
    print(_BANNER)
    print("DEMO SEED COMPLETE — copy the values below into your demo")
    print(_BANNER)
    print(f"Seed papers ingested:   {summary['seed_paper_count']}")
    print(f"Documents built:        {summary['document_count']}")
    print(f"Documents ingested:     {summary['documents_ingested']}")
    print(f"Edges registered:       {summary['edge_count']}")
    print(f"Ingestion errors:       {len(summary['ingestion_errors']) or 'none'}")
    print()
    print("DEMO_DOI  (Steps 1, 2, 4, 5 — paste into the graph view):")
    print(f"    {summary['demo_doi']}")
    print()
    if summary["second_hop_doi"]:
        print("SECOND_HOP_DOI  (Step 3 — outer ring / hop_depth=2 edge):")
        print(f"    {summary['second_hop_doi']}")
    else:
        print("SECOND_HOP_DOI: none available this run (see warnings above) — skip Step 3.")
    print()
    print("EDGE_ID_FLAGGED_FOR_LIVE_DEMO  (Steps 4 + 5 — click live, in order):")
    print(f"    {summary['edge_id_flagged_for_live_demo']}")
    print()
    print("EDGE_ID_CONFIRMED  (pre-staged 'before' example, already resolved):")
    print(f"    {summary['edge_id_confirmed']}")
    print()
    if summary["edge_id_self_corrected"]:
        print("EDGE_ID_SELF_CORRECTED  (pre-staged, already visible as resolved in the graph):")
        print(f"    {summary['edge_id_self_corrected']}")
    else:
        print(
            "EDGE_ID_SELF_CORRECTED: none pre-staged this run "
            f"({summary['self_corrected_skipped_reason']})"
        )
    print(_BANNER)
    print()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/seed_demo.py",
        description=(
            "Seed the Retraction Contagion Graph demo: run full ingestion, "
            "then stage flagged/confirmed/self_corrected edge states for "
            "the 4-minute demo script in DEMO.md."
        ),
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=6,
        metavar="N",
        help="Number of seed retractions to ingest (default: 6).",
    )
    parser.add_argument(
        "--force-refresh-csv",
        action="store_true",
        help="Re-download the Retraction Watch CSV even if a cached copy exists.",
    )
    return parser


def main() -> None:
    """CLI entrypoint. Exits 0 on success, 1 on any failure (with a clear message)."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.seed_count <= 0:
        print(
            f"Invalid arguments: --seed-count must be positive, got {args.seed_count}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        asyncio.run(
            seed_for_demo(seed_count=args.seed_count, force_refresh_csv=args.force_refresh_csv)
        )
    except DemoSeedingError as exc:
        print(f"\nDemo seeding failed: {exc}\n", file=sys.stderr)
        sys.exit(1)
    except DataValidationError as exc:
        # Propagated as-is from run_ingestion -> select_seed_retractions if
        # literally nothing survives CSV filtering (e.g. no fabrication/
        # falsified-data retractions in the current citation-count band).
        print(f"\nIngestion cannot proceed: {exc}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary, see docstring
        # Anything else (network failure exhausting retries, a genuine bug)
        # is still reported clearly and turned into a clean exit code
        # rather than a bare traceback, per this script's "exit(1) if it
        # raises" contract.
        logger.exception("seed_demo.py failed with an unexpected error")
        print(f"\nDemo seeding failed with an unexpected error: {exc}\n", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
