"""
memory/cognee_service.py — the single point of contact with Cognee.

Everything the rest of the app knows about "memory" comes through
`CogneeService`. It wraps two genuinely different kinds of state:

  1. What Cognee itself remembers — the actual knowledge graph of
     retracted papers, citing papers, and citation context. Cognee is the
     source of truth for that; we never re-implement it locally.
  2. `self._edges` — a plain in-process `dict[str, DependencyEdge]`, keyed
     by `edge.id`, that is *our* source of truth for human-review status
     (flagged / confirmed / false_positive / self_corrected). Cognee has
     no built-in notion of "a reviewer confirmed this dependency link", so
     that state is deliberately kept outside of it and merged back in at
     `recall()` time.

Cognee version note
--------------------
Cognee's public API surface has changed across releases: 1.0+ exposes
`cognee.remember()` / `cognee.recall()` directly; earlier releases only
expose the lower-level `cognee.add()` + `cognee.cognify()` + `cognee.search()`
primitives. Rather than hard-coding one shape, `_CogneeOps` introspects the
installed package at first use and routes through whichever surface is
actually present, so the rest of this class (and the rest of the codebase)
never has to know which cognee version is installed.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Optional

from shared.exceptions import MemoryServiceError, NotFoundError
from shared.schemas import (
    DependencyEdge,
    EdgeStatus,
    MemoryDocument,
    RecallResultItem,
    normalize_doi,
)

logger = logging.getLogger(__name__)

# Single dataset for the whole demo graph. Kept as a module constant (not a
# magic string scattered through the class) so it's obvious there's exactly
# one Cognee dataset in play for this project.
_DATASET_NAME = "retraction_contagion_graph"

# Maps the API-layer verdict strings to the internal EdgeStatus values.
# Re-checked defensively inside `improve()` even though `ImproveRequest`
# (shared/schemas.py) already constrains this via `Literal[...]`.
_VERDICT_TO_STATUS: dict[str, EdgeStatus] = {
    "confirmed": EdgeStatus.CONFIRMED,
    "false_positive": EdgeStatus.FALSE_POSITIVE,
}

# Statuses that must never be surfaced by recall() as "still live" edges.
# A false positive was reviewed and rejected; a self-corrected edge means
# the downstream paper already fixed the record. Both are resolved, not
# actionable, so recall() filters them out.
_SUPPRESSED_IN_RECALL = frozenset({EdgeStatus.FALSE_POSITIVE, EdgeStatus.SELF_CORRECTED})

# Loose DOI-shaped token matcher used only to *find candidate DOI strings*
# inside free text / metadata values coming back from cognee's recall
# results, so we can cross-reference them against our local edge store.
# This is intentionally permissive — real validation still goes through
# `shared.schemas.normalize_doi`, which will reject anything that doesn't
# actually parse as a DOI.
_DOI_TOKEN_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>,;]+", re.IGNORECASE)

# Metadata keys we know our own pipeline (Prompt 03) is likely to populate
# on a MemoryDocument, checked in addition to scanning free text.
_METADATA_DOI_KEYS = ("doi", "from_doi", "to_doi", "citing_doi", "cited_doi", "root_doi")


class _CogneeOps:
    """Runtime-negotiated adapter over whatever `cognee` actually exposes.

    Constructed once cognee has been successfully imported. Resolves, at
    construction time, whether the installed version has the modern
    `remember`/`recall` surface or only the legacy `add`/`cognify`/`search`
    primitives, and exposes a single stable async interface
    (`remember(...)`, `recall(...)`) regardless of which one it found.
    """

    def __init__(self, cognee_module: Any) -> None:
        self._cognee = cognee_module
        self._has_native_remember_recall = hasattr(cognee_module, "remember") and hasattr(
            cognee_module, "recall"
        )
        # Resolved lazily on first remember() call and cached, since the
        # import path for DataItem is itself version-specific and we don't
        # want to pay for (or fail on) that import if remember() is never
        # called (e.g. a process that only serves recall/improve/forget).
        self._data_item_cls: Any = None
        self._data_item_import_failed = False

    def _resolve_data_item_cls(self) -> Any:
        if self._data_item_cls is not None or self._data_item_import_failed:
            return self._data_item_cls
        try:
            from cognee.tasks.ingestion.data_item import DataItem  # version-specific path

            self._data_item_cls = DataItem
        except Exception as exc:
            self._data_item_import_failed = True
            logger.warning(
                "Could not import cognee's DataItem type (%s); falling back to "
                "ingesting raw text without attached metadata.",
                exc,
            )
        return self._data_item_cls

    async def remember(
        self, content: str, *, doc_id: str, metadata: dict, dataset_name: str
    ) -> None:
        cognee = self._cognee
        if self._has_native_remember_recall:
            data_item_cls = self._resolve_data_item_cls()
            if data_item_cls is not None:
                payload = data_item_cls(data=content, label=doc_id, external_metadata=metadata)
            else:
                payload = content
            await cognee.remember(data=payload, dataset_name=dataset_name)
        else:
            # Legacy fallback surface: add() ingests raw content, cognify()
            # builds/updates the graph from everything added so far.
            await cognee.add(content, dataset_name=dataset_name)
            await cognee.cognify(datasets=[dataset_name])

    async def recall(self, query_text: str, *, top_k: int) -> list[Any]:
        cognee = self._cognee
        if self._has_native_remember_recall:
            result = await cognee.recall(query_text=query_text, top_k=top_k)
        else:
            result = await cognee.search(query_text=query_text, top_k=top_k)
        return list(result) if result else []


class CogneeService:
    """Owns all interaction with the `cognee` library plus local edge-status state.

    Concurrency note: `self._edges` and `self._ingested_ids` are plain
    dict/set instances mutated without locking. That's safe for the
    single-process demo deployment this project targets, but is *not* safe
    if this service is ever run across multiple worker processes (e.g.
    `uvicorn --workers N`), since each process would hold its own copy of
    this state. For a 1-hour hackathon MVP this is an accepted, documented
    limitation — a real deployment would move `self._edges` into a shared
    store (Redis/Postgres) instead of an in-process dict.
    """

    def __init__(self) -> None:
        self._edges: dict[str, DependencyEdge] = {}
        self._ingested_ids: set[str] = set()
        self._ops: Optional[_CogneeOps] = None

        # Construction must never raise, even if cognee isn't installed or
        # can't be configured yet — actual connection errors are deferred
        # to first use via `_get_ops()`. We still probe eagerly here purely
        # so `is_ready()` can report accurately without requiring a prior
        # failed call, and so we can log the reason once at startup.
        try:
            import cognee  # noqa: F401  (import-only availability probe)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("cognee import failed during CogneeService construction: %s", exc)

    def _get_ops(self) -> _CogneeOps:
        """Resolve (and cache) the runtime cognee adapter.

        Raises:
            MemoryServiceError: if `cognee` cannot be imported/used at all.
        """
        if self._ops is not None:
            return self._ops
        try:
            import cognee
        except Exception as exc:
            raise MemoryServiceError(
                "cognee is not available: import failed", cause=exc, operation="init"
            ) from exc

        self._ops = _CogneeOps(cognee)
        return self._ops

    async def is_ready(self) -> bool:
        """Lightweight check that cognee is importable/initialized.

        Never raises; returns False on any failure and logs the reason.
        Deliberately avoids any network call (no remember/recall probe) so
        this stays cheap enough to call from a `/health` endpoint.
        """
        try:
            self._get_ops()
        except MemoryServiceError as exc:
            logger.warning("CogneeService.is_ready() -> False: %s", exc)
            return False
        return True

    async def remember_documents(self, documents: list[MemoryDocument]) -> int:
        """Ingest a batch of MemoryDocument into cognee.

        Idempotent per `doc_id` within this process: a `doc_id` already
        ingested earlier in the process's lifetime is skipped (logged at
        debug level) rather than re-ingested.

        Returns:
            The count of newly ingested documents (excludes skipped dupes).

        Raises:
            MemoryServiceError: on underlying cognee failure, or if cognee
                is unavailable.
        """
        if not documents:
            return 0

        ops = self._get_ops()  # raises MemoryServiceError if unavailable

        newly_ingested = 0
        for doc in documents:
            if doc.doc_id in self._ingested_ids:
                logger.debug("remember_documents: skipping already-ingested doc_id=%s", doc.doc_id)
                continue

            enriched_metadata = {
                **doc.metadata,
                "doc_id": doc.doc_id,
                "doc_type": doc.doc_type.value,
            }
            try:
                await ops.remember(
                    doc.content,
                    doc_id=doc.doc_id,
                    metadata=enriched_metadata,
                    dataset_name=_DATASET_NAME,
                )
            except Exception as exc:
                raise MemoryServiceError(
                    f"Failed to remember document {doc.doc_id!r}",
                    cause=exc,
                    operation="remember_documents",
                ) from exc

            self._ingested_ids.add(doc.doc_id)
            newly_ingested += 1

        logger.info(
            "remember_documents: ingested %d new document(s) out of %d submitted",
            newly_ingested,
            len(documents),
        )
        return newly_ingested

    async def register_edge(self, edge: DependencyEdge) -> None:
        """Register a DependencyEdge in the local edge store.

        Overwrites if the same id is registered again (last-write-wins),
        logging at debug level when overwriting.
        """
        if edge.id in self._edges:
            logger.debug("register_edge: overwriting existing edge id=%s (last-write-wins)", edge.id)
        self._edges[edge.id] = edge

    async def recall(self, query: str, max_results: int = 10) -> list[RecallResultItem]:
        """Run a natural-language query against cognee, enriched with edges.

        For each cognee result, any DependencyEdge in `self._edges` whose
        `from_doi`/`to_doi` can be matched against that result's metadata
        or text is attached as a `related_edge` — excluding edges whose
        status is FALSE_POSITIVE or SELF_CORRECTED, which are resolved and
        should stop surfacing here.

        Degradation contract:
            * If cognee is not installed or not configured (e.g. no LLM API
              key set), this returns `[]` with a WARNING log rather than
              raising. The graph, improve, and forget endpoints are all backed
              by `self._edges` (the local edge store) and are completely
              unaffected — only natural-language search is unavailable.
            * If cognee IS available but a transient error occurs mid-query,
              this raises `MemoryServiceError` so the caller sees a real
              failure rather than a silent empty result.

        Returns:
            `[]` (not an error) if cognee returns no matches, or if cognee
            is not available / not configured.

        Raises:
            MemoryServiceError: only on a transient query-level failure when
                cognee was already successfully initialised (i.e. _get_ops()
                succeeded but the actual recall() call failed).
        """
        # --- Step 1: resolve the cognee adapter (may fail if not installed) --
        try:
            ops = self._get_ops()
        except MemoryServiceError as exc:
            # cognee is not installed or the import is broken entirely.
            # Degrade to empty results rather than surfacing a 503 — the rest
            # of the demo (graph view, improve, forget) still works fine.
            logger.warning(
                "recall: cognee is not available (%s); returning empty results. "
                "Install cognee and set LLM_API_KEY (or OPENAI_API_KEY) in .env "
                "to enable natural-language search.",
                exc,
            )
            return []

        # --- Step 2: run the actual query (cognee IS available) --------------
        try:
            raw_results = await ops.recall(query, top_k=max_results)
        except Exception as exc:
            # Detect configuration errors (missing LLM/embedding API key) by
            # inspecting the exception type name and message string. We avoid
            # importing Cognee exception classes directly to stay version-
            # agnostic — Cognee's own exception hierarchy has changed across
            # releases, but the error *strings* are stable enough for this.
            exc_type = type(exc).__name__
            exc_msg = str(exc)
            exc_combined = exc_type + exc_msg

            _CREDENTIAL_SIGNALS = (
                "LLMAPIKeyNotSetError",
                "EmbeddingException",    # LiteLLM embedding key missing
                "Missing credentials",
                "InternalServerError",   # wraps OpenAI 401/422 on bad key
                "BadRequestError",       # missing model prefix e.g. openai/
                "LLM Provider NOT provided",
                "api key",               # generic fallback, case-insensitive
                "API key",
                "OPENAI_API_KEY",
            )
            is_credential_error = any(sig in exc_combined for sig in _CREDENTIAL_SIGNALS)

            if is_credential_error:
                logger.warning(
                    "recall: cognee LLM/embedding API key is not configured "
                    "(%s: %s); returning empty results. "
                    "Set OPENAI_API_KEY (or LLM_API_KEY) in your .env file "
                    "to enable natural-language search.",
                    exc_type,
                    exc_msg[:200],  # truncate very long embedding-engine error strings
                )
                return []
            # Any other failure (network error, model quota, genuine bug)
            # is a real transient fault — raise so the caller sees it.
            raise MemoryServiceError(
                f"cognee recall failed for query={query!r}", cause=exc, operation="recall"
            ) from exc

        if not raw_results:
            return []

        items: list[RecallResultItem] = []
        for idx, raw in enumerate(raw_results):
            try:
                items.append(self._normalize_recall_entry(query, idx, raw))
            except Exception as exc:
                # Degrade, don't crash (architecture invariant #5): one
                # unparseable cognee result must not sink the whole
                # recall() call — skip it and keep going.
                logger.warning("recall: skipping unparseable result #%d: %s", idx, exc)

        return items[:max_results]

    # -- recall() helpers ---------------------------------------------------

    def _normalize_recall_entry(self, query: str, idx: int, raw: Any) -> RecallResultItem:
        """Coerce one heterogeneous cognee result into a RecallResultItem.

        Cognee's actual result objects vary by version and search type
        (plain dict, a pydantic response model, a dataclass, ...), so this
        normalizes defensively via `_as_dict` rather than assuming one
        fixed shape.
        """
        data = self._as_dict(raw)

        doc_id = (
            data.get("doc_id")
            or data.get("id")
            or data.get("data_id")
            or data.get("node_id")
            or self._synthetic_doc_id(query, idx, data)
        )
        summary_raw = (
            data.get("summary")
            or data.get("answer")
            or data.get("text")
            or data.get("content")
            or ""
        )
        summary = str(summary_raw).strip() or "(cognee returned no summary text for this result)"

        related_edges = self._match_edges(data, summary)
        return RecallResultItem(doc_id=str(doc_id), summary=summary, related_edges=related_edges)

    @staticmethod
    def _as_dict(raw: Any) -> dict:
        """Best-effort coercion of an arbitrary cognee result into a dict."""
        if isinstance(raw, dict):
            return raw
        for method_name in ("model_dump", "dict"):
            method = getattr(raw, method_name, None)
            if callable(method):
                try:
                    return method()
                except Exception:
                    pass
        # Last resort: shallow attribute scrape (covers plain dataclasses /
        # simple namespaces that are neither dicts nor pydantic models).
        return {
            key: getattr(raw, key)
            for key in dir(raw)
            if not key.startswith("_") and not callable(getattr(raw, key, None))
        }

    @staticmethod
    def _synthetic_doc_id(query: str, idx: int, data: dict) -> str:
        """Deterministic fallback id when cognee's result carries none.

        Hashing (query, position, text) keeps repeated identical recall
        calls stable across a process's lifetime without requiring cognee
        to expose an id field at all.
        """
        basis = f"{query}:{idx}:{data.get('text') or data.get('content') or ''}"
        return "recall-" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]

    def _match_edges(self, data: dict, summary: str) -> list[DependencyEdge]:
        """Find locally-registered edges relevant to one recall result."""
        candidate_dois: set[str] = set()

        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            for key in _METADATA_DOI_KEYS:
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    candidate_dois.add(value)

        # Also scan the free text (summary + stringified metadata) for
        # DOI-shaped tokens, since not every cognee version/search-type is
        # guaranteed to surface structured metadata on every result.
        candidate_dois.update(_DOI_TOKEN_RE.findall(summary or ""))
        if metadata:
            candidate_dois.update(_DOI_TOKEN_RE.findall(str(metadata)))

        normalized_dois: set[str] = set()
        for raw_doi in candidate_dois:
            try:
                normalized_dois.add(normalize_doi(raw_doi))
            except ValueError:
                continue  # not actually DOI-shaped once normalized; ignore

        if not normalized_dois:
            return []

        matched: dict[str, DependencyEdge] = {}
        for edge in self._edges.values():
            if edge.status in _SUPPRESSED_IN_RECALL:
                continue
            if edge.from_doi in normalized_dois or edge.to_doi in normalized_dois:
                matched[edge.id] = edge
        return list(matched.values())

    # -- human feedback loop -------------------------------------------------

    async def improve(
        self, edge_id: str, verdict: str, reviewer_note: str | None = None
    ) -> DependencyEdge:
        """Record a reviewer's verdict on a flagged dependency edge.

        Args:
            edge_id: id of the edge being reviewed.
            verdict: `"confirmed"` or `"false_positive"`. Already validated
                by `ImproveRequest` (shared/schemas.py) at the API layer,
                but re-checked defensively here too.
            reviewer_note: optional free-text note from the reviewer.

        Raises:
            NotFoundError: if no edge with `edge_id` is registered.
            MemoryServiceError: if `verdict` is not a recognized value.
        """
        if verdict not in _VERDICT_TO_STATUS:
            raise MemoryServiceError(
                f"invalid verdict {verdict!r}; expected one of {sorted(_VERDICT_TO_STATUS)}",
                operation="improve",
            )

        edge = self._edges.get(edge_id)
        if edge is None:
            raise NotFoundError(f"No edge registered with id {edge_id!r}", identifier=edge_id)

        old_status = edge.status
        new_status = _VERDICT_TO_STATUS[verdict]

        if old_status is EdgeStatus.SELF_CORRECTED:
            # Reviewer can still reclassify a previously self-corrected
            # edge (e.g. a correction is later found to be inadequate) —
            # allowed, but worth flagging loudly since it looked terminal.
            logger.warning(
                "improve(edge_id=%s): overriding terminal SELF_CORRECTED status -> %s",
                edge_id,
                new_status.value,
            )

        updated = edge.model_copy(update={"status": new_status})
        self._edges[edge_id] = updated
        logger.info(
            "improve(edge_id=%s): %s -> %s%s",
            edge_id,
            old_status.value,
            new_status.value,
            f" (note: {reviewer_note})" if reviewer_note else "",
        )
        return updated

    async def forget(self, edge_id: str, reason: str) -> DependencyEdge:
        """Mark an edge as self-corrected: the downstream paper fixed itself.

        Raises:
            NotFoundError: if no edge with `edge_id` is registered.
        """
        edge = self._edges.get(edge_id)
        if edge is None:
            raise NotFoundError(f"No edge registered with id {edge_id!r}", identifier=edge_id)

        old_status = edge.status
        updated = edge.model_copy(update={"status": EdgeStatus.SELF_CORRECTED})
        self._edges[edge_id] = updated
        logger.info(
            "forget(edge_id=%s): %s -> self_corrected (reason: %s)",
            edge_id,
            old_status.value,
            reason,
        )
        return updated

    # -- sync lookups (used directly by the graph endpoint) ------------------

    def get_edge(self, edge_id: str) -> DependencyEdge:
        """Sync lookup. Raises NotFoundError if missing."""
        edge = self._edges.get(edge_id)
        if edge is None:
            raise NotFoundError(f"No edge registered with id {edge_id!r}", identifier=edge_id)
        return edge

    def list_edges_for_doi(self, doi: str) -> list[DependencyEdge]:
        """Sync lookup: all edges where from_doi == doi or to_doi == doi.

        An unparseable `doi` (fails `normalize_doi`) yields an empty list
        rather than raising, since this is used by a read-only graph
        endpoint where "no results" is a perfectly valid response.
        """
        try:
            normalized = normalize_doi(doi)
        except ValueError:
            logger.debug("list_edges_for_doi: %r does not normalize to a DOI; returning []", doi)
            return []
        return [
            edge for edge in self._edges.values() if edge.from_doi == normalized or edge.to_doi == normalized
        ]


_service_singleton: CogneeService | None = None


def get_cognee_service() -> CogneeService:
    """Module-level singleton accessor, used as a FastAPI dependency."""
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = CogneeService()
    return _service_singleton
