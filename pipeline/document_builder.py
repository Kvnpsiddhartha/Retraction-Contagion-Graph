"""
pipeline/document_builder.py — turns raw 02a/02b outputs into the two
artifacts the memory layer needs:

    * `MemoryDocument`s  — natural-language text blocks for Cognee to ingest.
    * `DependencyEdge`s  — our structured, typed relationship graph.

This module owns the only piece of "not just a citation list" logic in the
project: `classify_relation`, the heuristic that decides whether a citation
is structurally load-bearing (`relies_on`) or incidental
(`cites_in_passing`); and the bounded second-hop walk in
`build_contagion_graph`, which proves the tool reasons about *transitive*
exposure, not just direct citation counts.

Non-functional invariants (see 00-architecture-overview.md §5):
    * No network calls happen directly in this file — all I/O goes through
      the imported `data/external_apis.py` functions. This file is
      orchestration + pure transforms only.
    * No silent failures: every degrade-and-continue path is logged.
    * Idempotent: documents are deduped by `doc_id`, edges by `id`, so
      re-running the pipeline over the same seed set never produces
      duplicate graph entries.
"""

from __future__ import annotations

import logging

from data.external_apis import fetch_semantic_scholar_citations, parse_citations_to_models
from shared.exceptions import ExternalAPIError
from shared.schemas import (
    CitationContext,
    CitationIntent,
    CitingPaper,
    DependencyEdge,
    EdgeStatus,
    MemoryDocType,
    MemoryDocument,
    RelationType,
    RetractedPaper,
    RetractionReason,
    normalize_doi,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# classify_relation() confidence constants
#
# These are starting points for the MVP heuristic, not tuned values — kept
# as named constants (rather than magic numbers inline) because `improve()`
# feedback is meant to eventually replace/retune this function.
# --------------------------------------------------------------------------

CONF_RESULT_STRONG = 0.8  # RESULT intent + a substantive context sentence
CONF_RESULT_WEAK = 0.5  # RESULT intent but no/short context sentence (undocumented-by-spec fallback, still load-bearing signal)
CONF_METHOD = 0.6  # METHOD intent — the citing paper builds on the method itself
CONF_BACKGROUND = 0.7  # BACKGROUND intent — confidently NOT load-bearing
CONF_UNKNOWN_NO_CONTEXT = 0.3  # No signal at all — weak "in passing" default
CONF_UNKNOWN_WITH_CONTEXT = 0.45  # Ambiguous but worth flagging for human review

# Minimum word count for a context sentence to count as "substantive" rather
# than a throwaway fragment (e.g. a bare citation marker with no surrounding
# text extracted).
_MIN_SUBSTANTIVE_WORDS = 3

# Human-readable phrasing for each retraction reason, used when building the
# retracted-paper document content. Deliberately short and information-dense
# rather than a verbatim dump of the enum value.
_REASON_PHRASES: dict[RetractionReason, str] = {
    RetractionReason.FABRICATION: "fabricated data",
    RetractionReason.FALSIFIED_DATA: "falsified data",
    RetractionReason.PLAGIARISM: "plagiarism",
    RetractionReason.ETHICAL_VIOLATION: "an ethical violation",
    RetractionReason.ERROR: "an honest error",
    RetractionReason.OTHER: "reasons not otherwise classified",
}


# --------------------------------------------------------------------------
# Pure transforms
# --------------------------------------------------------------------------


def classify_relation(context: CitationContext) -> tuple[RelationType, float]:
    """Decide whether a citation is structurally load-bearing.

    Pure heuristic, no I/O, no logging side effects required for
    correctness. Exhaustive over `CitationIntent`:

        RESULT   + substantive context_sentence -> (RELIES_ON, 0.8)
        RESULT   + no/short context_sentence     -> (RELIES_ON, 0.5)
        METHOD                                   -> (RELIES_ON, 0.6)
        BACKGROUND                               -> (CITES_IN_PASSING, 0.7)
        UNKNOWN  + context_sentence is None      -> (CITES_IN_PASSING, 0.3)
        UNKNOWN  + context_sentence is not None  -> (RELIES_ON, 0.45)

    Args:
        context: the citation context to classify.

    Returns:
        A `(RelationType, confidence)` tuple, confidence in [0.0, 1.0].
    """
    sentence = context.context_sentence  # already blank-to-None normalized by the schema

    if context.intent == CitationIntent.RESULT:
        if sentence is not None and len(sentence.split()) > _MIN_SUBSTANTIVE_WORDS:
            return RelationType.RELIES_ON, CONF_RESULT_STRONG
        return RelationType.RELIES_ON, CONF_RESULT_WEAK

    if context.intent == CitationIntent.METHOD:
        return RelationType.RELIES_ON, CONF_METHOD

    if context.intent == CitationIntent.BACKGROUND:
        return RelationType.CITES_IN_PASSING, CONF_BACKGROUND

    # CitationIntent.UNKNOWN (the only remaining member)
    if sentence is None:
        return RelationType.CITES_IN_PASSING, CONF_UNKNOWN_NO_CONTEXT
    return RelationType.RELIES_ON, CONF_UNKNOWN_WITH_CONTEXT


def make_edge_id(from_doi: str, to_doi: str) -> str:
    """Deterministic edge id: `f'{from_doi}__{to_doi}'` over normalized DOIs.

    This is the single source of truth for edge-id format; `memory/
    cognee_service.py` stores whatever id it is given, so any drift here
    would silently break dedup on re-ingestion.

    Args:
        from_doi: the citing (dependent) paper's DOI.
        to_doi: the cited (depended-upon) paper's DOI.

    Returns:
        The deterministic edge id string.
    """
    return f"{normalize_doi(from_doi)}__{normalize_doi(to_doi)}"


def build_retracted_paper_document(paper: RetractedPaper) -> MemoryDocument:
    """Build the Cognee-ingestible document for a retracted paper.

    Args:
        paper: the retracted paper.

    Returns:
        A `MemoryDocument` with an information-dense natural-language
        paragraph as `content` (not a field dump).
    """
    reason_phrase = _REASON_PHRASES.get(paper.retraction_reason, paper.retraction_reason_raw)
    date_phrase = paper.retraction_date.isoformat() if paper.retraction_date else "an unspecified date"

    content = (
        f"\"{paper.title}\" (DOI: {paper.doi}) was retracted on {date_phrase} "
        f"due to {reason_phrase} (recorded reason: \"{paper.retraction_reason_raw}\"). "
        f"This paper is a RETRACTED source and any claim relying on it should be "
        f"treated as unsupported."
    )

    return MemoryDocument(
        doc_id=f"paper::{paper.doi}",
        doc_type=MemoryDocType.RETRACTED_PAPER,
        content=content,
        metadata={
            "doi": paper.doi,
            "type": "retracted_paper",
            "reason": paper.retraction_reason.value,
        },
    )


def build_citing_paper_document(
    paper: CitingPaper,
    context: CitationContext,
    relation: RelationType,
) -> MemoryDocument:
    """Build the Cognee-ingestible document for a citing paper.

    Args:
        paper: the citing paper.
        context: the citation context linking `paper` to what it cites.
        relation: the classification produced by `classify_relation`.

    Returns:
        A `MemoryDocument` describing the paper and, specifically, how it
        relates to the paper it cites.
    """
    relation_phrase = relation.value.replace("_", " ")

    if context.context_sentence:
        context_phrase = f' It cites the source with the sentence: "{context.context_sentence}"'
    else:
        context_phrase = " No citation-context sentence was available for this reference."

    content = (
        f'"{paper.title}" (DOI: {paper.doi}) cites DOI {context.cited_doi}.'
        f"{context_phrase} Based on the citation context, this reference is "
        f"classified as {relation_phrase} the cited work."
    )

    return MemoryDocument(
        doc_id=f"paper::{paper.doi}",
        doc_type=MemoryDocType.CITING_PAPER,
        content=content,
        metadata={
            "doi": paper.doi,
            "type": "citing_paper",
            "cites": context.cited_doi,
            "relation": relation.value,
        },
    )


def build_dependency_edge(context: CitationContext, hop_depth: int = 1) -> DependencyEdge:
    """Combine `classify_relation` + `make_edge_id` into a `DependencyEdge`.

    Args:
        context: the citation context to convert into an edge. The edge
            points from `context.citing_doi` (depends on) to
            `context.cited_doi` (depended upon).
        hop_depth: how many hops this edge is from the original retracted
            paper (1 = direct citer, 2 = citer-of-a-citer, etc). Defaults
            to 1; callers doing the second-hop walk must pass 2 explicitly.

    Returns:
        A `DependencyEdge` with `status=EdgeStatus.FLAGGED`.
    """
    relation, confidence = classify_relation(context)
    edge_id = make_edge_id(context.citing_doi, context.cited_doi)

    return DependencyEdge(
        id=edge_id,
        from_doi=context.citing_doi,
        to_doi=context.cited_doi,
        relation=relation,
        confidence=confidence,
        hop_depth=hop_depth,
        status=EdgeStatus.FLAGGED,
        evidence_text=context.context_sentence,
    )


def build_edge_document(edge: DependencyEdge) -> MemoryDocument:
    """Build the Cognee-ingestible document describing a dependency edge.

    Args:
        edge: the edge to describe.

    Returns:
        A `MemoryDocument` with a one-sentence plain-English description.
    """
    relation_phrase = edge.relation.value.replace("_", " ")
    content = f"Paper {edge.from_doi} {relation_phrase} paper {edge.to_doi} (confidence {edge.confidence})."

    return MemoryDocument(
        doc_id=f"edge::{edge.id}",
        doc_type=MemoryDocType.DEPENDENCY_EDGE,
        content=content,
        metadata={
            "edge_id": edge.id,
            "from_doi": edge.from_doi,
            "to_doi": edge.to_doi,
            "hop_depth": edge.hop_depth,
        },
    )


# --------------------------------------------------------------------------
# Dedup helpers (idempotent-ingestion invariant)
# --------------------------------------------------------------------------


def _merge_documents(documents: list[MemoryDocument]) -> list[MemoryDocument]:
    """Dedupe by `doc_id`, last-write-wins. Logs the collapsed count."""
    by_id: dict[str, MemoryDocument] = {}
    for doc in documents:
        by_id[doc.doc_id] = doc

    n_duplicates = len(documents) - len(by_id)
    if n_duplicates:
        logger.info("Collapsed %d duplicate document(s) by doc_id.", n_duplicates)

    return list(by_id.values())


def _merge_edges(edges: list[DependencyEdge]) -> list[DependencyEdge]:
    """Dedupe by `id`. On collision, keep the edge with the SMALLER
    `hop_depth` (closer to the original retraction is more informative),
    per the explicit edge case in the spec — this is not left to
    incidental dict insertion order. Ties (equal hop_depth) resolve
    last-write-wins. Logs the collapsed count.
    """
    by_id: dict[str, DependencyEdge] = {}
    n_duplicates = 0

    for edge in edges:
        existing = by_id.get(edge.id)
        if existing is None:
            by_id[edge.id] = edge
            continue

        n_duplicates += 1
        if edge.hop_depth < existing.hop_depth:
            by_id[edge.id] = edge
        # else: keep `existing` (either it's already closer, or it's an
        # exact-hop-depth tie and we deliberately keep the first-seen one
        # rather than silently flipping on iteration order).

    if n_duplicates:
        logger.info("Collapsed %d duplicate edge(s) by id (kept smaller hop_depth on conflict).", n_duplicates)

    return list(by_id.values())


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _process_citation_batch(
    cited_doi: str,
    hop_depth: int,
) -> tuple[list[MemoryDocument], list[DependencyEdge], list[CitingPaper]]:
    """Fetch + parse + build documents/edges for everything citing `cited_doi`.

    Shared by both the hop-1 and hop-2 walks so the build steps (classify,
    build edge, build documents) aren't duplicated across the two loops.

    Returns:
        `(documents, edges, citing_papers)` — `citing_papers` is returned
        separately (not just embedded in metadata) so the caller can select
        second-hop candidates without re-parsing anything.

    Raises:
        ExternalAPIError: propagated from `fetch_semantic_scholar_citations`
            after retries are exhausted; callers decide how to degrade.
    """
    raw_citations = fetch_semantic_scholar_citations(cited_doi)
    citing_papers, contexts = parse_citations_to_models(cited_doi, raw_citations)

    documents: list[MemoryDocument] = []
    edges: list[DependencyEdge] = []

    if not citing_papers:
        logger.info("No citers found for %s.", normalize_doi(cited_doi))
        return documents, edges, citing_papers

    for citing_paper, context in zip(citing_papers, contexts):
        edge = build_dependency_edge(context, hop_depth=hop_depth)
        edges.append(edge)
        documents.append(build_citing_paper_document(citing_paper, context, edge.relation))
        documents.append(build_edge_document(edge))

    return documents, edges, citing_papers


def build_contagion_graph(
    seed_papers: list[RetractedPaper],
    max_second_hop_papers: int = 3,
) -> tuple[list[MemoryDocument], list[DependencyEdge]]:
    """Orchestrate the full contagion-graph build for the given seed papers.

    Steps:
        1. Build a `retracted_paper` document for every seed paper.
        2. Direct (hop-1) citers: fetch + classify + build documents/edges
           for each seed paper's citers. A seed paper whose fetch fails
           (`ExternalAPIError`) is logged and skipped; the rest continue.
        3. Bounded second hop: take up to `max_second_hop_papers` of the
           hop-1 citing papers whose edge relation is `RELIES_ON` (the
           interesting ones) and fetch/classify/build hop-2 edges pointing
           from a third paper -> that hop-1 paper (never -> the original
           retracted paper). A candidate's own fetch failure is logged and
           skipped without affecting the other candidates.
        4. Dedupe documents by `doc_id` and edges by `id` before returning.

    Args:
        seed_papers: the retracted papers to build the graph from.
        max_second_hop_papers: cap on how many hop-1 RELIES_ON papers get a
            second-hop walk, to keep the demo graph small. If fewer
            candidates exist than this cap, all of them are used — no
            padding, no error.

    Returns:
        `(all_documents, all_edges)`, both deduplicated.
    """
    if max_second_hop_papers < 0:
        raise ValueError(f"max_second_hop_papers must be >= 0, got {max_second_hop_papers}")

    all_documents: list[MemoryDocument] = []
    all_edges: list[DependencyEdge] = []

    # Hop-1 RELIES_ON candidates for the second-hop walk, keyed by doi to
    # dedupe (e.g. the same paper reliant-citing two different seeds) while
    # preserving first-seen order.
    hop1_reliant_candidates: dict[str, CitingPaper] = {}

    # --- Step 1 + 2: seed documents + direct (hop-1) citers -------------
    for seed in seed_papers:
        all_documents.append(build_retracted_paper_document(seed))

        try:
            documents, edges, citing_papers = _process_citation_batch(seed.doi, hop_depth=1)
        except ExternalAPIError as exc:
            logger.error(
                "Failed to fetch citations for seed paper %s; skipping this seed and continuing. %s",
                seed.doi,
                exc,
            )
            continue

        all_documents.extend(documents)
        all_edges.extend(edges)

        # Track RELIES_ON hop-1 citers as second-hop candidates.
        citing_by_doi = {paper.doi: paper for paper in citing_papers}
        for edge in edges:
            if edge.relation == RelationType.RELIES_ON and edge.from_doi in citing_by_doi:
                hop1_reliant_candidates.setdefault(edge.from_doi, citing_by_doi[edge.from_doi])

    # --- Step 3: bounded second hop --------------------------------------
    second_hop_selection = list(hop1_reliant_candidates.values())[:max_second_hop_papers]

    for hop1_paper in second_hop_selection:
        try:
            documents, edges, _third_papers = _process_citation_batch(hop1_paper.doi, hop_depth=2)
        except ExternalAPIError as exc:
            logger.error(
                "Failed to fetch second-hop citations for %s; skipping this candidate and continuing. %s",
                hop1_paper.doi,
                exc,
            )
            continue

        all_documents.extend(documents)
        all_edges.extend(edges)

    # --- Step 4: dedupe ---------------------------------------------------
    return _merge_documents(all_documents), _merge_edges(all_edges)
