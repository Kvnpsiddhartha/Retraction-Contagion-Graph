# Retraction Contagion Graph

> **Cognee-powered memory that tracks how a retracted paper's false claims silently spread through the literature.**

When a scientific paper is retracted, the paper itself gets a "RETRACTED" stamp — but every other paper that already **cited it and built a claim on top of it** keeps circulating with no warning, and nobody is tracking that chain. This tool maps that contagion, hop by hop.

---

## The Problem

A 2012 paper gets retracted for fabricated data in 2019. Between 2012–2019, 40 other papers cited it and used its result as a supporting claim ("as shown by X, this effect is well established…"). Those 40 papers are never flagged. Their readers, and the papers that cite *them*, have no idea they're standing on a retracted foundation.

This is a named, studied phenomenon in meta-science — **"citation contagion"** or **"zombie citations"** — and real researchers currently do this analysis manually, paper by paper, because no tool automates the multi-hop reasoning.

---

## Architecture

```
retraction-graph/
  shared/
    schemas.py           # Pydantic models + API contract  (Phase 1)
    config.py            # Settings + logging setup         (Phase 1)
    exceptions.py        # Typed exception hierarchy        (Phase 1)
  data/
    retraction_watch.py  # Retraction Watch CSV download + seed selection (Phase 2a)
    external_apis.py     # Crossref + Semantic Scholar clients            (Phase 2b)
  memory/
    cognee_service.py    # remember/recall/improve/forget wrapper          (Phase 2c)
  pipeline/
    document_builder.py  # Turns raw data into MemoryDocuments             (Phase 3)
    ingest_runner.py     # Orchestrates full ingestion run                 (Phase 4)
  api/
    main.py              # FastAPI app — 6 routes                          (Phase 5)
  frontend/
    index.html           # Single-file demo UI (HTML + CSS + JS)           (Phase 2d)
  scripts/
    seed_demo.py         # End-to-end demo seeding + narrative moments     (Phase 6)
  DEMO.md                # Presenter runbook
  README.md              # This file
  .env.example           # Config template
  requirements.txt       # Python dependencies
```

**Data flow:**

```
Retraction Watch CSV  ──► retraction_watch.py ──► seed papers
                                                       │
Crossref + Semantic Scholar ──► external_apis.py ──► citations + context
                                                       │
                                             document_builder.py
                                             (classify, build edges)
                                                       │
                                        cognee_service.py (remember)
                                                       │
                                    FastAPI /api/* ◄───┘
                                          │
                                  frontend/index.html
```

**Why Cognee is load-bearing here (not decorative):**

| Cognee operation | What it does here |
|---|---|
| `remember()` | Ingests retracted-paper metadata, citing-paper abstracts, and citation-context sentences into a hybrid graph-vector store |
| `recall()` | Answers "which papers treat this retracted claim as established fact?" — requires multi-hop graph traversal, not just similarity ranking |
| `improve()` | Reviewer marks a flagged dependency as confirmed or false-positive; graph gets more precise over time |
| `forget()` | Once a downstream paper self-corrects, its exposure flag is removed — memory evolves as the literature does |

---

## Data Sources

All free, no signup required:

| Source | What we get | Access |
|---|---|---|
| **Retraction Watch DB** (via Crossref/GitLab) | ~50,000 retractions: title, DOI, reason, dates | GitLab CSV clone, no key needed |
| **Crossref REST API** | Metadata for any DOI (title, authors, abstract) | `https://api.crossref.org/works/{doi}` — add `mailto=` to be polite |
| **Semantic Scholar API** | Citing papers for any DOI, with citation context/intent | `https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}/citations` — free, no key |

No synthetic data anywhere — every node in the graph is a real DOI, a real retraction reason, a real citation.

---

## Local Dev Setup

### 1. Clone and set up a virtual environment

```bash
git clone <repo-url>
cd retraction-graph
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure (optional)

```bash
cp .env.example .env
# Edit .env to set CROSSREF_MAILTO to your real email
# (improves Crossref API priority — not required for the demo to work)
```

### 4. Seed the demo data

```bash
python scripts/seed_demo.py
```

This downloads the Retraction Watch CSV, ingests 6 seed retractions, builds the citation-contagion graph, and prints the exact edge IDs and DOI you'll use in the live demo. Copy those values out.

**Options:**

```bash
python scripts/seed_demo.py --seed-count 10        # more seed papers
python scripts/seed_demo.py --force-refresh-csv    # re-download the CSV
```

### 5. Start the backend

```bash
uvicorn api.main:app --port 8000
```

Swagger UI is available at `http://localhost:8000/docs`.

### 6. Open the frontend

```bash
# Option A — open directly (works for most browsers):
open frontend/index.html

# Option B — serve locally if your browser blocks file:// fetch:
python -m http.server 8080 --directory frontend
# then visit http://localhost:8080
```

See [DEMO.md](DEMO.md) for the full presenter runbook.

---

## API Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe (`cognee_ready` bool) |
| `POST` | `/api/ingest` | Trigger a fresh ingestion run |
| `POST` | `/api/recall` | Natural-language query over Cognee |
| `POST` | `/api/improve` | Record confirmed / false_positive verdict on an edge |
| `POST` | `/api/forget` | Mark an edge as self-corrected |
| `GET` | `/api/graph/{doi}` | Return the dependency subgraph anchored at a DOI |

---

## Explicitly Out of Scope for the MVP

The following are known gaps, acknowledged before the demo, not surprises:

- **Full-text PDF parsing** — abstracts + citation-context sentences from Semantic Scholar are sufficient to prove the concept; full-text ingestion would improve precision but is not needed for the demo.
- **Automatic detection of new retractions** — the current pipeline must be re-run manually; real-time monitoring (polling Crossref/Retraction Watch) is a natural next step.
- **Unbounded multi-hop traversal** — the pipeline walks exactly two hops (direct citer → citer-of-a-citer). A general N-hop walk is architecturally simple to add but would produce a graph too large to demo clearly and was deliberately excluded.
- **Multi-worker / persistent edge state** — the edge-review state (`confirmed`, `flagged`, etc.) lives in-process. Running `uvicorn --workers N > 1` would give each worker its own copy. A real deployment would move this state to a shared store (Redis / Postgres).
- **Authentication / authorization** — the API is fully open; `improve()` and `forget()` can be called by anyone. Acceptable for a hackathon demo, not for production.

---

## Judging Criteria Mapping

| Criterion | This project |
|---|---|
| **Potential Impact** | Real, named problem in research integrity; directly usable by journal editors, research-integrity offices, grant reviewers |
| **Creativity** | Graph-contagion framing not in hackathon example list; multi-hop dependency is novel |
| **Technical Excellence** | Real multi-source data pipeline (Retraction Watch + Semantic Scholar + Crossref), typed schemas, bounded retries, idempotent ingestion |
| **Best Use of Cognee** | All 4 lifecycle ops functionally necessary; the core answer (multi-hop dependency) is impossible with plain vector search |
| **UX** | Graph visualization of "here's how far this lie traveled" is inherently compelling |
| **Presentation** | Problem → graph-traversal-proof → extension arc; see DEMO.md |
