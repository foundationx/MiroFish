# MiroFish with MemPalace (local graph memory) and xAI Grok

This branch (`mempalace-grok`) lets MiroFish run **without Zep Cloud**. Graph memory
uses [MemPalace](https://github.com/MemPalace/mempalace) (MIT, local ChromaDB + SQLite
knowledge graph, local MiniLM embeddings). All chat/LLM calls go to xAI Grok through its
OpenAI-compatible API. Zep is still the default, and nothing changes unless you set
`GRAPH_MEMORY_BACKEND=mempalace`.

## How it fits together

```
services/* ──► app/graph_memory.get_graph_memory_client()
                 ├─ GRAPH_MEMORY_BACKEND=zep        → zep_cloud.Zep (unchanged)
                 └─ GRAPH_MEMORY_BACKEND=mempalace  → MemPalaceGraphMemory (Zep-shaped facade)
```

`app/graph_memory/mempalace_backend.py` implements the subset of the Zep client that
MiroFish uses (`graph.create/get/delete/set_ontology/add/search`,
`graph.node.*`, `graph.edge.*`, `graph.episode.get`, `batch.create/add/process/get/list/list_items`).
That keeps the services, API routes and frontend unchanged.

Each `graph_id` gets its own directory under `MEMPALACE_DATA_DIR/graphs/<graph_id>/`:

| file | contents |
| --- | --- |
| `meta.json` | name, description, ontology |
| `kg.sqlite3` | MemPalace `KnowledgeGraph` (entities + temporal triples: `valid_from`/`valid_to`) |
| `mirofish.sqlite3` | sidecar: node type/summary/attributes, edge fact text + episode links, episodes |
| `palace/` | MemPalace palace (ChromaDB) with rooms `episodes`, `facts`, `nodes` |

**Ingestion.** Each text chunk or simulation-activity batch becomes an episode. An LLM
call (`app/graph_memory/extraction.py`) turns it into entities and facts. The prompt is
constrained by the project's ontology, and the output is normalized to the allowed entity
types, relation names and attributes, falling back to `Entity` / `RELATED_TO`. Entities
are upserted into the KG and sidecar, facts become KG triples, and episode, fact and node
text are written to the palace for semantic search.

**Search.** `graph.search` combines MemPalace semantic search (MiniLM, local) with a boost
for KG name matches, scoped to edges or nodes. The `as_of` keyword filters facts by their
validity window. Node/edge listing reads the KG plus the sidecar.

**Batches.** `batch.process` runs locally on a background thread pool
(`MEMPALACE_EXTRACT_WORKERS`). Results are persisted under `MEMPALACE_DATA_DIR/batches/`,
and the existing poll loop sees `succeeded` / `partial` / `failed`. Every sqlite write
from the worker threads is serialized under a per-graph lock.

## Setup

```bash
cd backend
uv sync --python 3.12          # installs mempalace>=3.9 (tested with 3.10.0) + chromadb
cd ..
cp .env.mempalace-grok.example .env   # .env is git-ignored
export XAI_API_KEY=...                # never put it in .env (see below)
scripts/mempalace_grok_backend.sh     # backend on 127.0.0.1:5001
npm run setup && npm run dev          # optional: frontend, as upstream
```

The first search downloads the MiniLM ONNX model (~80 MB) to `~/.cache/chroma`.
After that, embeddings run offline.

### Headless end-to-end run

```bash
backend/.venv/bin/python scripts/e2e_mempalace_grok.py --rounds 12
```

This uploads `examples/mempalace_grok/sample_seed_product_launch.md`, a fictional SAMPLE
product launch, then runs ontology → graph build → personas/config → simulation (with
graph-memory updates) → stop → report. Output goes to
`backend/uploads/e2e_runs/<sim_id>_report.md` and `<sim_id>_summary.json`. Pass
`--stop-after graph|prepare` for partial runs.

### Offline smoke test (no API spend)

`scripts/mock_openai_server.py` is a tiny OpenAI-compatible mock. Point
`LLM_BASE_URL=http://127.0.0.1:5098/v1` at it to drive the real HTTP flow:
upload → ontology → MemPalace graph build → personas → simulation start/stop → report.
The content it produces is meaningless (the simulation does 0 actions and the report is
a stub). It only checks the wiring.

## Environment variables

| var | value | notes |
| --- | --- | --- |
| `GRAPH_MEMORY_BACKEND` | `zep` (default) or `mempalace` | with `mempalace`, `ZEP_API_KEY` is not required |
| `MEMPALACE_DATA_DIR` | path | default `backend/uploads/mempalace` |
| `MEMPALACE_EXTRACT_WORKERS` | int, default 4 | parallel extraction calls during batch processing |
| `LLM_BASE_URL` | `https://api.x.ai/v1` | |
| `LLM_MODEL_NAME` | `grok-4.3` | ontology, extraction, personas, config, report agent |
| `LLM_BOOST_BASE_URL` / `LLM_BOOST_MODEL_NAME` | `https://api.x.ai/v1` / `grok-4.20-0309-non-reasoning` | used by the OASIS simulation agents when the boost key is also set |
| `LLM_API_KEY`, `LLM_BOOST_API_KEY` | **not in .env** | the launcher exports both from `$XAI_API_KEY` at runtime |

`config.py` loads `.env` with `override=True`, so any key written in `.env` would beat the
environment. Leave both keys out of the file.

**Model choice.** At the time of writing, xAI's docs list `grok-4.7` as the flagship
($2 in / $6 out per 1M tokens) and `grok-4.3` / `grok-4.20-0309-*` at $1.25 / $2.50,
with tool calling and structured outputs on the Grok 4 family. MiroFish makes many calls,
so the cheaper tier is the default. Swap in `grok-4.7` for better reports. These model
IDs come from the docs and could not be checked against `/v1/models`, because the account
returned 403 (no credits) during development.

**Rough cost** for the SAMPLE run (12 rounds, ~15–25 agents, both platforms): about
1.5–2.5M input tokens and 0.1–0.2M output tokens. That's roughly **$2–6** at
grok-4.3 / 4.20 prices, less with cached input. The OASIS agent turns are the bulk of it.
This is an estimate, not a measurement.

## Limitations / differences from Zep

- **No cross-encoder reranker.** `reranker=` is accepted and ignored. Ranking is cosine
  similarity plus a KG name-match boost.
- **Extraction costs LLM calls.** It takes one call per chunk at build time and one per
  simulation-activity batch when graph-memory updates are on. Zep did this server-side.
- **Weaker entity resolution.** Entities merge by normalized name, so "Dana Okafor" and
  "Okafor" end up as two nodes. There's no LLM dedup and no summary rewriting beyond
  appending.
- **`expired_at` is always `None`.** `valid_at`/`invalid_at` come only from dates stated
  in the text, and nothing auto-invalidates contradicted facts.
- **`graph.add` and batch items report "processed" even when extraction failed** (the
  error is stored on the episode and logged). This keeps the existing poll loops moving.
- **Batches are in-process threads.** A backend restart mid-batch leaves that batch
  `processing` forever; rebuild the graph.
- **Single-process only.** The per-graph lock doesn't cover several backend processes
  sharing one `MEMPALACE_DATA_DIR`.
- **Tests use a mock LLM** (`backend/tests/test_mempalace_graph_memory.py`). Extraction
  quality with real Grok output hasn't been measured yet.
- **The live Grok E2E run hasn't been done.** The xAI team behind the key returned
  `403 permission-denied` (credits exhausted / spending limit) during development. Run
  `scripts/e2e_mempalace_grok.py` once credits are available.
- **Licensing.** MiroFish is AGPL-3.0, so a hosted deployment of this fork must offer
  its source. MemPalace is MIT.
