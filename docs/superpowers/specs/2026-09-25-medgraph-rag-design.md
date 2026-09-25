# MedGraph-RAG: GraphRAG for Drug Interactions with Graph-Aware Semantic Caching

**Date:** 2026-09-25
**Deadline:** 2026-09-26 (college-wide competition demo)
**Judged on:** live demo, innovation/novelty, technical depth, presentation/report

## 1. Goal

Build a working GraphRAG system over real FDA drug labels that answers drug-interaction
questions — including multi-hop ones plain RAG misses — and pair it with a
**graph-aware semantic cache** that (a) never confuses questions about different drugs and
(b) invalidates only the cached answers affected by a changed source document.

Success = a reliable live demo + a results table showing:
- GraphRAG beats vanilla RAG on multi-hop questions (accuracy).
- The cache cuts latency and LLM requests on paraphrased questions without wrong hits.
- Targeted invalidation: after a label update, only dependent answers are evicted.

Non-goals (cut for the 1-day deadline): Neo4j/graph DB server, community detection /
global search, auth, multi-user, cloud deployment, more than one domain.

## 2. Data

- Source: openFDA drug label API (`https://api.fda.gov/drug/label.json`), public domain, no key.
- ~30 commonly co-prescribed drugs with well-known interactions, e.g.: warfarin, aspirin,
  ibuprofen, naproxen, clopidogrel, simvastatin, atorvastatin, clarithromycin, erythromycin,
  ketoconazole, fluconazole, sertraline, fluoxetine, tramadol, metformin, lisinopril,
  spironolactone, digoxin, amiodarone, omeprazole, rifampin, carbamazepine, lithium,
  methotrexate, allopurinol, sildenafil, nitroglycerin, levothyroxine, prednisone, acetaminophen.
- Sections kept per label: `drug_interactions`, `contraindications`, `warnings_and_cautions`
  (or `warnings`), `clinical_pharmacology` (for enzyme/metabolism facts).
- Raw JSON saved to `data/raw/<drug>.json`; chunked text to `data/chunks.jsonl`
  (chunk = {id, drug, section, text}), ~800 chars with overlap.

## 3. Knowledge graph

- In-memory NetworkX `MultiDiGraph`, persisted to `data/graph.json`.
- Node types: `Drug`, `DrugClass`, `Enzyme`, `Condition`, `SideEffect`.
- Edge types: `INHIBITS`, `INDUCES`, `METABOLIZED_BY`, `INTERACTS_WITH`,
  `CONTRAINDICATED_IN`, `CAUSES`, `BELONGS_TO`.
- Every edge carries provenance: `chunk_id`, `drug` (source label), `evidence` (short quote).
- Extraction: LLM ("quality" task, see Section 6) → strict JSON `{entities:[...], relations:[...]}`,
  validated against the allowed types; invalid items dropped. Entity names normalized
  (lowercase, alias map for enzymes like "CYP 3A4" → "cyp3a4").
- **Quota-aware batching:** one extraction call covers one or more whole drug labels (all its
  chunks, with chunk ids inline so relations keep provenance), packed up to ~60K input tokens.
  Target: ~10–30 extraction calls for the whole corpus.

## 4. Retrieval & answering

Three pipelines behind one interface `answer(question) -> Answer{text, citations, path_nodes, path_edges, timings, provider}`:

1. **Vanilla RAG:** embed question → top-k (k=6) chunks by cosine → LLM answer with citations.
   (All LLM answering uses the `answer` task profile from Section 6.)
2. **GraphRAG:**
   - Extract drug/entity mentions from the question **deterministically, with no LLM call**:
     word-boundary + fuzzy matching against graph node names and an alias map
     (brand names, e.g. "Coumadin" → warfarin). Zero quota, instant, and stable — which the
     cache's `entity_key` depends on.
   - Seed nodes = matched entities (fallback: nodes from top vector chunks).
   - Expand up to 2 hops; find all simple paths (≤3 edges) between seed drug pairs
     (captures drug→enzyme←drug mechanisms).
   - Context = serialized path triples + their evidence chunks (deduplicated, capped).
   - LLM answer instructed to explain the mechanism and cite chunk ids.
3. **GraphRAG + cache:** pipeline 2 wrapped by the cache (Section 5).

Embeddings: local `fastembed` (BAAI/bge-small-en-v1.5, CPU), stored in `data/embeddings.npy`.

## 5. Graph-aware semantic cache (the novel contribution)

- Entry: `{query_text, query_embedding, entity_key, answer, node_ids, created_at, graph_version}`.
- **Entity-aware key:** `entity_key` = sorted tuple of normalized drug entities in the query.
  A hit requires `entity_key` exact match AND cosine similarity ≥ threshold (default 0.90).
  Prevents "warfarin + aspirin" returning the "warfarin + ibuprofen" answer.
- **Provenance-scoped invalidation:** each entry stores the graph node ids its answer used.
  When a source label is updated → re-chunk & re-extract that drug only → diff nodes/edges →
  evict every cache entry whose `node_ids` intersect the changed nodes. Others stay valid.
- Storage: in-memory list + numpy matrix, persisted to `data/cache.json`. Brute-force cosine
  (cache is small; no vector DB needed).
- Metrics recorded per query: hit/miss, latency, LLM requests used, estimated $ saved
  (priced at a reference paid-model rate for illustration).

## 6. LLM router

Single module `llm_router` used by every LLM call:

```
disk cache (sha256 of task+prompt) → hit: return
→ 1. Gemini   models/gemini-3.8-flash      (primary)
→ 2. OpenRouter qwen/qwen3.8-27b:free       (fallback on 429 / 5xx / timeout / empty output)
→ 3. Gemini   models/gemini-3.5-flash      (last resort; separate per-model free quota)
→ record which provider/model served the call
```

- Task profiles:
  - `extract` (offline graph build, judging): default thinking, JSON output, long timeout.
  - `answer` (live queries): Gemini `thinkingLevel: "low"` (measured ~6 s vs ~17 s default;
    `"minimal"` is rejected by the API), Qwen with reasoning disabled.
- Measured 2026-09-25: Qwen free returned 429 (upstream saturation) with 0/50 daily quota used —
  the fallback chain is required, not optional.
- Clients: Gemini via REST `generateContent`; OpenRouter via the OpenAI-compatible
  `/chat/completions`. Both through `httpx`, no vendor SDK needed.
- Config via env vars: `OPENROUTER_API_KEY`, `GEMINI_API_KEY` (read from process env, falling
  back to the Windows user environment in the registry since `setx` doesn't update open shells);
  model ids in `config.py`.
- Quota: OpenRouter free = 50 requests/day on this account (1,000/day after a one-time $10
  credit purchase — optional insurance).
- The disk cache (`data/llm_cache/`) is separate from the semantic cache: it exists to save
  quota during development and to make the demo reproducible offline.

## 7. App (Streamlit)

- **Chat panel:** question box, answer with citations (drug + section), mode selector
  (Vanilla RAG / GraphRAG / GraphRAG + Cache), "cache HIT" badge, provider used.
- **Graph panel:** pyvis graph of the retrieved subgraph with the answer path highlighted.
- **Metrics panel:** hit rate, avg latency hit vs miss, LLM requests saved, $ saved.
- **Invalidation demo button:** "Simulate FDA label update for <drug>" → applies a prepared
  edit to that drug's label, re-extracts it, shows evicted vs retained cache entries.
- Persistent disclaimer: "Educational demo — not medical advice."

## 8. Benchmark

- `eval/questions.json`: 20 questions (6 single-hop, 10 multi-hop/mechanism, 4 contraindication),
  each with a reference answer written from the labels, plus 1 paraphrase each (40 total).
- Also 5 "trap" pairs (same wording, different drug) to measure false cache hits (target 0).
- Runner compares the 3 pipelines: accuracy (one batched LLM-judge call + manual spot-check of 10),
  latency, LLM requests. The cached pipeline's misses reuse GraphRAG's answers via the LLM disk
  cache (identical prompts), so the budget is ~85 LLM calls total.
  Output: `eval/results.md` table + `eval/results.json`.
- Latency for cache hits/misses is measured live; "requests saved" and "$ saved" are computed
  from the recorded call counts.

## 9. Error handling

- Every LLM call goes through the router (retries + fallback). Extraction JSON parse failure →
  one retry with a repair prompt, then skip chunk (logged).
- openFDA fetch failure for a drug → skip that drug, log it.
- Demo safety: graph, embeddings, LLM disk cache, and a pre-warmed semantic cache are committed
  as artifacts so the demo runs even with no network.

## 10. Project layout

```
medgraph/
  config.py        # models, paths, thresholds
  llm_router.py    # provider chain + disk cache
  ingest.py        # openFDA fetch + chunking
  extract.py       # LLM entity/relation extraction → graph
  graph_store.py   # NetworkX graph load/save/query/diff
  embeddings.py    # fastembed wrapper + vector index
  pipelines.py     # vanilla, graphrag, cached
  semantic_cache.py
  metrics.py
app.py             # Streamlit UI
eval/run_eval.py, eval/questions.json
tests/             # unit tests for cache, graph diff/invalidation, router fallback
data/              # generated artifacts
```

## 11. Deliverables for the demo

1. Working Streamlit app with pre-built data.
2. `eval/results.md` results table.
3. README (architecture diagram, how to run, results).
4. Slide outline (6–8 slides): problem → GraphRAG → cache novelty → live demo → results → future work.
