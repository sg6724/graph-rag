# Slide 1 — Title
MedGraph-RAG: safer, faster drug-interaction answers with GraphRAG + graph-aware semantic caching
Notes: one line on the problem — patients take multiple drugs; interactions are hidden across labels.

# Slide 2 — Problem
- Interaction facts are spread across different drug labels (multi-hop)
- Vector RAG retrieves similar text, not connected facts
- GraphRAG is accurate but slow/expensive; naive semantic caching is unsafe in medicine
Notes: show the simvastatin + clarithromycin → CYP3A4 example.

# Slide 3 — Architecture
Mermaid diagram from README.
Notes: walk left→right; stress that entity detection is deterministic (no LLM).

# Slide 4 — Novelty
1. Entity-keyed semantic cache (meaning + exact drug set)
2. Provenance-scoped invalidation (graph nodes → cached answers)
3. Resilient multi-provider routing on free tiers (per-task model chains, daily-quota aware)
Notes: "warfarin + aspirin" vs "warfarin + naproxen" example.

# Slide 5 — LIVE DEMO
Order: (1) simvastatin + clarithromycin in Compare mode → (2) "Biaxin while on Zocor?" → cache HIT
→ (3) warfarin + aspirin (HIT), then warfarin + naproxen → not served aspirin's answer → (4) Apply warfarin label
update → 5 evicted vs 16 kept. All steps replay from disk (verified with the network blocked).

# Slide 6 — Results
| Pipeline | Accuracy | Multi-hop | Single-hop |
|---|---|---|---|
| Vanilla RAG | 90% | 90% | 83% |
| GraphRAG | 95% | 95% | 100% |
| GraphRAG + cache (reworded) | 100% | 100% | 100% |
- Cache: 18/20 reworded questions in ~50 ms, 0/5 wrong-drug answers
- Threshold sweep: at 0.80 a plain cache serves 5 wrong-drug answers; with the drug-set key: 0
- Label update: 5 warfarin answers evicted, 16 others kept
Notes: LLM-judged; manual spot-check agreed 8/10 (both disagreements under-scored GraphRAG). Judge varies ±5 points,
so present GraphRAG vs vanilla as "slightly ahead + explainable", and the cache safety result as the headline.

# Slide 7 — Limitations & future work
- Scale to the full openFDA corpus; graph database (Neo4j) for millions of edges
- Patient context (age, kidney function) as graph nodes
- Clinician review loop for extracted edges
