# MedGraph-RAG benchmark results

| Pipeline | Accuracy | Multi-hop | Single-hop | Contraindication | Avg latency | LLM calls | Answered |
|---|---|---|---|---|---|---|---|
| Vanilla RAG | 90% | 90% | 83% | 100% | 14.52 s | 20 | 20 |
| GraphRAG | 95% | 95% | 100% | 88% | 14.09 s | 20 | 20 |
| GraphRAG + semantic cache (paraphrases) | 100% | 100% | 100% | 100% | 2.95 s | 2 | 20 |

## Semantic cache

- Paraphrase cache hits: **18/20**
- Avg latency: hit **52 ms** vs miss **29.02 s**
- Accuracy of answers served on paraphrases: **100%**
- Wrong-drug trap questions served another drug's answer: **0/5** (same-drug reuse: 0)

## Threshold sweep (offline, no LLM calls)

Each probe (20 paraphrases + traps) is matched to its most similar cached question.

| Threshold | Correct hits (key) | Wrong-drug hits (key) | Same-drug reuse (key) | Correct hits (no key) | Wrong-drug hits (no key) | Same-drug reuse (no key) |
|---|---|---|---|---|---|---|
| 0.80 | 19 | 0 | 0 | 19 | 5 | 0 |
| 0.85 | 18 | 0 | 0 | 18 | 3 | 0 |
| 0.90 | 15 | 0 | 0 | 15 | 0 | 0 |
| 0.95 | 7 | 0 | 0 | 7 | 0 | 0 |

_Latency for LLM answers is the original generation time (recorded even when replayed from the dev disk cache); cache-hit latency is measured live._
