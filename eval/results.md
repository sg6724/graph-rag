# MedGraph-RAG benchmark results

| Pipeline | Accuracy | Multi-hop | Single-hop | Contraindication | Avg latency | LLM calls | Answered |
|---|---|---|---|---|---|---|---|
| Vanilla RAG | 95% | 90% | 100% | 100% | 5.58 s | 20 | 20 |
| GraphRAG | 88% | 100% | 67% | 88% | 3.27 s | 20 | 20 |
| GraphRAG + semantic cache (paraphrases) | 92% | 95% | 83% | 100% | 0.51 s | 5 | 20 |

## Semantic cache

- Paraphrase cache hits: **15/20**
- Avg latency: hit **39 ms** vs miss **1.92 s**
- Accuracy of answers served on paraphrases: **92%**
- Wrong-drug trap questions served from cache: **2/5**

## Threshold sweep (offline, no LLM calls)

| Threshold | Correct hits (key) | Wrong hits (key) | Trap hits (key) | Correct hits (no key) | Wrong hits (no key) | Trap hits (no key) |
|---|---|---|---|---|---|---|
| 0.80 | 19 | 0 | 2 | 19 | 0 | 5 |
| 0.85 | 18 | 0 | 2 | 18 | 0 | 4 |
| 0.90 | 15 | 0 | 2 | 15 | 0 | 2 |
| 0.95 | 7 | 0 | 0 | 7 | 0 | 0 |

_Latency for LLM answers is the original generation time (recorded even when replayed from the dev disk cache); cache-hit latency is measured live._
