# MedGraph-RAG benchmark results

| Pipeline | Accuracy | Multi-hop | Single-hop | Contraindication | Avg latency | LLM calls | Answered |
|---|---|---|---|---|---|---|---|
| Vanilla RAG | 90% | 90% | 92% | 88% | 5.58 s | 20 | 20 |
| GraphRAG | 90% | 90% | 92% | 88% | 2.05 s | 20 | 20 |
| GraphRAG + semantic cache (paraphrases) | 95% | 90% | 100% | 100% | 0.52 s | 5 | 20 |

## Semantic cache

- Paraphrase cache hits: **15/20**
- Avg latency: hit **36 ms** vs miss **1.96 s**
- Accuracy of answers served on paraphrases: **95%**
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
