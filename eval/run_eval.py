"""Benchmark: vanilla RAG vs GraphRAG vs GraphRAG + semantic cache, plus a threshold sweep."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medgraph import config
from medgraph.entities import canonicalize, drug_key, match_entities
from medgraph.llm_router import LLMError, parse_json_object

HERE = Path(__file__).parent
QUESTIONS_PATH = HERE / "questions.json"
PIPELINES = [("vanilla", "Vanilla RAG"), ("graphrag", "GraphRAG"),
             ("cached_para", "GraphRAG + semantic cache (paraphrases)")]

JUDGE_PROMPT = """You grade answers from a drug-interaction QA system against reference answers written from FDA labels.
Score each item:
- 1   = same clinical verdict as the reference AND states its key mechanism or risk
- 0.5 = verdict right but the mechanism/risk is missing or partly wrong
- 0   = wrong verdict, says the information is unavailable, or contradicts the reference
Return JSON only: {"results": [{"id": "...", "score": 0 | 0.5 | 1, "reason": "<= 15 words"}]}

Items:
"""


def _avg(xs):
    return sum(xs) / len(xs) if xs else None


def judge(router, items: list[dict], batch: int = 10) -> dict[str, float]:
    scores: dict[str, float] = {}
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        ids = {it["id"] for it in chunk}
        try:
            res = router.complete(JUDGE_PROMPT + json.dumps(chunk, ensure_ascii=False, indent=1),
                                  task="extract", json_mode=True)
            data = parse_json_object(res.text)
        except (ValueError, LLMError) as e:
            print(f"judge batch {i // batch} failed: {e}")
            continue
        for r in data.get("results", []):
            if isinstance(r, dict) and r.get("id") in ids:
                try:
                    scores[r["id"]] = float(r["score"])
                except (TypeError, ValueError):
                    pass
    return scores


def summarize(questions, answers, scores, traps) -> dict:
    qtype = {q["id"]: q["type"] for q in questions}
    types = sorted(set(qtype.values()))
    rows = []
    for key, label in PIPELINES:
        keys = [f"{key}:{q['id']}" for q in questions if answers.get(f"{key}:{q['id']}") is not None]
        judged = [k for k in keys if k in scores]
        by_type = {t: _avg([scores[k] for k in judged if qtype[k.split(":")[1]] == t]) for t in types}
        rows.append({"pipeline": label, "answered": len(keys),
                     "accuracy": _avg([scores[k] for k in judged]), "by_type": by_type,
                     "avg_ms": _avg([answers[k].timings.get("total_ms", 0.0) for k in keys]),
                     "llm_calls": sum(answers[k].llm_calls for k in keys)})
    para = [answers[f"cached_para:{q['id']}"] for q in questions if answers.get(f"cached_para:{q['id']}")]
    hits = [a for a in para if a.cache_hit]
    misses = [a for a in para if not a.cache_hit]
    cache = {"paraphrase_hits": len(hits), "paraphrases": len(para),
             "avg_hit_ms": _avg([a.timings["total_ms"] for a in hits]),
             "avg_miss_ms": _avg([a.timings["total_ms"] for a in misses]),
             "trap_false_hits": sum(1 for t in traps if t["hit"]), "traps": len(traps),
             "cached_para_accuracy": rows[2]["accuracy"]}
    return {"rows": rows, "cache": cache}


def threshold_sweep(engine, questions, traps, thresholds) -> list[dict]:
    """Offline (no LLM): how many paraphrases/traps would hit, with vs without the entity key."""
    types = engine.graph.node_types()

    def enc(text):
        return engine.embedder.embed_query(canonicalize(text)), drug_key(match_entities(text, types), types)

    base = {q["id"]: enc(q["question"]) for q in questions}
    ids = list(base)
    matrix = np.stack([base[i][0] for i in ids])
    probes = [(q["id"], *enc(q["paraphrase"]), False) for q in questions] + \
             [(t["base_id"], *enc(t["question"]), True) for t in traps]
    rows = []
    for th in thresholds:
        row = {"threshold": th}
        for use_key in (True, False):
            correct = wrong = trap_hits = 0
            for own_id, vec, key, is_trap in probes:
                sims = matrix @ vec
                if use_key:
                    sims = np.where([base[i][1] == key for i in ids], sims, -np.inf)
                j = int(np.argmax(sims))
                if not np.isfinite(sims[j]) or sims[j] < th:
                    continue
                if is_trap:
                    trap_hits += 1
                elif ids[j] == own_id:
                    correct += 1
                else:
                    wrong += 1
            suffix = "with_key" if use_key else "without_key"
            row[f"correct_hits_{suffix}"] = correct
            row[f"wrong_hits_{suffix}"] = wrong
            row[f"trap_false_hits_{suffix}"] = trap_hits
        rows.append(row)
    return rows


def _pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def write_results(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["# MedGraph-RAG benchmark results", "",
             "| Pipeline | Accuracy | Multi-hop | Single-hop | Contraindication | Avg latency | LLM calls | Answered |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results["rows"]:
        bt = r["by_type"]
        avg = "n/a" if r["avg_ms"] is None else f"{r['avg_ms'] / 1000:.2f} s"
        lines.append(f"| {r['pipeline']} | {_pct(r['accuracy'])} | {_pct(bt.get('multi_hop'))} | "
                     f"{_pct(bt.get('single_hop'))} | {_pct(bt.get('contraindication'))} | {avg} | "
                     f"{r['llm_calls']} | {r['answered']} |")
    c = results["cache"]
    hit_ms = "n/a" if c["avg_hit_ms"] is None else f"{c['avg_hit_ms']:.0f} ms"
    miss_ms = "n/a" if c["avg_miss_ms"] is None else f"{c['avg_miss_ms'] / 1000:.2f} s"
    lines += ["", "## Semantic cache", "",
              f"- Paraphrase cache hits: **{c['paraphrase_hits']}/{c['paraphrases']}**",
              f"- Avg latency: hit **{hit_ms}** vs miss **{miss_ms}**",
              f"- Accuracy of answers served on paraphrases: **{_pct(c['cached_para_accuracy'])}**",
              f"- Wrong-drug trap questions served from cache: **{c['trap_false_hits']}/{c['traps']}**",
              "", "## Threshold sweep (offline, no LLM calls)", "",
              "| Threshold | Correct hits (key) | Wrong hits (key) | Trap hits (key) | "
              "Correct hits (no key) | Wrong hits (no key) | Trap hits (no key) |",
              "|---|---|---|---|---|---|---|"]
    for s in results.get("sweep", []):
        lines.append(f"| {s['threshold']:.2f} | {s['correct_hits_with_key']} | {s['wrong_hits_with_key']} | "
                     f"{s['trap_false_hits_with_key']} | {s['correct_hits_without_key']} | "
                     f"{s['wrong_hits_without_key']} | {s['trap_false_hits_without_key']} |")
    lines += ["", "_Latency for LLM answers is the original generation time (recorded even when replayed from "
              "the dev disk cache); cache-hit latency is measured live._"]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe(fn, q):
    try:
        return fn(q)
    except LLMError as e:
        print(f"  LLM failure on {q!r}: {e}")
        return None


def main() -> None:
    from medgraph.pipelines import load_engine

    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions, traps_def = data["questions"], data["traps"]
    engine = load_engine(cache_path=None)
    answers = {}
    for q in questions:
        print(f"[{q['id']}] {q['question']}", flush=True)
        answers[f"vanilla:{q['id']}"] = _safe(engine.vanilla, q["question"])
        answers[f"graphrag:{q['id']}"] = _safe(engine.graphrag, q["question"])
        answers[f"cached_first:{q['id']}"] = _safe(engine.cached, q["question"])  # warms cache; replays graphrag prompt
    for q in questions:
        a = _safe(engine.cached, q["paraphrase"])
        answers[f"cached_para:{q['id']}"] = a
        print(f"[{q['id']}] paraphrase -> {'HIT' if a and a.cache_hit else 'MISS'}"
              f" ({a.cache_score if a else None})", flush=True)
    traps = []
    for t in traps_def:
        a = _safe(engine.cached, t["question"])
        traps.append({"id": t["id"], "hit": bool(a and a.cache_hit)})
    refs = {q["id"]: q["reference"] for q in questions}
    items = [{"id": k, "question": a.question, "reference": refs[k.split(":")[1]], "answer": a.text}
             for k, a in answers.items() if a is not None and k.split(":")[0] in ("vanilla", "graphrag", "cached_para")]
    scores = judge(engine.router, items)
    results = summarize(questions, answers, scores, traps)
    results["sweep"] = threshold_sweep(engine, questions, traps_def, [0.80, 0.85, 0.90, 0.95])
    results["scores"] = scores
    results["answers"] = {k: a.to_dict() for k, a in answers.items() if a is not None}
    results["llm_providers"] = engine.router.by_provider
    write_results(results, HERE)
    engine.cache.save(config.CACHE_PATH)  # pre-warmed cache for the live demo
    print((HERE / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
