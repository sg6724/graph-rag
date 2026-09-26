"""Per-session counters for the live metrics panel."""
from __future__ import annotations

from dataclasses import dataclass, field

from medgraph import config


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


@dataclass
class Metrics:
    records: list[dict] = field(default_factory=list)

    def record(self, a) -> None:
        self.records.append({"mode": a.mode, "hit": a.cache_hit,
                             "total_ms": a.timings.get("total_ms", 0.0), "llm_calls": a.llm_calls})

    def summary(self) -> dict:
        cached = [r for r in self.records if r["mode"] == "cached"]
        hits = [r for r in cached if r["hit"]]
        misses = [r for r in cached if not r["hit"]]
        return {
            "queries": len(self.records),
            "cache_queries": len(cached),
            "hits": len(hits),
            "hit_rate": len(hits) / len(cached) if cached else 0.0,
            "avg_hit_ms": _avg([r["total_ms"] for r in hits]),
            "avg_miss_ms": _avg([r["total_ms"] for r in misses]),
            "avg_answer_ms": _avg([r["total_ms"] for r in self.records if not r["hit"]]),  # any pipeline
            "llm_calls": sum(r["llm_calls"] for r in self.records),
            "llm_calls_saved": len(hits),
            "usd_saved": len(hits) * config.REF_USD_PER_CALL,
        }
