"""Semantic answer cache keyed by (query meaning, exact drug set), invalidated by graph provenance."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from medgraph import config


@dataclass
class CacheEntry:
    query: str
    embedding: list[float]
    entity_key: str
    answer: dict
    node_ids: list[str]
    created_at: float

    def vec(self) -> np.ndarray:
        return np.asarray(self.embedding, dtype=np.float32)


class SemanticCache:
    def __init__(self, threshold: float = config.CACHE_THRESHOLD, path: Path | None = None):
        self.threshold = threshold
        self.path = Path(path) if path else None
        self.entries: list[CacheEntry] = []
        if self.path and self.path.exists():
            self.load(self.path)

    def lookup(self, qvec: np.ndarray, entity_key: str) -> tuple[CacheEntry | None, float]:
        best, best_score = None, -1.0
        for e in self.entries:
            if e.entity_key != entity_key:
                continue
            s = float(np.dot(qvec, e.vec()))
            if s > best_score:
                best, best_score = e, s
        if best is not None and best_score >= self.threshold:
            return best, best_score
        return None, best_score

    def nearest(self, qvec: np.ndarray) -> tuple[CacheEntry | None, float]:
        best, best_score = None, -1.0
        for e in self.entries:
            s = float(np.dot(qvec, e.vec()))
            if s > best_score:
                best, best_score = e, s
        return best, best_score

    def store(self, query: str, qvec: np.ndarray, entity_key: str, answer: dict,
              node_ids: list[str]) -> CacheEntry:
        entry = CacheEntry(query, [float(x) for x in qvec], entity_key, answer, list(node_ids), time.time())
        self.entries.append(entry)
        self._autosave()
        return entry

    def invalidate_nodes(self, nodes: set[str]) -> list[CacheEntry]:
        evicted = [e for e in self.entries if set(e.node_ids) & nodes]
        self.entries = [e for e in self.entries if not set(e.node_ids) & nodes]
        self._autosave()
        return evicted

    def clear(self) -> None:
        self.entries = []
        self._autosave()

    def _autosave(self) -> None:
        if self.path:
            self.save(self.path)

    def save(self, path: Path | None = None) -> None:
        target = Path(path or self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps([asdict(e) for e in self.entries]), encoding="utf-8")

    def load(self, path: Path) -> None:
        self.entries = [CacheEntry(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]
