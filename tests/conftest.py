import hashlib
import re

import numpy as np

from medgraph.llm_router import LLMResult


class FakeRouter:
    """Stands in for Router: returns queued replies (or a default), records prompts."""

    def __init__(self, replies=None, default="ok"):
        self.replies = list(replies or [])
        self.default = default
        self.prompts: list[tuple[str, str]] = []
        self.by_provider: dict[str, int] = {}

    def complete(self, prompt, task="answer", json_mode=False):
        self.prompts.append((task, prompt))
        reply = self.replies.pop(0) if self.replies else self.default
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(text=reply, provider="fake", model="fake-1", cached=False,
                         latency_ms=5.0, gen_latency_ms=5.0)


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (unit-normalized)."""

    dim = 64

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_query(self, text):
        return self._vec(text)

    def embed_docs(self, texts):
        return np.stack([self._vec(t) for t in texts])
