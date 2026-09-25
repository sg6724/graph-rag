"""Single entry point for every LLM call: disk cache + provider fallback chain."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from medgraph import config


class LLMError(RuntimeError):
    """Every provider in the chain failed."""


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    cached: bool
    latency_ms: float      # wall time of this call (tiny when served from disk)
    gen_latency_ms: float  # time the original generation took (for honest benchmarks)


# (provider, model, prompt, task, json_mode) -> text
Transport = Callable[[str, str, str, str, bool], str]


def get_key(name: str) -> str | None:
    """Env var first; then the Windows user environment (setx doesn't update open shells)."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, name)[0]
    except (ImportError, OSError):
        return None


def _gemini(model: str, prompt: str, task: str, json_mode: bool) -> str:
    key = get_key("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    gen: dict = {}
    if json_mode:
        gen["responseMimeType"] = "application/json"
    if task == "answer":
        gen["thinkingConfig"] = {"thinkingLevel": "low"}
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}
    r = httpx.post(f"{config.GEMINI_URL}/{model}:generateContent",
                   headers={"x-goog-api-key": key}, json=body, timeout=config.TIMEOUTS[task])
    r.raise_for_status()
    parts = r.json()["candidates"][0]["content"].get("parts", [])
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def _openrouter(model: str, prompt: str, task: str, json_mode: bool) -> str:
    key = get_key("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    body: dict = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if task == "answer":
        body["reasoning"] = {"enabled": False}
    r = httpx.post(config.OPENROUTER_URL, headers={"Authorization": f"Bearer {key}"},
                   json=body, timeout=config.TIMEOUTS[task])
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content") or ""


def default_transport(provider: str, model: str, prompt: str, task: str, json_mode: bool) -> str:
    fn = {"gemini": _gemini, "openrouter": _openrouter}[provider]
    return fn(model, prompt, task, json_mode)


class Router:
    def __init__(self, providers=None, cache_dir: Path | None = None,
                 transport: Transport | None = None, retry_pause_s: float = 5.0, rounds: int = 4,
                 sleep: Callable[[float], None] = time.sleep):
        self.providers = providers or config.PROVIDERS
        self.cache_dir = Path(cache_dir or config.LLM_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.transport = transport or default_transport
        self.retry_pause_s = retry_pause_s
        self.rounds = rounds  # full passes over the chain; free tiers have short demand spikes
        self.sleep = sleep
        self.calls = 0
        self.cache_hits = 0
        self.by_provider: dict[str, int] = {}
        self.errors: list[str] = []

    def _path(self, prompt: str, task: str, json_mode: bool) -> Path:
        digest = hashlib.sha256(f"{task}|{json_mode}|{prompt}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def complete(self, prompt: str, task: str = "answer", json_mode: bool = False) -> LLMResult:
        path = self._path(prompt, task, json_mode)
        t0 = time.perf_counter()
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            self.cache_hits += 1
            return LLMResult(d["text"], d["provider"], d["model"], True,
                             (time.perf_counter() - t0) * 1000, d.get("gen_latency_ms", 0.0))
        for attempt in range(self.rounds):
            for provider, model in self.providers:
                t_call = time.perf_counter()
                try:
                    text = self.transport(provider, model, prompt, task, json_mode)
                except Exception as e:  # any provider failure → next in chain
                    self.errors.append(f"{provider}/{model}: {type(e).__name__}: {e}"[:300])
                    continue
                if not text or not text.strip():
                    self.errors.append(f"{provider}/{model}: empty output")
                    continue
                gen_ms = (time.perf_counter() - t_call) * 1000
                self.calls += 1
                label = f"{provider}/{model}"
                self.by_provider[label] = self.by_provider.get(label, 0) + 1
                path.write_text(json.dumps({"text": text, "provider": provider, "model": model,
                                            "gen_latency_ms": gen_ms}), encoding="utf-8")
                return LLMResult(text, provider, model, False,
                                 (time.perf_counter() - t0) * 1000, gen_ms)
            if attempt < self.rounds - 1:
                self.sleep(self.retry_pause_s * 3 ** attempt)  # 5 s, 15 s, 45 s
        raise LLMError("all providers failed: " + " | ".join(self.errors[-len(self.providers):]))


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_object(text: str) -> dict:
    """Extract the outermost JSON object from an LLM reply (tolerates fences and prose)."""
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
