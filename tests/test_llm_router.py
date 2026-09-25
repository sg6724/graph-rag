import pytest

from medgraph.llm_router import LLMError, Router, parse_json_object

PROV = [("gemini", "g1"), ("openrouter", "q1"), ("gemini", "g2")]


def make(transport, tmp_path):
    return Router(providers=PROV, cache_dir=tmp_path, transport=transport, retry_pause_s=0)


def test_primary_success(tmp_path):
    seen = []

    def t(provider, model, prompt, task, json_mode):
        seen.append(model)
        return "hi"

    r = make(t, tmp_path).complete("q")
    assert (r.text, r.provider, r.model, r.cached) == ("hi", "gemini", "g1", False)
    assert seen == ["g1"]


def test_falls_back_on_error_and_empty_output(tmp_path):
    def t(provider, model, prompt, task, json_mode):
        if model == "g1":
            raise RuntimeError("429 Too Many Requests")
        if model == "q1":
            return "   "
        return "ok"

    router = make(t, tmp_path)
    r = router.complete("q")
    assert r.model == "g2"
    assert router.by_provider == {"gemini/g2": 1}
    assert len(router.errors) == 2


def test_disk_cache_hit_skips_transport(tmp_path):
    n = [0]

    def t(*args):
        n[0] += 1
        return "x"

    router = make(t, tmp_path)
    first = router.complete("q")
    second = router.complete("q")
    assert second.cached and second.text == "x"
    assert second.gen_latency_ms == pytest.approx(first.gen_latency_ms)
    assert n[0] == 1 and router.cache_hits == 1 and router.calls == 1


def test_cache_key_depends_on_task_and_json_mode(tmp_path):
    n = [0]

    def t(*args):
        n[0] += 1
        return "{}"

    router = make(t, tmp_path)
    router.complete("q", task="answer")
    router.complete("q", task="extract")
    router.complete("q", task="extract", json_mode=True)
    assert n[0] == 3


def test_all_fail_raises(tmp_path):
    def t(*args):
        raise RuntimeError("down")

    with pytest.raises(LLMError):
        make(t, tmp_path).complete("q")


def test_parse_json_object_strips_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go: {"a": [1, 2]} thanks') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_retries_whole_chain_with_backoff_until_a_provider_recovers(tmp_path):
    rounds = [0]
    pauses = []

    def t(provider, model, prompt, task, json_mode):
        if model == "g1":
            rounds[0] += 1
            if rounds[0] < 3:  # overloaded for the first two rounds
                raise RuntimeError("503 high demand")
            return "recovered"
        raise RuntimeError("429")

    router = Router(providers=PROV, cache_dir=tmp_path, transport=t, retry_pause_s=1.0, rounds=4,
                    sleep=pauses.append)
    r = router.complete("q")
    assert r.text == "recovered" and r.model == "g1"
    assert pauses == [1.0, 3.0]  # backoff x3 between rounds


def test_providers_can_differ_per_task(tmp_path):
    seen = []

    def t(provider, model, prompt, task, json_mode):
        seen.append((task, model))
        return "ok"

    chains = {"answer": [("gemini", "fast")], "extract": [("openrouter", "big")]}
    router = Router(providers=chains, cache_dir=tmp_path, transport=t, retry_pause_s=0)
    router.complete("a", task="answer")
    router.complete("b", task="extract")
    assert seen == [("answer", "fast"), ("extract", "big")]


def test_model_with_exhausted_daily_quota_is_skipped_afterwards(tmp_path):
    calls = []

    def t(provider, model, prompt, task, json_mode):
        calls.append(model)
        if model == "g1":
            raise RuntimeError("429 quota GenerateRequestsPerDayPerProjectPerModel-FreeTier exceeded")
        return "ok"

    router = make(t, tmp_path)
    router.complete("first")
    router.complete("second")
    assert calls == ["g1", "q1", "q1"]  # g1 not retried once its daily quota is gone
    assert "gemini/g1" in router.disabled


def test_unknown_task_falls_back_to_answer_chain(tmp_path):
    seen = []

    def t(provider, model, prompt, task, json_mode):
        seen.append(model)
        return "ok"

    chains = {"answer": [("gemini", "fast")], "extract": [("openrouter", "big")]}
    Router(providers=chains, cache_dir=tmp_path, transport=t, retry_pause_s=0).complete("x", task="extract_fast")
    assert seen == ["fast"]


def test_gemini_thinking_low_for_all_but_offline_extract(monkeypatch):
    import medgraph.llm_router as lr

    bodies = []

    class Resp:
        status_code = 200

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

    monkeypatch.setattr(lr, "get_key", lambda name: "k")
    monkeypatch.setattr(lr.httpx, "post", lambda url, headers, json, timeout: bodies.append((json, timeout)) or Resp())
    lr._gemini("m", "p", "extract_fast", True)
    lr._gemini("m", "p", "extract", True)
    assert bodies[0][0]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert "thinkingConfig" not in bodies[1][0]["generationConfig"]
    assert bodies[0][1] < bodies[1][1]  # fast task gets the short timeout
