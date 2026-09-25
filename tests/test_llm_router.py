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
