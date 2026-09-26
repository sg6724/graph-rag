"""Drives the real Streamlit app headlessly (streamlit.testing AppTest) through the 5 demo steps.

Live: loads the real graph/index/embedder and may call LLMs on a disk-cache miss.
Run: uv run pytest tests/test_app_live.py -m live -s
"""
import pytest
from streamlit.testing.v1 import AppTest


def ask(at, question, mode_label):
    at.sidebar.radio[0].set_value(mode_label)
    at.text_input(key="question").set_value(question)
    at.button[0].click()  # form submit ("Ask") is the first button in the main area
    at.run(timeout=300)
    assert not at.exception, at.exception


def texts(at):
    parts = [m.value for m in at.markdown] + [e.value for e in at.success] + [e.value for e in at.info] + \
            [e.value for e in at.warning] + [e.value for e in at.error]
    return "\n".join(str(p) for p in parts)


@pytest.mark.live
def test_demo_walkthrough(monkeypatch):
    monkeypatch.setenv("MEDGRAPH_BACKEND", "files")  # never mutate the real Supabase cache from a test
    at = AppTest.from_file("../app.py", default_timeout=300)
    at.run()
    at.sidebar.selectbox(key="dataset").set_value("fda")
    at.run()
    assert not at.exception, at.exception
    assert "not medical advice" in at.caption[0].value

    # 1. GraphRAG + cache: the demo cache is pre-warmed with the 20 benchmark questions → served from cache
    ask(at, "Can a patient taking simvastatin start clarithromycin?", "cached")
    assert "cyp3a4" in texts(at).lower()
    first_answer = at.session_state["last"].text
    # metrics must reflect this question in the same run (not one click late)
    hit_rate = [m for m in at.sidebar.metric if m.label == "Cache hit rate"][0]
    assert hit_rate.delta == "1/1", hit_rate.delta

    # 2. Brand-name paraphrase → HIT, 0 LLM calls, same answer
    ask(at, "Is it safe to take Biaxin while on Zocor?", "cached")
    last = at.session_state["last"]
    assert last.cache_hit and last.llm_calls == 0 and last.text == first_answer, last.cache_score
    print("paraphrase hit score", round(last.cache_score, 3), "latency ms", round(last.timings["total_ms"]))

    # 3. Same wording, different drug → never served warfarin+aspirin's answer
    ask(at, "Can a patient taking warfarin also take aspirin?", "cached")
    assert at.session_state["last"].cache_hit  # pre-warmed
    ask(at, "Can a patient taking warfarin also take naproxen?", "cached")
    last = at.session_state["last"]
    assert not last.cache_hit
    print("trap blocked_by_key:", last.blocked_by_key, "| score", last.cache_score)

    # 4. Label update → warfarin answers evicted, simvastatin answer kept
    at.sidebar.selectbox(key="update_pick").set_value(0)
    at.sidebar.button[0].click()  # "Apply update"
    at.run(timeout=300)
    assert not at.exception, at.exception
    log = at.session_state["update_log"]
    print("update:", log)
    assert "Can a patient taking warfarin also take aspirin?" in log["evicted"]
    assert "Can a patient taking simvastatin start clarithromycin?" in log["retained"]

    # 5. Compare mode renders both pipelines
    ask(at, "Can a patient taking simvastatin start clarithromycin?", "compare")
    v, g = at.session_state["compare"]
    assert v.mode == "vanilla" and g.mode == "graphrag" and "cyp3a4" in g.path_nodes
    s = at.session_state["metrics"].summary()
    print("metrics:", s)
    assert s["hits"] >= 1
