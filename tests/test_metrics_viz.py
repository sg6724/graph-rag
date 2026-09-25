import pytest

from medgraph.metrics import Metrics
from medgraph.pipelines import Answer
from medgraph.viz import subgraph_html


def test_metrics_summary():
    m = Metrics()
    m.record(Answer(mode="cached", question="a", text="", cache_hit=False, llm_calls=1, timings={"total_ms": 6000}))
    m.record(Answer(mode="cached", question="b", text="", cache_hit=True, llm_calls=0, timings={"total_ms": 20}))
    m.record(Answer(mode="vanilla", question="c", text="", llm_calls=1, timings={"total_ms": 5000}))
    s = m.summary()
    assert s["queries"] == 3 and s["cache_queries"] == 2 and s["hits"] == 1
    assert s["hit_rate"] == 0.5 and s["avg_hit_ms"] == 20 and s["avg_miss_ms"] == 6000
    assert s["llm_calls"] == 2 and s["llm_calls_saved"] == 1 and s["usd_saved"] == pytest.approx(0.01)


def test_metrics_empty():
    assert Metrics().summary()["hit_rate"] == 0.0


def test_subgraph_html_contains_nodes_and_relations():
    html = subgraph_html({"simvastatin": "Drug", "cyp3a4": "Enzyme", "clarithromycin": "Drug"},
                         ["simvastatin", "cyp3a4", "clarithromycin"],
                         [{"source": "clarithromycin", "target": "cyp3a4", "type": "INHIBITS", "evidence": "strong"}],
                         ["simvastatin", "clarithromycin"])
    assert "<html" in html.lower()
    for token in ("simvastatin", "cyp3a4", "INHIBITS"):
        assert token in html
