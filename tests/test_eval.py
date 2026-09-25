import json

from eval.run_eval import judge, summarize, threshold_sweep, write_results
from medgraph.pipelines import Answer
from tests.conftest import FakeEmbedder, FakeRouter
from tests.test_pipelines import engine  # noqa: F401  (fixture reuse)

QS = [{"id": "m1", "type": "multi_hop", "question": "Can simvastatin be taken with clarithromycin?",
       "paraphrase": "Can Zocor be taken with Biaxin?", "reference": "avoid"},
      {"id": "s1", "type": "single_hop", "question": "Can I take aspirin with warfarin?",
       "paraphrase": "Aspirin plus warfarin okay?", "reference": "bleeding"}]
TRAPS = [{"id": "t1", "base_id": "s1", "question": "Can I take ibuprofen with warfarin?"}]


def ans(mode, q, hit=False, calls=1, ms=100.0):
    return Answer(mode=mode, question=q, text="x", cache_hit=hit, llm_calls=calls, timings={"total_ms": ms})


def test_judge_batches_and_parses():
    router = FakeRouter(replies=[json.dumps({"results": [{"id": "a", "score": 1}, {"id": "zzz", "score": 1}]})])
    scores = judge(router, [{"id": "a", "question": "q", "reference": "r", "answer": "x"}])
    assert scores == {"a": 1.0}


def test_judge_survives_bad_json():
    assert judge(FakeRouter(replies=["garbage"]), [{"id": "a", "question": "q", "reference": "r", "answer": "x"}]) == {}


def test_summarize_rows_and_cache_stats():
    answers = {
        "vanilla:m1": ans("vanilla", "q", ms=5000), "vanilla:s1": ans("vanilla", "q", ms=5000),
        "graphrag:m1": ans("graphrag", "q", ms=6000), "graphrag:s1": None,
        "cached_para:m1": ans("cached", "p", hit=True, calls=0, ms=20),
        "cached_para:s1": ans("cached", "p", hit=False, calls=1, ms=6000),
    }
    scores = {"vanilla:m1": 0.0, "vanilla:s1": 1.0, "graphrag:m1": 1.0, "cached_para:m1": 1.0, "cached_para:s1": 0.5}
    res = summarize(QS, answers, scores, [{"id": "t1", "hit": False}])
    rows = {r["pipeline"]: r for r in res["rows"]}
    assert rows["Vanilla RAG"]["accuracy"] == 0.5 and rows["Vanilla RAG"]["by_type"]["multi_hop"] == 0.0
    assert rows["GraphRAG"]["answered"] == 1 and rows["GraphRAG"]["accuracy"] == 1.0
    assert res["cache"]["paraphrase_hits"] == 1 and res["cache"]["avg_hit_ms"] == 20
    assert res["cache"]["trap_false_hits"] == 0


def test_threshold_sweep_entity_key_blocks_traps(engine):  # noqa: F811
    rows = threshold_sweep(engine, QS, TRAPS, [0.5])
    row = rows[0]
    assert row["trap_false_hits_with_key"] == 0
    assert row["trap_false_hits_without_key"] == 1


def test_write_results(tmp_path):
    res = {"rows": [{"pipeline": "GraphRAG", "accuracy": 0.9, "answered": 20, "by_type": {"multi_hop": 0.9},
                     "avg_ms": 6000.0, "llm_calls": 20}],
           "cache": {"paraphrase_hits": 15, "paraphrases": 20, "avg_hit_ms": 25.0, "avg_miss_ms": 6000.0,
                     "trap_false_hits": 0, "traps": 5, "cached_para_accuracy": 0.9},
           "sweep": [{"threshold": 0.9, "correct_hits_with_key": 15, "wrong_hits_with_key": 0,
                      "trap_false_hits_with_key": 0, "correct_hits_without_key": 16,
                      "wrong_hits_without_key": 1, "trap_false_hits_without_key": 3}],
           "scores": {}, "answers": {}}
    write_results(res, tmp_path)
    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "| GraphRAG | 90% |" in md and "15/20" in md
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["rows"][0]["llm_calls"] == 20
