import numpy as np

from medgraph.semantic_cache import SemanticCache


def v(*xs):
    a = np.array(xs, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_hit_on_similar_query_same_key():
    c = SemanticCache(threshold=0.9)
    c.store("q1", v(1, 0.1), "aspirin|warfarin", {"text": "A"}, ["aspirin", "warfarin"])
    entry, score = c.lookup(v(1, 0.12), "aspirin|warfarin")
    assert entry is not None and entry.answer == {"text": "A"} and score > 0.99


def test_different_entity_key_never_hits():
    c = SemanticCache(threshold=0.9)
    c.store("q1", v(1, 0), "aspirin|warfarin", {"text": "A"}, ["warfarin"])
    entry, score = c.lookup(v(1, 0), "ibuprofen|warfarin")
    assert entry is None and score == -1.0
    near, s = c.nearest(v(1, 0))
    assert near.query == "q1" and s > 0.99


def test_miss_below_threshold():
    c = SemanticCache(threshold=0.9)
    c.store("q1", v(1, 0), "k", {}, [])
    entry, score = c.lookup(v(1, 1), "k")
    assert entry is None and 0.7 < score < 0.72


def test_invalidate_only_entries_touching_nodes():
    c = SemanticCache()
    c.store("sim+clari", v(1, 0), "a", {}, ["simvastatin", "cyp3a4", "clarithromycin"])
    c.store("warf+asp", v(0, 1), "b", {}, ["warfarin", "aspirin"])
    evicted = c.invalidate_nodes({"warfarin", "metronidazole"})
    assert [e.query for e in evicted] == ["warf+asp"]
    assert [e.query for e in c.entries] == ["sim+clari"]


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "cache.json"
    c = SemanticCache(path=p)
    c.store("q", v(1, 0), "k", {"text": "A"}, ["n"])
    d = SemanticCache(path=p)
    assert len(d.entries) == 1 and d.lookup(v(1, 0), "k")[0].answer == {"text": "A"}


def test_in_memory_cache_can_load_and_save_elsewhere(tmp_path):
    p = tmp_path / "warm.json"
    SemanticCache(path=p).store("q", v(1, 0), "k", {}, [])
    mem = SemanticCache()
    mem.load(p)
    mem.clear()
    assert len(SemanticCache(path=p).entries) == 1  # clearing in-memory copy leaves the file alone
    mem.store("q2", v(0, 1), "k", {}, [])
    mem.save(tmp_path / "out.json")
    assert len(SemanticCache(path=tmp_path / "out.json").entries) == 1
