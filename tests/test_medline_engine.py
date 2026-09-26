import pytest

from medgraph.embeddings import VectorIndex
from medgraph.graph_store import GraphStore
from medgraph.pipelines import Engine
from medgraph.profiles import medline_profile
from medgraph.semantic_cache import SemanticCache
from tests.conftest import FakeEmbedder, FakeRouter

TEXTS = {
    "dengue:summary:0": ("dengue", "Dengue — You can get it if an infected mosquito bites you."),
    "malaria:summary:0": ("malaria", "Malaria — You get it when an infected mosquito bites you."),
    "lymedisease:summary:0": ("lyme disease", "Lyme Disease — You get it from the bite of an infected tick."),
    "mosquitobites:summary:0": ("mosquito bites", "Mosquito Bites — Mosquitoes can spread diseases."),
    "tickbites:summary:0": ("tick bites", "Tick Bites — Ticks can spread diseases."),
    "zikavirus:summary:0": ("zika virus", "Zika Virus — Zika spreads mainly through mosquito bites."),
}
ALIASES = {"mosquito": "mosquito bites", "tick": "tick bites", "break-bone fever": "dengue"}


def mention(src, tgt, cid):
    return {"source": src, "target": tgt, "type": "MENTIONS", "chunk_id": cid, "evidence": TEXTS[cid][1]}


@pytest.fixture
def engine():
    g = GraphStore()
    topics = [{"name": n, "type": "Topic"} for n, _ in TEXTS.values()] + [{"name": "infections", "type": "Group"}]
    g.add_extraction("dengue", topics, [mention("dengue", "mosquito bites", "dengue:summary:0")], "Topic")
    g.add_extraction("malaria", [], [mention("malaria", "mosquito bites", "malaria:summary:0")], "Topic")
    g.add_extraction("lyme disease", [], [mention("lyme disease", "tick bites", "lymedisease:summary:0")], "Topic")
    g.add_extraction("zika virus", [], [mention("zika virus", "mosquito bites", "zikavirus:summary:0")], "Topic")
    chunks = {cid: {"id": cid, "drug": ent, "section": "summary", "text": t} for cid, (ent, t) in TEXTS.items()}
    emb = FakeEmbedder()
    ids = list(chunks)
    index = VectorIndex(ids, emb.embed_docs([chunks[i]["text"] for i in ids]))
    router = FakeRouter(default="Dengue, malaria and Zika spread through mosquito bites [dengue:summary:0].")
    return Engine(router, g, chunks, index, emb, SemanticCache(threshold=0.9), profile=medline_profile(ALIASES))


def test_graph_search_collects_every_topic_linked_to_the_seed(engine):
    a = engine.graphrag("Which diseases spread through mosquitoes?")
    assert a.entities == ["mosquito bites"]
    assert {"dengue", "malaria", "zika virus"} <= set(a.path_nodes)
    assert "lyme disease" not in a.path_nodes
    prompt = engine.router.prompts[-1][1]
    assert "MedlinePlus" in prompt and "clinical pharmacology" not in prompt
    assert "malaria --MENTIONS--> mosquito bites" in prompt


def test_cache_key_is_the_topic_set(engine):
    engine.cache.threshold = 0.5
    engine.cached("Which diseases spread through mosquitoes?")
    same = engine.cached("Which diseases are spread through mosquitoes?")
    assert same.cache_hit and same.served_from == "Which diseases spread through mosquitoes?"
    other = engine.cached("Which diseases spread through ticks?")
    assert not other.cache_hit and other.blocked_by_key == "Which diseases spread through mosquitoes?"


def test_topic_update_links_new_text_without_llm_and_evicts_only_dependents(engine):
    engine.cached("Which diseases spread through mosquitoes?")
    engine.cached("Which diseases spread through ticks?")
    calls = len(engine.router.prompts)
    res = engine.update_label("malaria", "summary", "Malaria can also be confused with dengue.")
    assert len(engine.router.prompts) == calls  # deterministic linking, no LLM call
    assert res["chunk_id"] == "malaria:summary:1"  # malaria already has chunk 0
    assert engine.graph.g.has_edge("malaria", "dengue")
    assert res["evicted"] == ["Which diseases spread through mosquitoes?"]
    assert res["retained"] == ["Which diseases spread through ticks?"]
