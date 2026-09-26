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


def test_symptom_question_ranks_topics_linked_to_all_symptoms(engine):
    g = engine.graph
    for t in ("fever", "rashes", "flu"):
        g.add_entity(t, "Topic")
    g.add_relation("dengue", "fever", "MENTIONS", "dengue:summary:0", "dengue", "high fever")
    g.add_relation("dengue", "rashes", "MENTIONS", "dengue:summary:0", "dengue", "rash")
    g.add_relation("flu", "fever", "MENTIONS", "dengue:summary:0", "flu", "fever")  # fever only
    a = engine.graphrag("I have a high fever and a rash. What could it be?")
    assert set(a.entities) == {"fever", "rashes"}
    assert "dengue" in a.path_nodes and "flu" not in a.path_nodes
    facts = engine.router.prompts[-1][1].split("Evidence passages:")[0]
    assert "dengue --MENTIONS--> fever" in facts and "dengue --MENTIONS--> rashes" in facts


def test_symptom_ties_are_broken_by_semantic_similarity(engine):
    g = engine.graph
    for t in ("fever", "rashes", "antibiotics"):
        g.add_entity(t, "Topic")
    for src in ("dengue", "antibiotics"):  # both link to fever and rash → tie on count
        g.add_relation(src, "fever", "MENTIONS", "dengue:summary:0", src, "e")
        g.add_relation(src, "rashes", "MENTIONS", "dengue:summary:0", src, "e")
    engine.chunks["antibiotics:summary:0"] = {"id": "antibiotics:summary:0", "drug": "antibiotics",
                                              "section": "summary", "text": "Antibiotics treat bacteria."}
    engine.index.upsert(["antibiotics:summary:0"], engine.embedder.embed_docs(["Antibiotics treat bacteria."]))
    a = engine.graphrag("You can get it if an infected mosquito bites you: fever and rash")
    ranked = [n for n in a.path_nodes if n in ("dengue", "antibiotics")]
    assert engine.last_candidates[:2] == ["dengue", "antibiotics"], engine.last_candidates
