import json

import pytest

from medgraph.embeddings import VectorIndex
from medgraph.graph_store import GraphStore
from medgraph.llm_router import LLMError
from medgraph.pipelines import Answer, Engine
from medgraph.semantic_cache import SemanticCache
from tests.conftest import FakeEmbedder, FakeRouter

TEXTS = {
    "simvastatin:clinical_pharmacology:0": "Simvastatin is metabolized by CYP3A4.",
    "simvastatin:warnings_and_cautions:0": "Simvastatin can cause myopathy and rhabdomyolysis.",
    "clarithromycin:drug_interactions:0": "Clarithromycin is a strong CYP3A4 inhibitor.",
    "warfarin:warnings:0": "Warfarin can cause major or fatal bleeding.",
    "aspirin:drug_interactions:0": "Aspirin increases bleeding risk with warfarin.",
    "ibuprofen:drug_interactions:0": "Ibuprofen increases bleeding risk with warfarin.",
}
DEFAULT = "Avoid combination. Clarithromycin inhibits CYP3A4 [clarithromycin:drug_interactions:0]."


def rel(s, t, typ, cid):
    return {"source": s, "target": t, "type": typ, "chunk_id": cid, "evidence": TEXTS[cid]}


@pytest.fixture
def engine():
    g = GraphStore()
    g.add_extraction("simvastatin", [{"name": "cyp3a4", "type": "Enzyme"}], [
        rel("simvastatin", "cyp3a4", "METABOLIZED_BY", "simvastatin:clinical_pharmacology:0"),
        rel("simvastatin", "myopathy", "CAUSES", "simvastatin:warnings_and_cautions:0")])
    g.add_extraction("clarithromycin", [], [rel("clarithromycin", "cyp3a4", "INHIBITS", "clarithromycin:drug_interactions:0")])
    g.add_extraction("warfarin", [], [rel("warfarin", "bleeding", "CAUSES", "warfarin:warnings:0")])
    g.add_extraction("aspirin", [], [rel("aspirin", "warfarin", "INTERACTS_WITH", "aspirin:drug_interactions:0")])
    g.add_extraction("ibuprofen", [], [rel("ibuprofen", "warfarin", "INTERACTS_WITH", "ibuprofen:drug_interactions:0")])
    chunks = {cid: {"id": cid, "drug": cid.split(":")[0], "section": cid.split(":")[1], "text": t}
              for cid, t in TEXTS.items()}
    emb = FakeEmbedder()
    ids = list(chunks)
    index = VectorIndex(ids, emb.embed_docs([chunks[i]["text"] for i in ids]))
    return Engine(FakeRouter(default=DEFAULT), g, chunks, index, emb, SemanticCache(threshold=0.9))


def test_graphrag_finds_enzyme_path(engine):
    a = engine.graphrag("Can simvastatin be taken with clarithromycin?")
    assert a.mode == "graphrag" and a.llm_calls == 1
    assert "cyp3a4" in a.path_nodes
    prompt = engine.router.prompts[-1][1]
    assert "clarithromycin --INHIBITS--> cyp3a4" in prompt
    assert "[simvastatin:clinical_pharmacology:0]" in prompt
    assert a.citations == ["clarithromycin:drug_interactions:0"]
    assert a.timings["total_ms"] >= a.timings["generate_ms"]


def test_vanilla_uses_vector_chunks_only(engine):
    a = engine.vanilla("Can simvastatin be taken with clarithromycin?")
    assert a.mode == "vanilla" and a.path_nodes == [] and a.llm_calls == 1
    assert "Knowledge-graph facts" not in engine.router.prompts[-1][1]


def test_graphrag_without_entities_falls_back(engine):
    a = engine.graphrag("What causes muscle damage and rhabdomyolysis?")
    assert a.path_nodes  # seeded from vector search
    assert len(engine.router.prompts) == 1


def test_cached_hit_on_repeat_and_brand_name(engine):
    first = engine.cached("Can simvastatin be taken with clarithromycin?")
    second = engine.cached("Can Zocor be taken with Biaxin?")
    assert not first.cache_hit
    assert second.cache_hit and second.llm_calls == 0 and second.provider == "semantic cache"
    assert second.text == first.text and second.cache_score >= 0.9
    assert len(engine.router.prompts) == 1


def test_cached_blocks_different_drug(engine):
    engine.cache.threshold = 0.5
    engine.cached("Can I take aspirin with warfarin?")
    b = engine.cached("Can I take ibuprofen with warfarin?")
    assert not b.cache_hit
    assert b.blocked_by_key == "Can I take aspirin with warfarin?"
    assert len(engine.router.prompts) == 2


def test_llm_failure_not_cached(engine):
    engine.router.replies = [LLMError("all providers failed")]
    with pytest.raises(LLMError):
        engine.cached("Can simvastatin be taken with clarithromycin?")
    assert engine.cache.entries == []


def test_update_label_evicts_only_affected(engine):
    q1 = "Can simvastatin be taken with clarithromycin?"
    q2 = "Can I take aspirin with warfarin?"
    engine.cached(q1)
    engine.cached(q2)
    engine.router.replies = [json.dumps({"entities": [{"name": "metronidazole", "type": "Drug"}], "relations": [
        {"source": "warfarin", "target": "metronidazole", "type": "INTERACTS_WITH",
         "chunk_id": "warfarin:drug_interactions:0", "evidence": "markedly increase INR"}]})]
    res = engine.update_label("warfarin", "drug_interactions", "Metronidazole may markedly increase INR.")
    assert res["chunk_id"] == "warfarin:drug_interactions:0"
    assert res["evicted"] == [q2] and res["retained"] == [q1]
    assert "metronidazole" in res["changed_nodes"] and "warfarin" in res["changed_nodes"]
    assert engine.graph.g.has_edge("warfarin", "metronidazole")
    assert "warfarin:drug_interactions:0" in engine.chunks
    assert engine.chunks["warfarin:drug_interactions:0"]["text"].startswith("SIMULATED LABEL UPDATE")
    assert "warfarin:drug_interactions:0" in engine.index.ids
    assert engine.router.prompts[-1][0] == "extract_fast"  # live demo must use the fast model chain


def test_answer_roundtrip():
    a = Answer(mode="graphrag", question="q", text="t", citations=["x:y:0"], timings={"total_ms": 1.0})
    assert Answer.from_dict(a.to_dict()) == a


def test_cache_hit_records_which_cached_question_served_it(engine):
    engine.cached("Can simvastatin be taken with clarithromycin?")
    hit = engine.cached("Can Zocor be taken with Biaxin?")
    assert hit.served_from == "Can simvastatin be taken with clarithromycin?"
    miss = engine.cached("Can I take aspirin with warfarin?")
    assert miss.served_from is None


def test_graphrag_ignores_paths_through_unrelated_drugs(engine):
    # warfarin–ibuprofen–... : "A interacts with X, X interacts with B" is not a mechanism linking A and B
    engine.graph.add_extraction("naproxen", [], [
        {"source": "naproxen", "target": "ibuprofen", "type": "INTERACTS_WITH",
         "chunk_id": "ibuprofen:drug_interactions:0", "evidence": "x"}])
    engine.graphrag("Can I take naproxen with warfarin?")
    facts = engine.router.prompts[-1][1].split("Evidence passages:")[0]
    # the drug-bridge path is dropped, so the context falls back to the question drugs' own facts
    assert "warfarin --CAUSES--> bleeding" in facts


def test_graphrag_keeps_mechanism_paths_through_enzymes(engine):
    a = engine.graphrag("Can simvastatin be taken with clarithromycin?")
    assert {"simvastatin", "cyp3a4", "clarithromycin"} <= set(a.path_nodes)



def test_answer_prompt_ties_verdict_to_label_wording(engine):
    engine.vanilla("Can simvastatin be taken with clarithromycin?")
    prompt = engine.router.prompts[-1][1]
    assert "only if the label text for these two drugs says so" in prompt
