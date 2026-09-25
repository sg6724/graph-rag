import json

import pytest

from medgraph.extract import build_prompt, extract_drug, parse_extraction
from tests.conftest import FakeRouter

VALID = {"simvastatin:drug_interactions:0"}


def test_build_prompt_lists_chunks_with_ids():
    p = build_prompt("simvastatin", [{"id": "simvastatin:drug_interactions:0", "section": "drug_interactions",
                                      "text": "Avoid with clarithromycin."}])
    assert "[simvastatin:drug_interactions:0]" in p and "Avoid with clarithromycin." in p
    assert "INHIBITS" in p and "Return JSON only" in p


def test_parse_normalizes_and_infers_endpoint_types():
    raw = json.dumps({"entities": [], "relations": [
        {"source": "Clarithromycin", "target": "CYP 3A4", "type": "INHIBITS",
         "chunk_id": "simvastatin:drug_interactions:0", "evidence": "strong CYP3A4 inhibitors"}]})
    ents, rels = parse_extraction("```json\n" + raw + "\n```", VALID)
    assert rels == [{"source": "clarithromycin", "target": "cyp3a4", "type": "INHIBITS",
                     "chunk_id": "simvastatin:drug_interactions:0", "evidence": "strong CYP3A4 inhibitors"}]
    assert {"name": "cyp3a4", "type": "Enzyme"} in ents
    assert {"name": "clarithromycin", "type": "Drug"} in ents


def test_parse_drops_invalid_items():
    raw = json.dumps({"entities": [{"name": "x", "type": "Planet"}, {"name": "myopathy", "type": "SideEffect"}],
                      "relations": [
                          {"source": "a", "target": "b", "type": "LIKES", "chunk_id": "simvastatin:drug_interactions:0"},
                          {"source": "a", "target": "b", "type": "CAUSES", "chunk_id": "made:up:9"},
                          {"source": "a", "target": "a", "type": "CAUSES", "chunk_id": "simvastatin:drug_interactions:0"},
                          {"source": "", "target": "b", "type": "CAUSES", "chunk_id": "simvastatin:drug_interactions:0"}]})
    ents, rels = parse_extraction(raw, VALID)
    assert rels == []
    assert ents == [{"name": "myopathy", "type": "SideEffect"}]


def test_parse_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_extraction("sorry, I cannot help", VALID)


def test_extract_drug_repairs_bad_json():
    good = json.dumps({"entities": [], "relations": [
        {"source": "simvastatin", "target": "myopathy", "type": "CAUSES",
         "chunk_id": "simvastatin:drug_interactions:0", "evidence": "myopathy"}]})
    router = FakeRouter(replies=["{not json", good])
    chunks = [{"id": "simvastatin:drug_interactions:0", "section": "drug_interactions", "text": "t"}]
    ents, rels = extract_drug(router, "simvastatin", chunks)
    assert len(rels) == 1 and len(router.prompts) == 2
    assert "not valid JSON" in router.prompts[1][1]


def test_extract_drug_gives_up_after_one_repair():
    router = FakeRouter(replies=["nope", "still nope"])
    chunks = [{"id": "simvastatin:drug_interactions:0", "section": "drug_interactions", "text": "t"}]
    assert extract_drug(router, "simvastatin", chunks) == ([], [])


def test_build_graph_parallel_keeps_order_and_survives_failures():
    from medgraph.extract import build_graph
    from medgraph.llm_router import LLMError

    class Router:
        def complete(self, prompt, task="extract", json_mode=False):
            drug = prompt.split("FDA label for ")[1].split(".")[0]
            if drug == "bad":
                raise LLMError("down")
            cid = f"{drug}:drug_interactions:0"
            from medgraph.llm_router import LLMResult
            return LLMResult(json.dumps({"entities": [], "relations": [
                {"source": drug, "target": "cyp3a4", "type": "METABOLIZED_BY", "chunk_id": cid, "evidence": "e"}]}),
                "fake", "m", False, 1.0, 1.0)

    by_drug = {d: [{"id": f"{d}:drug_interactions:0", "section": "drug_interactions", "text": "t"}]
               for d in ["alpha", "bad", "gamma"]}
    graph, failed = build_graph(Router(), by_drug, ["alpha", "bad", "gamma"], workers=3)
    assert failed == ["bad"]
    assert graph.paths_between("alpha", "gamma") == [["alpha", "cyp3a4", "gamma"]]
