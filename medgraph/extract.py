"""LLM extraction of entities/relations from drug-label chunks, and the graph build CLI."""
from __future__ import annotations

from collections import defaultdict

from medgraph import config
from medgraph.aliases import normalize_name
from medgraph.graph_store import GraphStore
from medgraph.llm_router import Router, parse_json_object

NODE_TYPES = {"Drug", "DrugClass", "Enzyme", "Condition", "SideEffect"}
EDGE_TYPES = {"INHIBITS", "INDUCES", "METABOLIZED_BY", "INTERACTS_WITH",
              "CONTRAINDICATED_IN", "CAUSES", "BELONGS_TO"}
TARGET_TYPE = {"INHIBITS": "Enzyme", "INDUCES": "Enzyme", "METABOLIZED_BY": "Enzyme",
               "INTERACTS_WITH": "Drug", "CONTRAINDICATED_IN": "Condition",
               "CAUSES": "SideEffect", "BELONGS_TO": "DrugClass"}

PROMPT = """You are building a drug-interaction knowledge graph from the FDA label for {drug}.

Allowed entity types: Drug, DrugClass, Enzyme, Condition, SideEffect.
Allowed relation types:
- INHIBITS (Drug -> Enzyme or transporter such as p-gp)
- INDUCES (Drug -> Enzyme)
- METABOLIZED_BY (Drug -> Enzyme or transporter)
- INTERACTS_WITH (Drug -> Drug or DrugClass) for a clinically meaningful interaction
- CONTRAINDICATED_IN (Drug -> Condition, or Drug -> Drug/DrugClass when co-use is contraindicated)
- CAUSES (Drug -> SideEffect) for serious or frequently warned adverse effects
- BELONGS_TO (Drug -> DrugClass)

Rules:
- Use lowercase generic names ("simvastatin", not "Zocor"). Enzymes like "cyp3a4", "cyp2c9", "p-gp".
- Only extract facts stated in the text. Every relation must cite the chunk_id it came from and include
  a short evidence quote (max 25 words) copied from that chunk.
- Focus on interactions, metabolism, contraindications and serious warnings. Skip dosing tables and trivia.

Return JSON only, in exactly this shape:
{{"entities": [{{"name": "...", "type": "..."}}],
  "relations": [{{"source": "...", "target": "...", "type": "...", "chunk_id": "...", "evidence": "..."}}]}}

Label text (each passage starts with its chunk_id in brackets):
{passages}
"""

REPAIR = """The text below was supposed to be valid JSON with keys "entities" and "relations" but it is not valid JSON.
Return only the corrected JSON object, nothing else.

{text}
"""


def build_prompt(drug: str, chunks: list[dict]) -> str:
    passages = "\n\n".join(f"[{c['id']}] ({c['section']}) {c['text']}" for c in chunks)
    return PROMPT.format(drug=drug, passages=passages)


def parse_extraction(text: str, valid_chunk_ids: set[str]) -> tuple[list[dict], list[dict]]:
    data = parse_json_object(text)
    types: dict[str, str] = {}
    for e in data.get("entities") or []:
        if not isinstance(e, dict):
            continue
        name, t = normalize_name(str(e.get("name", ""))), e.get("type")
        if name and t in NODE_TYPES:
            types.setdefault(name, t)
    relations = []
    for r in data.get("relations") or []:
        if not isinstance(r, dict):
            continue
        src, tgt = normalize_name(str(r.get("source", ""))), normalize_name(str(r.get("target", "")))
        rtype, cid = r.get("type"), r.get("chunk_id")
        if not src or not tgt or src == tgt or rtype not in EDGE_TYPES or cid not in valid_chunk_ids:
            continue
        relations.append({"source": src, "target": tgt, "type": rtype, "chunk_id": cid,
                          "evidence": str(r.get("evidence", ""))[:300]})
        types.setdefault(src, "Drug")
        types.setdefault(tgt, TARGET_TYPE[rtype])
    entities = [{"name": n, "type": t} for n, t in types.items()]
    return entities, relations


def extract_drug(router, drug: str, chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    valid = {c["id"] for c in chunks}
    res = router.complete(build_prompt(drug, chunks), task="extract", json_mode=True)
    try:
        return parse_extraction(res.text, valid)
    except ValueError:
        fixed = router.complete(REPAIR.format(text=res.text), task="extract", json_mode=True)
        try:
            return parse_extraction(fixed.text, valid)
        except ValueError:
            print(f"WARN {drug}: extraction JSON unusable, skipped")
            return [], []


def main() -> None:
    from medgraph.ingest import load_chunks

    router = Router()
    by_drug: dict[str, list[dict]] = defaultdict(list)
    for c in load_chunks().values():
        by_drug[c["drug"]].append(c)
    graph = GraphStore()
    for drug in config.DRUGS:
        if drug not in by_drug:
            continue
        try:
            ents, rels = extract_drug(router, drug, by_drug[drug])
        except Exception as e:  # keep building; a missing drug is better than no graph
            print(f"FAIL {drug}: {e}")
            continue
        graph.add_extraction(drug, ents, rels)
        graph.save()  # save as we go so a crash mid-way keeps progress
        print(f"{drug:15s} entities={len(ents):3d} relations={len(rels):3d}  via={router.by_provider}", flush=True)
    print(graph.stats())


if __name__ == "__main__":
    main()
