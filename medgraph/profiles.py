"""Per-dataset behaviour: how questions map to graph entities, which graph paths count as evidence,
how the cache key is formed, how new text becomes graph edges, and how the answer is worded."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from medgraph.entities import canonicalize, drug_key, match_entities, match_topics

FDA_PROMPT = """You are a clinical pharmacology assistant in an educational demo. Answer the question using ONLY the
context below, which comes from FDA drug labels{graph_note}.
- Start with a one-line verdict (e.g. "Avoid combination", "Use with caution / monitor", "Contraindicated",
  or "No significant interaction documented in these labels").
  Say "Contraindicated" or "Avoid" only if the label text for these two drugs says so; if it says to monitor,
  adjust the dose or use caution, say "Use with caution / monitor". Facts about other drugs never set the verdict.
- Then explain the mechanism or risk in 2-4 sentences (e.g. which enzyme is inhibited and whose levels rise).
- Cite sources inline with their ids in square brackets, e.g. [simvastatin:drug_interactions:2].
- If the context does not contain the answer, say so plainly. Do not use outside knowledge.

Question: {question}

{context}
"""

MEDLINE_PROMPT = """You are a health information assistant in an educational demo. Answer the question using ONLY the
context below, which comes from MedlinePlus health topics (US National Library of Medicine){graph_note}.
- Answer in plain, simple language in 2-5 sentences, or a short bullet list.
- If the question asks "which" conditions or topics, list every matching one found in the context.
- Cite sources inline with their ids in square brackets, e.g. [dengue:summary:2].
- If the context does not contain the answer, say so plainly. Do not use outside knowledge.

Question: {question}

{context}
"""


@dataclass
class Profile:
    name: str
    label: str
    prompt: str
    match: Callable[[str, dict[str, str]], list[str]]
    key: Callable[[list[str], dict[str, str]], str]
    canonical: Callable[[str], str]
    entity_type: str               # node type of a source document's own entity (a drug label's drug, a topic)
    seed_types: set[str]           # entity types whose pairs we connect with graph paths
    path_types: set[str]           # node types allowed *between* two seeds on an evidence path
    edge_priority: list[str]       # which neighbour edges to show first when there is no path
    link_text: Callable = None     # (router, entity, chunk, node_types) -> (entities, relations) for new text
    examples: list[str] = field(default_factory=list)


def _fda_link_text(router, entity, chunk, node_types):
    from medgraph.extract import extract_drug

    return extract_drug(router, entity, [chunk], task="extract_fast")  # live demo: fast model chain


def fda_profile() -> Profile:
    return Profile(
        name="fda", label="FDA drug labels (drug interactions)", prompt=FDA_PROMPT,
        match=match_entities, key=drug_key, canonical=canonicalize,
        entity_type="Drug", seed_types={"Drug"}, path_types={"Enzyme", "DrugClass", "SideEffect"},
        edge_priority=["INTERACTS_WITH", "CONTRAINDICATED_IN", "INHIBITS", "INDUCES", "METABOLIZED_BY",
                       "BELONGS_TO", "CAUSES"],
        link_text=_fda_link_text,
        examples=["Can a patient taking simvastatin start clarithromycin?", "Is it safe to take Biaxin while on Zocor?",
                  "Can a patient taking warfarin also take aspirin?", "Can a patient taking warfarin also take naproxen?"])


def medline_profile(aliases: dict[str, str]) -> Profile:
    from medgraph.aliases import normalize_text

    def match(question, node_types):
        return match_topics(question, node_types, aliases)

    def key(entities, node_types):
        return "|".join(sorted(e for e in entities if node_types.get(e) == "Topic"))

    def link_text(router, entity, chunk, node_types):
        """New topic text → MENTIONS edges to every topic it names (deterministic, no LLM)."""
        targets = [t for t in match_topics(chunk["text"], node_types, aliases) if t != entity]
        rels = [{"source": entity, "target": t, "type": "MENTIONS", "chunk_id": chunk["id"],
                 "evidence": chunk["text"][:300]} for t in targets]
        return [{"name": t, "type": "Topic"} for t in targets], rels

    return Profile(
        name="medline", label="MedlinePlus health topics (plain English)", prompt=MEDLINE_PROMPT,
        match=match, key=key, canonical=normalize_text,
        entity_type="Topic", seed_types={"Topic"}, path_types={"Topic"},
        edge_priority=["MENTIONS", "RELATED_TO", "IN_GROUP"], link_text=link_text,
        examples=["Which diseases spread through mosquito bites?",
                  "Which diseases are spread by mosquitoes?",
                  "Which diseases spread through ticks?",
                  "I have a high fever, a rash and pain behind my eyes. What could it be?",
                  "Is dengue related to Zika?"])
