"""The three answering pipelines and the simulated label-update flow."""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import combinations

import numpy as np

from medgraph import config
from medgraph.embeddings import doc_text
from medgraph.profiles import Profile, fda_profile

CHUNK_ID_RE = re.compile(r"[a-z0-9\-]+:[a-z_]+:\d+")


@dataclass
class Answer:
    mode: str
    question: str
    text: str
    citations: list[str] = field(default_factory=list)
    path_nodes: list[str] = field(default_factory=list)
    path_edges: list[dict] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    provider: str = ""
    llm_calls: int = 0
    cache_hit: bool = False
    cache_score: float | None = None
    blocked_by_key: str | None = None
    served_from: str | None = None  # on a cache hit: the cached question whose answer was reused

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Answer":
        return cls(**d)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


class Engine:
    def __init__(self, router, graph, chunks: dict[str, dict], index, embedder, cache, profile: Profile | None = None):
        self.profile = profile or fda_profile()
        self.last_candidates: list[str] = []  # ranked symptom-style candidates from the last graphrag call
        self.router = router
        self.graph = graph
        self.chunks = chunks
        self.index = index
        self.embedder = embedder
        self.cache = cache

    # ---- helpers --------------------------------------------------------
    def _block(self, cid: str) -> str:
        c = self.chunks[cid]
        return f"[{cid}] ({c['drug']} label, {c['section'].replace('_', ' ')}) {c['text']}"

    def _citations(self, text: str) -> list[str]:
        seen: list[str] = []
        for cid in CHUNK_ID_RE.findall(text):
            if cid in self.chunks and cid not in seen:
                seen.append(cid)
        return seen

    def _generate(self, question: str, context: str, graph_note: str) -> tuple[str, str, float]:
        res = self.router.complete(self.profile.prompt.format(graph_note=graph_note, question=question,
                                                        context=context), task="answer")
        provider = f"{res.provider}/{res.model}" + (" (disk cache)" if res.cached else "")
        return res.text, provider, res.gen_latency_ms

    # ---- pipelines ------------------------------------------------------
    def vanilla(self, question: str) -> Answer:
        t0 = time.perf_counter()
        qvec = self.embedder.embed_query(question)
        hits = self.index.search(qvec, config.TOP_K)
        context = "Context passages:\n" + "\n\n".join(self._block(cid) for cid, _ in hits)
        retrieve_ms = _ms(t0)
        text, provider, gen_ms = self._generate(question, context, "")
        return Answer(mode="vanilla", question=question, text=text, citations=self._citations(text),
                      timings={"retrieve_ms": retrieve_ms, "generate_ms": gen_ms,
                               "total_ms": retrieve_ms + gen_ms},
                      provider=provider, llm_calls=1)

    def graphrag(self, question: str, qvec: np.ndarray | None = None) -> Answer:
        t0 = time.perf_counter()
        types = self.graph.node_types()
        ents = self.profile.match(question, types)
        if qvec is None:
            qvec = self.embedder.embed_query(self.profile.canonical(question))
        seeds = list(ents)
        if not seeds:
            seeds = sorted({self.chunks[cid]["drug"] for cid, _ in self.index.search(qvec, 3)})
        # A path counts as evidence only if every hop in between is a mechanism (e.g. a shared enzyme);
        # "A interacts with X, X interacts with B" says nothing about A + B and floods the context.
        drugs = [s for s in seeds if types.get(s) in self.profile.seed_types]

        nodes: set[str] = set(seeds)
        edges: list[dict] = []
        if self.profile.common_neighbors and len(drugs) >= 2:
            # e.g. fever + rash + eye pain: topics that link to *most* of the symptoms are the candidates
            shared = dict(self.graph.common_neighbors(drugs, self.profile.seed_types, limit=25))
            own = {cid for cid, c in self.chunks.items() if c["drug"] in shared}
            sim: dict[str, float] = {}
            for cid, score in self.index.search(qvec, 200, allowed=own):  # ties → most similar to the question
                ent = self.chunks[cid]["drug"]
                sim[ent] = max(sim.get(ent, -1.0), score)
            self.last_candidates = sorted(shared, key=lambda n: (-shared[n], -sim.get(n, -1.0)))[:8]
            for cand in self.last_candidates:
                nodes.add(cand)
                for s in drugs:
                    edges.extend(self.graph.edges_between(cand, s))
        for a, b in ([] if edges else combinations(drugs, 2)):
            for path in self.graph.paths_between(a, b):
                if any(types.get(n) not in self.profile.path_types for n in path[1:-1]):
                    continue
                nodes.update(path)
                for u, v in zip(path, path[1:]):
                    edges.extend(self.graph.edges_between(u, v))
        if not edges:
            edges = self.graph.neighborhood_edges(seeds, config.NEIGHBOR_EDGE_LIMIT, self.profile.edge_priority)
            for e in edges:
                nodes.update((e["source"], e["target"]))
        unique: dict[tuple, dict] = {}
        for e in edges:
            unique.setdefault((e["source"], e["target"], e["type"], e["chunk_id"]), e)
        edges = list(unique.values())[: config.MAX_FACTS]

        counts = Counter(e["chunk_id"] for e in edges if e["chunk_id"] in self.chunks)
        chunk_ids = [cid for cid, _ in counts.most_common(config.MAX_CONTEXT_CHUNKS - 2)]
        allowed = {cid for cid, c in self.chunks.items() if c["drug"] in drugs} or None
        for cid, _ in self.index.search(qvec, 2, allowed=allowed):
            if cid not in chunk_ids:
                chunk_ids.append(cid)

        facts = "\n".join(f"- {e['source']} --{e['type']}--> {e['target']} [{e['chunk_id']}]" for e in edges)
        context = ("Knowledge-graph facts (source --RELATION--> target [evidence id]):\n" + (facts or "(none)")
                   + "\n\nEvidence passages:\n" + "\n\n".join(self._block(cid) for cid in chunk_ids))
        retrieve_ms = _ms(t0)
        text, provider, gen_ms = self._generate(question, context, " plus facts from a knowledge graph built from them")
        return Answer(mode="graphrag", question=question, text=text, citations=self._citations(text),
                      path_nodes=sorted(nodes), path_edges=edges, entities=ents,
                      timings={"retrieve_ms": retrieve_ms, "generate_ms": gen_ms,
                               "total_ms": retrieve_ms + gen_ms},
                      provider=provider, llm_calls=1)

    def cached(self, question: str) -> Answer:
        t0 = time.perf_counter()
        types = self.graph.node_types()
        key = self.profile.key(self.profile.match(question, types), types)
        qvec = self.embedder.embed_query(self.profile.canonical(question))
        entry, score = self.cache.lookup(qvec, key)
        lookup_ms = _ms(t0)
        if entry is not None:
            a = Answer.from_dict(entry.answer)
            a.mode, a.question, a.cache_hit, a.cache_score = "cached", question, True, score
            a.llm_calls, a.provider, a.blocked_by_key = 0, "semantic cache", None
            a.served_from = entry.query
            a.timings = {"lookup_ms": lookup_ms, "total_ms": lookup_ms}
            return a
        near, near_score = self.cache.nearest(qvec)
        blocked = near.query if near is not None and near_score >= self.cache.threshold and near.entity_key != key else None
        a = self.graphrag(question, qvec=qvec)  # raises LLMError before anything is cached
        a.mode = "cached"
        a.cache_score = score if score >= 0 else None
        a.blocked_by_key = blocked
        a.timings["lookup_ms"] = lookup_ms
        a.timings["total_ms"] += lookup_ms
        self.cache.store(question, qvec, key, a.to_dict(), a.path_nodes)
        return a

    # ---- demo: simulated source update (FDA label / MedlinePlus topic) ---
    def update_label(self, drug: str, section: str, text: str) -> dict:
        n = sum(1 for c in self.chunks.values() if c["drug"] == drug and c["section"] == section)
        prefix = next((cid.split(":")[0] for cid, c in self.chunks.items() if c["drug"] == drug),
                      drug.replace(" ", ""))  # ids use the source slug, e.g. "lymedisease"
        cid = f"{prefix}:{section}:{n}"
        chunk = {"id": cid, "drug": drug, "section": section, "text": f"SIMULATED LABEL UPDATE (demo): {text}"}
        self.chunks[cid] = chunk
        self.index.upsert([cid], self.embedder.embed_docs([doc_text(chunk)]))
        ents, rels = self.profile.link_text(self.router, drug, chunk, self.graph.node_types())
        changed = self.graph.replace_chunks(drug, {cid}, ents, rels, self.profile.entity_type)
        evicted = self.cache.invalidate_nodes(changed)
        return {"drug": drug, "chunk_id": cid, "changed_nodes": sorted(changed),
                "evicted": [e.query for e in evicted], "retained": [e.query for e in self.cache.entries]}


def load_engine(cache_path=config.CACHE_PATH, dataset: str = "fda", backend: str | None = None) -> Engine:
    """backend "pg" = Supabase Postgres + pgvector (default when DATABASE_URL is set), "files" = local artifacts."""
    import json

    from medgraph.embeddings import Embedder, VectorIndex
    from medgraph.graph_store import GraphStore
    from medgraph.ingest import load_chunks
    from medgraph.llm_router import Router, get_key
    from medgraph.profiles import medline_profile
    from medgraph.semantic_cache import SemanticCache

    profile = fda_profile()
    if dataset == "medline":
        profile = medline_profile(json.loads((config.MEDLINE_DIR / "aliases.json").read_text(encoding="utf-8")))
    backend = backend or ("pg" if get_key("DATABASE_URL") else "files")
    if backend == "pg":
        from medgraph.pgstore import PgSemanticCache, PgVectorIndex, connect, load_graph, read_chunks

        conn = connect()
        chunks = read_chunks(conn, dataset)
        return Engine(Router(), load_graph(conn, dataset), chunks, PgVectorIndex(conn, dataset, chunks), Embedder(),
                      PgSemanticCache(conn, dataset), profile)
    d = config.MEDLINE_DIR if dataset == "medline" else config.DATA
    if dataset == "medline":
        cache_path = d / "cache.json" if cache_path == config.CACHE_PATH else cache_path
    cache = SemanticCache()  # in-memory: demo edits never overwrite the pre-warmed file
    if cache_path and cache_path.exists():
        cache.load(cache_path)
    return Engine(Router(), GraphStore.load(d / "graph.json"), load_chunks(d / "chunks.jsonl"),
                  VectorIndex.load(d / "embeddings.npy", d / "embedding_ids.json"), Embedder(), cache, profile)
