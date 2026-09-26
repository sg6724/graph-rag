"""Postgres + pgvector storage (Supabase): chunk embeddings, semantic cache and the knowledge graph.

Every table has a `dataset` column so the FDA-label and MedlinePlus corpora live side by side.
The classes mirror VectorIndex / SemanticCache so the Engine works with either backend.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from medgraph import config
from medgraph.embeddings import _normalize
from medgraph.graph_store import GraphStore
from medgraph.llm_router import get_key
from medgraph.semantic_cache import CacheEntry

DIM = 384

SCHEMA_SQL = [
    """create table if not exists chunks (
        dataset text not null, id text not null, entity text not null, section text not null,
        text text not null, embedding vector(384) not null, primary key (dataset, id))""",
    "create index if not exists chunks_embedding_hnsw on chunks using hnsw (embedding vector_cosine_ops)",
    """create table if not exists graph_nodes (
        dataset text not null, name text not null, type text not null, primary key (dataset, name))""",
    """create table if not exists graph_edges (
        dataset text not null, source text not null, target text not null, type text not null,
        chunk_id text not null, drug text not null, evidence text not null default '',
        primary key (dataset, source, target, type, chunk_id))""",
    """create table if not exists semantic_cache (
        id bigserial primary key, dataset text not null, query text not null, embedding vector(384) not null,
        entity_key text not null, answer jsonb not null, node_ids text[] not null,
        created_at timestamptz not null default now())""",
    "create index if not exists semantic_cache_embedding_hnsw on semantic_cache using hnsw (embedding vector_cosine_ops)",
    "create index if not exists semantic_cache_node_ids on semantic_cache using gin (node_ids)",
    "create index if not exists semantic_cache_key on semantic_cache (dataset, entity_key)",
]


class PgConn:
    """A Postgres connection that reopens itself. Supabase's pooler drops idle connections, and a Streamlit
    app keeps one engine (and connection) alive for hours — without this the next query fails."""

    def __init__(self, url: str, schema: str | None = None):
        self.url, self.schema = url, schema
        self.raw: psycopg.Connection = self._open()

    def _open(self) -> psycopg.Connection:
        conn = psycopg.connect(self.url, autocommit=True, prepare_threshold=None, connect_timeout=15,
                               keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=3)
        conn.execute("create extension if not exists vector")
        if self.schema:
            conn.execute(f'create schema if not exists "{self.schema}"')
            conn.execute(f'set search_path to "{self.schema}", public, extensions')
        register_vector(conn)
        return conn

    def _live(self) -> psycopg.Connection:
        if self.raw.closed:
            self.raw = self._open()
        return self.raw

    def execute(self, query, params=None):
        try:
            return self._live().execute(query, params)
        except psycopg.OperationalError:  # dropped mid-idle: reopen once and retry
            self.raw = self._open()
            return self.raw.execute(query, params)

    def cursor(self):
        return self._live().cursor()

    def close(self) -> None:
        self.raw.close()


def connect(url: str | None = None, schema: str | None = None) -> PgConn:
    url = url or get_key("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    return PgConn(url, schema)


def init_schema(conn: psycopg.Connection) -> None:
    for sql in SCHEMA_SQL:
        conn.execute(sql)


def _vec(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float32)


# ---------------- chunks + vector search ----------------
def write_chunks(conn, dataset: str, chunks: list[dict], matrix: np.ndarray) -> None:
    matrix = _normalize(np.asarray(matrix, dtype=np.float32))
    with conn.cursor() as cur:
        cur.executemany(
            """insert into chunks (dataset, id, entity, section, text, embedding) values (%s, %s, %s, %s, %s, %s)
               on conflict (dataset, id) do update set entity = excluded.entity, section = excluded.section,
               text = excluded.text, embedding = excluded.embedding""",
            [(dataset, c["id"], c["drug"], c["section"], c["text"], row) for c, row in zip(chunks, matrix)])


def read_chunks(conn, dataset: str) -> dict[str, dict]:
    rows = conn.execute("select id, entity, section, text from chunks where dataset = %s order by id",
                        (dataset,)).fetchall()
    return {r[0]: {"id": r[0], "drug": r[1], "section": r[2], "text": r[3]} for r in rows}


class PgVectorIndex:
    """pgvector-backed drop-in for VectorIndex (cosine distance via the <=> operator + HNSW index)."""

    def __init__(self, conn, dataset: str, chunks: dict[str, dict]):
        self.conn, self.dataset, self.chunks = conn, dataset, chunks

    @property
    def ids(self) -> list[str]:
        return [r[0] for r in self.conn.execute("select id from chunks where dataset = %s order by id",
                                                (self.dataset,)).fetchall()]

    def search(self, qvec, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        q = _vec(qvec)
        if allowed is not None:
            rows = self.conn.execute(
                """select id, 1 - (embedding <=> %s) from chunks where dataset = %s and id = any(%s)
                   order by embedding <=> %s limit %s""", (q, self.dataset, list(allowed), q, k)).fetchall()
        else:
            rows = self.conn.execute(
                "select id, 1 - (embedding <=> %s) from chunks where dataset = %s order by embedding <=> %s limit %s",
                (q, self.dataset, q, k)).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def upsert(self, ids: list[str], matrix: np.ndarray) -> None:
        write_chunks(self.conn, self.dataset, [self.chunks[i] for i in ids], matrix)


# ---------------- graph ----------------
def save_graph(conn, dataset: str, graph: GraphStore) -> None:
    conn.execute("delete from graph_edges where dataset = %s", (dataset,))
    conn.execute("delete from graph_nodes where dataset = %s", (dataset,))
    with conn.cursor() as cur:
        cur.executemany("insert into graph_nodes (dataset, name, type) values (%s, %s, %s)",
                        [(dataset, n, d.get("type", "Unknown")) for n, d in graph.g.nodes(data=True)])
        cur.executemany(
            """insert into graph_edges (dataset, source, target, type, chunk_id, drug, evidence)
               values (%s, %s, %s, %s, %s, %s, %s) on conflict do nothing""",
            [(dataset, u, v, d["type"], d["chunk_id"], d["drug"], d.get("evidence", ""))
             for u, v, d in graph.g.edges(data=True)])


def load_graph(conn, dataset: str) -> GraphStore:
    gs = GraphStore()
    for name, type_ in conn.execute("select name, type from graph_nodes where dataset = %s", (dataset,)):
        gs.g.add_node(name, type=type_)
    for s, t, ty, cid, drug, ev in conn.execute(
            "select source, target, type, chunk_id, drug, evidence from graph_edges where dataset = %s", (dataset,)):
        gs.add_relation(s, t, ty, cid, drug, ev)
    return gs


# ---------------- semantic cache ----------------
class PgSemanticCache:
    """pgvector-backed drop-in for SemanticCache. Invalidation is one DELETE using array overlap (&&)."""

    def __init__(self, conn, dataset: str, threshold: float = config.CACHE_THRESHOLD):
        self.conn, self.dataset, self.threshold = conn, dataset, threshold

    @staticmethod
    def _entry(row) -> CacheEntry:
        query, emb, key, answer, nodes, created = row
        values = emb.to_list() if hasattr(emb, "to_list") else list(emb)  # pgvector returns a Vector
        return CacheEntry(query, [float(x) for x in values], key, answer, list(nodes), created.timestamp())

    _COLS = "query, embedding, entity_key, answer, node_ids, created_at"

    @property
    def entries(self) -> list[CacheEntry]:
        rows = self.conn.execute(f"select {self._COLS} from semantic_cache where dataset = %s order by id",
                                 (self.dataset,)).fetchall()
        return [self._entry(r) for r in rows]

    def lookup(self, qvec, entity_key: str) -> tuple[CacheEntry | None, float]:
        q = _vec(qvec)
        row = self.conn.execute(
            f"""select {self._COLS}, 1 - (embedding <=> %s) from semantic_cache
                where dataset = %s and entity_key = %s order by embedding <=> %s limit 1""",
            (q, self.dataset, entity_key, q)).fetchone()
        if row is None:
            return None, -1.0
        score = float(row[-1])
        return (self._entry(row[:-1]), score) if score >= self.threshold else (None, score)

    def nearest(self, qvec) -> tuple[CacheEntry | None, float]:
        q = _vec(qvec)
        row = self.conn.execute(
            f"""select {self._COLS}, 1 - (embedding <=> %s) from semantic_cache where dataset = %s
                order by embedding <=> %s limit 1""", (q, self.dataset, q)).fetchone()
        return (None, -1.0) if row is None else (self._entry(row[:-1]), float(row[-1]))

    def store(self, query: str, qvec, entity_key: str, answer: dict, node_ids: list[str]) -> CacheEntry:
        self.conn.execute(
            """insert into semantic_cache (dataset, query, embedding, entity_key, answer, node_ids)
               values (%s, %s, %s, %s, %s, %s)""",
            (self.dataset, query, _vec(qvec), entity_key, Jsonb(answer), list(node_ids)))
        return CacheEntry(query, [float(x) for x in qvec], entity_key, answer, list(node_ids), time.time())

    def invalidate_nodes(self, nodes: set[str]) -> list[CacheEntry]:
        rows = self.conn.execute(
            f"delete from semantic_cache where dataset = %s and node_ids && %s returning {self._COLS}",
            (self.dataset, list(nodes))).fetchall()
        return [self._entry(r) for r in rows]

    def clear(self) -> None:
        self.conn.execute("delete from semantic_cache where dataset = %s", (self.dataset,))

    def load(self, path: Path) -> None:
        """Replace this dataset's cache with the entries in a JSON export (used to pre-warm the demo)."""
        self.clear()
        for d in json.loads(Path(path).read_text(encoding="utf-8")):
            self.store(d["query"], d["embedding"], d["entity_key"], d["answer"], d["node_ids"])

    def save(self, path: Path | None = None) -> None:
        from dataclasses import asdict

        Path(path).write_text(json.dumps([asdict(e) for e in self.entries]), encoding="utf-8")


# ---------------- loader CLI ----------------
def dataset_files(dataset: str) -> dict:
    d = config.MEDLINE_DIR if dataset == "medline" else config.DATA
    return {"chunks": d / "chunks.jsonl", "graph": d / "graph.json", "emb": d / "embeddings.npy",
            "ids": d / "embedding_ids.json", "cache": d / "cache.json"}


def load_dataset(conn, dataset: str) -> dict:
    """Copy a dataset's local artifacts (chunks + embeddings, graph, warm cache) into Postgres."""
    from medgraph.embeddings import VectorIndex
    from medgraph.ingest import load_chunks

    f = dataset_files(dataset)
    chunks = load_chunks(f["chunks"])
    index = VectorIndex.load(f["emb"], f["ids"])
    conn.execute("delete from chunks where dataset = %s", (dataset,))
    write_chunks(conn, dataset, [chunks[i] for i in index.ids], index.matrix)
    save_graph(conn, dataset, GraphStore.load(f["graph"]))
    cache = PgSemanticCache(conn, dataset)
    if f["cache"].exists():
        cache.load(f["cache"])
    else:
        cache.clear()
    counts = {t: conn.execute(f"select count(*) from {t} where dataset = %s", (dataset,)).fetchone()[0]
              for t in ("chunks", "graph_nodes", "graph_edges", "semantic_cache")}
    return counts


def main() -> None:
    import sys

    datasets = sys.argv[2:] if len(sys.argv) > 2 and sys.argv[1] == "load" else ["fda", "medline"]
    conn = connect()
    init_schema(conn)
    for ds in datasets:
        print(ds, load_dataset(conn, ds), flush=True)


if __name__ == "__main__":
    main()
