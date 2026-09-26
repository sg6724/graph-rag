"""Integration tests against real Postgres + pgvector (your Supabase), inside a throwaway schema.

Run: uv run pytest -m db   (needs DATABASE_URL; the schema is dropped afterwards)
"""
import uuid

import numpy as np
import pytest

from medgraph.graph_store import GraphStore
from medgraph.llm_router import get_key

pytestmark = pytest.mark.db


def v(*xs):
    a = np.zeros(384, dtype=np.float32)
    a[: len(xs)] = xs
    return a / np.linalg.norm(a)


@pytest.fixture
def conn():
    if not get_key("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")
    from medgraph.pgstore import connect, init_schema

    schema = f"test_{uuid.uuid4().hex[:8]}"
    c = connect(schema=schema)
    init_schema(c)
    yield c
    c.execute(f'drop schema "{schema}" cascade')
    c.close()


CHUNKS = {
    "dengue:summary:0": {"id": "dengue:summary:0", "drug": "dengue", "section": "summary", "text": "mosquito virus"},
    "flu:summary:0": {"id": "flu:summary:0", "drug": "flu", "section": "summary", "text": "cough"},
    "dengue:summary:1": {"id": "dengue:summary:1", "drug": "dengue", "section": "summary", "text": "rash"},
}


def test_vector_search_and_allowed_filter(conn):
    from medgraph.pgstore import PgVectorIndex, read_chunks, write_chunks

    write_chunks(conn, "t", list(CHUNKS.values()), np.stack([v(1, 0), v(0, 1), v(0.9, 0.1)]))
    write_chunks(conn, "other", [CHUNKS["flu:summary:0"]], np.stack([v(1, 0)]))  # other dataset is invisible
    assert read_chunks(conn, "t") == CHUNKS
    idx = PgVectorIndex(conn, "t", dict(CHUNKS))
    res = idx.search(v(1, 0), 2)
    assert [r[0] for r in res] == ["dengue:summary:0", "dengue:summary:1"] and res[0][1] == pytest.approx(1, abs=1e-4)
    assert [r[0] for r in idx.search(v(1, 0), 3, allowed={"flu:summary:0"})] == ["flu:summary:0"]


def test_vector_upsert_replaces_row(conn):
    from medgraph.pgstore import PgVectorIndex, write_chunks

    write_chunks(conn, "t", list(CHUNKS.values()), np.stack([v(1, 0), v(0, 1), v(0.9, 0.1)]))
    chunks = dict(CHUNKS)
    chunks["new:summary:0"] = {"id": "new:summary:0", "drug": "new", "section": "summary", "text": "n"}
    idx = PgVectorIndex(conn, "t", chunks)
    idx.upsert(["new:summary:0"], np.stack([v(0, 0, 1)]))
    assert idx.search(v(0, 0, 1), 1)[0][0] == "new:summary:0" and "new:summary:0" in idx.ids


def test_graph_roundtrip(conn):
    from medgraph.pgstore import load_graph, save_graph

    g = GraphStore()
    g.add_extraction("dengue", [{"name": "mosquito bites", "type": "Topic"}], [
        {"source": "dengue", "target": "mosquito bites", "type": "MENTIONS", "chunk_id": "dengue:summary:0",
         "evidence": "an infected mosquito bites you"}])
    save_graph(conn, "t", g)
    save_graph(conn, "t", g)  # idempotent
    h = load_graph(conn, "t")
    assert h.node_types() == g.node_types()
    assert h.edges_between("dengue", "mosquito bites") == g.edges_between("dengue", "mosquito bites")


def test_cache_lookup_key_threshold_and_invalidation(conn):
    from medgraph.pgstore import PgSemanticCache

    c = PgSemanticCache(conn, "t", threshold=0.9)
    c.store("mosquito diseases?", v(1, 0.1), "mosquito bites", {"text": "A"}, ["mosquito bites", "dengue"])
    c.store("tick diseases?", v(0, 1), "tick bites", {"text": "B"}, ["tick bites", "lyme disease"])
    hit, s = c.lookup(v(1, 0.12), "mosquito bites")
    assert hit.answer == {"text": "A"} and s > 0.99
    assert c.lookup(v(1, 0.12), "tick bites")[0] is None  # same meaning, different entity → never
    assert c.lookup(v(1, 1), "mosquito bites") == (None, pytest.approx(0.77, abs=0.02))
    assert c.nearest(v(0, 1))[0].query == "tick diseases?"
    evicted = c.invalidate_nodes({"dengue", "zika"})
    assert [e.query for e in evicted] == ["mosquito diseases?"]
    assert [e.query for e in c.entries] == ["tick diseases?"]
    assert PgSemanticCache(conn, "other").entries == []  # datasets are isolated


def test_cache_export_import(conn, tmp_path):
    from medgraph.pgstore import PgSemanticCache

    c = PgSemanticCache(conn, "t")
    c.store("q", v(1, 0), "k", {"text": "A"}, ["n"])
    c.save(tmp_path / "warm.json")
    d = PgSemanticCache(conn, "t2")
    d.load(tmp_path / "warm.json")
    assert [e.query for e in d.entries] == ["q"] and d.lookup(v(1, 0), "k")[0].answer == {"text": "A"}
