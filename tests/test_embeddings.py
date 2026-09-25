import numpy as np
import pytest

from medgraph.embeddings import Embedder, VectorIndex, doc_text


def idx():
    m = np.array([[1, 0, 0], [0, 1, 0], [0.8, 0.6, 0]], dtype=np.float32)
    return VectorIndex(["a", "b", "c"], m)


def test_search_orders_by_cosine():
    res = idx().search(np.array([1, 0, 0], dtype=np.float32), k=2)
    assert [r[0] for r in res] == ["a", "c"]
    assert res[0][1] == pytest.approx(1.0)


def test_search_allowed_filter():
    res = idx().search(np.array([1, 0, 0], dtype=np.float32), k=3, allowed={"b"})
    assert [r[0] for r in res] == ["b"]


def test_upsert_replaces_and_appends():
    i = idx()
    i.upsert(["a", "d"], np.array([[0, 0, 1], [0, 1, 0]], dtype=np.float32))
    assert i.ids == ["a", "b", "c", "d"]
    assert i.search(np.array([0, 0, 1], dtype=np.float32), k=1)[0][0] == "a"


def test_save_load(tmp_path):
    i = idx()
    i.save(tmp_path / "e.npy", tmp_path / "ids.json")
    j = VectorIndex.load(tmp_path / "e.npy", tmp_path / "ids.json")
    assert j.ids == i.ids and np.allclose(j.matrix, i.matrix)


def test_doc_text_includes_drug_and_section():
    assert doc_text({"drug": "warfarin", "section": "drug_interactions", "text": "x"}) == \
        "warfarin drug interactions: x"


@pytest.mark.live
def test_embedder_semantics():
    e = Embedder()
    a = e.embed_query("Can I take simvastatin with clarithromycin?")
    b = e.embed_query("Is clarithromycin safe with simvastatin?")
    c = e.embed_query("What is the capital of France?")
    assert float(a @ b) > float(a @ c)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-4)
