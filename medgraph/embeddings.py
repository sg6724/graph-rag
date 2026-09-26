"""Local CPU embeddings (fastembed) and a tiny brute-force cosine index."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medgraph import config


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


def doc_text(chunk: dict) -> str:
    return f"{chunk['drug']} {chunk['section'].replace('_', ' ')}: {chunk['text']}"


class Embedder:
    def __init__(self, model_name: str = config.EMBED_MODEL):
        from fastembed import TextEmbedding

        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(config.MODEL_DIR))

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self.model.embed(texts, batch_size=32)), dtype=np.float32))  # small batches: low RAM

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(list(self.model.query_embed([text])), dtype=np.float32))[0]


class VectorIndex:
    def __init__(self, ids: list[str], matrix: np.ndarray):
        self.ids = list(ids)
        self.matrix = _normalize(np.asarray(matrix, dtype=np.float32))
        self._pos = {cid: i for i, cid in enumerate(self.ids)}

    def search(self, qvec: np.ndarray, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        scores = self.matrix @ np.asarray(qvec, dtype=np.float32)
        if allowed is not None:
            mask = np.array([cid in allowed for cid in self.ids])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in order if np.isfinite(scores[i])]

    def upsert(self, ids: list[str], matrix: np.ndarray) -> None:
        matrix = _normalize(np.asarray(matrix, dtype=np.float32))
        for cid, row in zip(ids, matrix):
            if cid in self._pos:
                self.matrix[self._pos[cid]] = row
            else:
                self._pos[cid] = len(self.ids)
                self.ids.append(cid)
                self.matrix = np.vstack([self.matrix, row[None, :]])

    def save(self, emb_path: Path = config.EMB_PATH, ids_path: Path = config.EMB_IDS_PATH) -> None:
        np.save(emb_path, self.matrix)
        Path(ids_path).write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, emb_path: Path = config.EMB_PATH, ids_path: Path = config.EMB_IDS_PATH) -> "VectorIndex":
        return cls(json.loads(Path(ids_path).read_text(encoding="utf-8")), np.load(emb_path))


def main() -> None:
    from medgraph.ingest import load_chunks

    chunks = load_chunks()
    ids = list(chunks)
    matrix = Embedder().embed_docs([doc_text(chunks[i]) for i in ids])
    VectorIndex(ids, matrix).save()
    print(f"indexed {len(ids)} chunks, dim={matrix.shape[1]}")


if __name__ == "__main__":
    main()
