"""Knowledge graph over drug-label facts; every edge keeps its source chunk (provenance)."""
from __future__ import annotations

import json
import sys
from itertools import islice
from pathlib import Path

import networkx as nx

from medgraph import config

# Edge types most useful as context, first
EDGE_PRIORITY = ["INTERACTS_WITH", "CONTRAINDICATED_IN", "INHIBITS", "INDUCES",
                 "METABOLIZED_BY", "BELONGS_TO", "CAUSES"]


class GraphStore:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()
        self.version = 0
        self._und: nx.Graph | None = None  # cached undirected view; reset on every mutation

    # ---- building -------------------------------------------------------
    def add_entity(self, name: str, type_: str) -> None:
        if name not in self.g:
            self.g.add_node(name, type=type_)
            self._und = None
        elif type_ == "Drug" or self.g.nodes[name].get("type") == "Unknown":
            self.g.nodes[name]["type"] = type_

    def add_relation(self, source: str, target: str, type_: str, chunk_id: str,
                     drug: str, evidence: str = "") -> None:
        for n in (source, target):
            if n not in self.g:
                self.g.add_node(n, type="Unknown")
        self.g.add_edge(source, target, key=f"{type_}|{chunk_id}", type=type_,
                        chunk_id=chunk_id, drug=drug, evidence=evidence)
        self._und = None

    def add_extraction(self, drug: str, entities: list[dict], relations: list[dict]) -> None:
        self.add_entity(drug, "Drug")
        for e in entities:
            self.add_entity(e["name"], e["type"])
        for r in relations:
            self.add_relation(r["source"], r["target"], r["type"], r["chunk_id"], drug, r.get("evidence", ""))
        self.version += 1

    def _signature(self, chunk_ids: set[str]) -> set[tuple[str, str, str]]:
        return {(u, v, d["type"]) for u, v, d in self.g.edges(data=True) if d["chunk_id"] in chunk_ids}

    def replace_chunks(self, drug: str, chunk_ids: set[str], entities: list[dict],
                       relations: list[dict]) -> set[str]:
        """Swap the facts sourced from `chunk_ids`; return nodes whose facts changed (always incl. drug)."""
        before = self._signature(chunk_ids)
        stale = [(u, v, k) for u, v, k, d in self.g.edges(keys=True, data=True) if d["chunk_id"] in chunk_ids]
        self.g.remove_edges_from(stale)
        self.add_extraction(drug, entities, relations)  # bumps version once
        after = self._signature(chunk_ids)
        changed = {drug}
        for u, v, _ in before ^ after:
            changed.update((u, v))
        self.g.remove_nodes_from([n for n in list(self.g.nodes) if self.g.degree(n) == 0 and n != drug])
        self._und = None
        return changed

    # ---- querying -------------------------------------------------------
    def node_types(self) -> dict[str, str]:
        return {n: d.get("type", "Unknown") for n, d in self.g.nodes(data=True)}

    def _undirected(self) -> nx.Graph:
        if self._und is None:
            self._und = nx.Graph(self.g)
        return self._und

    def paths_between(self, a: str, b: str, max_len: int = config.MAX_PATH_LEN) -> list[list[str]]:
        if a not in self.g or b not in self.g or a == b:
            return []
        und = self._undirected()
        if not nx.has_path(und, a, b):
            return []
        out = []
        for path in islice(nx.shortest_simple_paths(und, a, b), config.MAX_PATHS):
            if len(path) - 1 > max_len:
                break
            out.append(path)
        return out

    def edges_between(self, u: str, v: str) -> list[dict]:
        out = []
        for a, b in ((u, v), (v, u)):
            if self.g.has_edge(a, b):
                for d in self.g.get_edge_data(a, b).values():
                    out.append({"source": a, "target": b, **d})
        return out

    def neighborhood_edges(self, nodes: list[str], limit: int) -> list[dict]:
        edges = []
        for n in nodes:
            if n not in self.g:
                continue
            for u, v, d in list(self.g.out_edges(n, data=True)) + list(self.g.in_edges(n, data=True)):
                edges.append({"source": u, "target": v, **d})
        rank = {t: i for i, t in enumerate(EDGE_PRIORITY)}
        edges.sort(key=lambda e: rank.get(e["type"], len(rank)))
        return edges[:limit]

    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        for _, t in self.node_types().items():
            by_type[t] = by_type.get(t, 0) + 1
        edge_types: dict[str, int] = {}
        for _, _, d in self.g.edges(data=True):
            edge_types[d["type"]] = edge_types.get(d["type"], 0) + 1
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(),
                "node_types": by_type, "edge_types": edge_types}

    # ---- persistence ----------------------------------------------------
    def save(self, path: Path = config.GRAPH_PATH) -> None:
        data = {
            "version": self.version,
            "nodes": [{"name": n, "type": d.get("type", "Unknown")} for n, d in self.g.nodes(data=True)],
            "edges": [{"source": u, "target": v, **d} for u, v, d in self.g.edges(data=True)],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = config.GRAPH_PATH) -> "GraphStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        gs = cls()
        for n in data["nodes"]:
            gs.g.add_node(n["name"], type=n["type"])
        for e in data["edges"]:
            gs.add_relation(e["source"], e["target"], e["type"], e["chunk_id"], e["drug"], e.get("evidence", ""))
        gs.version = data.get("version", 0)
        return gs


def main() -> None:
    gs = GraphStore.load()
    print(json.dumps(gs.stats(), indent=1))
    if len(sys.argv) == 3:
        a, b = sys.argv[1], sys.argv[2]
        for path in gs.paths_between(a, b):
            print(" -> ".join(path))
            for u, v in zip(path, path[1:]):
                for e in gs.edges_between(u, v):
                    print(f"    {e['source']} --{e['type']}--> {e['target']}  [{e['chunk_id']}]")


if __name__ == "__main__":
    main()
