from medgraph.graph_store import GraphStore


def sample() -> GraphStore:
    g = GraphStore()
    g.add_extraction("simvastatin", [{"name": "cyp3a4", "type": "Enzyme"}], [
        {"source": "simvastatin", "target": "cyp3a4", "type": "METABOLIZED_BY",
         "chunk_id": "simvastatin:clinical_pharmacology:0", "evidence": "metabolized by CYP3A4"}])
    g.add_extraction("clarithromycin", [{"name": "cyp3a4", "type": "Enzyme"}], [
        {"source": "clarithromycin", "target": "cyp3a4", "type": "INHIBITS",
         "chunk_id": "clarithromycin:drug_interactions:0", "evidence": "strong CYP3A4 inhibitor"}])
    return g


def test_types_and_drug_node():
    g = sample()
    t = g.node_types()
    assert t["simvastatin"] == "Drug" and t["cyp3a4"] == "Enzyme"


def test_paths_ignore_edge_direction():
    g = sample()
    assert g.paths_between("simvastatin", "clarithromycin") == [["simvastatin", "cyp3a4", "clarithromycin"]]
    assert g.paths_between("simvastatin", "nonexistent") == []


def test_edges_between_both_directions():
    g = sample()
    edges = g.edges_between("cyp3a4", "clarithromycin")
    assert len(edges) == 1
    assert edges[0]["source"] == "clarithromycin" and edges[0]["type"] == "INHIBITS"
    assert edges[0]["drug"] == "clarithromycin"


def test_neighborhood_edges_limit_and_priority():
    g = sample()
    g.add_relation("simvastatin", "headache", "CAUSES", "simvastatin:warnings:0", "simvastatin")
    edges = g.neighborhood_edges(["simvastatin"], limit=1)
    assert len(edges) == 1 and edges[0]["type"] == "METABOLIZED_BY"


def test_replace_chunks_reports_changed_nodes_and_drops_orphans():
    g = sample()
    g.add_relation("simvastatin", "headache", "CAUSES", "simvastatin:warnings:0", "simvastatin")
    v0 = g.version
    changed = g.replace_chunks("simvastatin", {"simvastatin:warnings:0"},
                               [{"name": "rhabdomyolysis", "type": "SideEffect"}],
                               [{"source": "simvastatin", "target": "rhabdomyolysis", "type": "CAUSES",
                                 "chunk_id": "simvastatin:warnings:0", "evidence": "rhabdo"}])
    assert changed == {"simvastatin", "headache", "rhabdomyolysis"}
    assert "headache" not in g.g  # orphan removed
    assert g.g.has_edge("simvastatin", "cyp3a4")  # untouched chunk kept
    assert g.version == v0 + 1  # one replace = one version bump (cached path search invalidates)


def test_save_load_roundtrip(tmp_path):
    g = sample()
    p = tmp_path / "g.json"
    g.save(p)
    h = GraphStore.load(p)
    assert h.node_types() == g.node_types()
    assert h.edges_between("simvastatin", "cyp3a4") == g.edges_between("simvastatin", "cyp3a4")


def test_paths_see_edges_added_after_a_search():
    g = sample()
    assert g.paths_between("simvastatin", "clarithromycin")  # builds the cached undirected view
    g.add_relation("ketoconazole", "cyp3a4", "INHIBITS", "ketoconazole:drug_interactions:0", "ketoconazole")
    assert g.paths_between("simvastatin", "ketoconazole") == [["simvastatin", "cyp3a4", "ketoconazole"]]


def test_add_extraction_source_type_is_configurable():
    g = GraphStore()
    g.add_extraction("dengue", [], [{"source": "dengue", "target": "fever", "type": "MENTIONS",
                                     "chunk_id": "dengue:summary:0", "evidence": "e"}], source_type="Topic")
    assert g.node_types()["dengue"] == "Topic"
    g.replace_chunks("dengue", {"dengue:summary:1"}, [], [], source_type="Topic")
    assert g.node_types()["dengue"] == "Topic"
