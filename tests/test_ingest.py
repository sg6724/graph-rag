from medgraph.ingest import build_chunks, chunk_text, label_sections, load_chunks, pick_label, save_chunks


def test_chunk_text_short_text_is_one_chunk():
    assert chunk_text("hello   world", size=50, overlap=10) == ["hello world"]
    assert chunk_text("", size=50, overlap=10) == []


def test_chunk_text_respects_size_and_overlaps():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 5
    assert all(len(c) <= 100 for c in chunks)
    # consecutive chunks share some text (overlap)
    assert chunks[0][-10:].split()[-1] in chunks[1]
    # nothing lost at the end
    assert chunks[-1].endswith("word199")


def test_label_sections_cleans_and_caps():
    label = {"drug_interactions": ["7 DRUG   INTERACTIONS\n x" + "a" * 20000],
             "contraindications": ["None."], "unrelated": ["zzz"]}
    secs = label_sections(label)
    assert set(secs) == {"drug_interactions", "contraindications"}
    assert "  " not in secs["drug_interactions"]
    assert len(secs["drug_interactions"]) <= 15000


def test_build_chunks_ids_and_fields():
    label = {"drug_interactions": ["Clarithromycin increases exposure. " * 40],
             "contraindications": ["Pregnancy."]}
    chunks = build_chunks("simvastatin", label)
    ids = [c["id"] for c in chunks]
    assert ids[0] == "simvastatin:contraindications:0"
    assert "simvastatin:drug_interactions:0" in ids and "simvastatin:drug_interactions:1" in ids
    assert all(c["drug"] == "simvastatin" and c["text"] for c in chunks)


def test_pick_label_prefers_single_ingredient_with_interactions():
    combo = {"openfda": {"generic_name": ["ASPIRIN AND DIPYRIDAMOLE"]},
             "drug_interactions": ["x" * 9000]}
    otc = {"openfda": {"generic_name": ["ASPIRIN"]}, "warnings": ["y" * 3000]}
    rx = {"openfda": {"generic_name": ["ASPIRIN"]}, "drug_interactions": ["z" * 2000]}
    assert pick_label("aspirin", [combo, otc, rx]) is rx
    assert pick_label("aspirin", []) is None


def test_chunks_roundtrip(tmp_path):
    p = tmp_path / "c.jsonl"
    chunks = [{"id": "a:b:0", "drug": "a", "section": "b", "text": "t"}]
    save_chunks(chunks, p)
    assert load_chunks(p) == {"a:b:0": chunks[0]}
