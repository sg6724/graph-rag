from pathlib import Path

from medgraph.medline import build_medline, parse_topics, plural_variants, topic_slug

XML = """<health-topics>
<health-topic title="Dengue" url="https://medlineplus.gov/dengue.html" id="1" language="English">
<also-called>Break-bone fever</also-called>
<full-summary>&lt;h3&gt;What is dengue?&lt;/h3&gt;&lt;p&gt;Dengue is a viral infection. You can get it if an infected
&lt;a href="https://medlineplus.gov/mosquitobites.html"&gt;mosquito&lt;/a&gt; bites you. It is common in warm areas.&lt;/p&gt;
&lt;h3&gt;What are the symptoms?&lt;/h3&gt;&lt;ul&gt;&lt;li&gt;A high &lt;a href="https://medlineplus.gov/fever.html"&gt;fever&lt;/a&gt;&lt;/li&gt;&lt;li&gt;Rash&lt;/li&gt;&lt;/ul&gt;</full-summary>
<group url="https://medlineplus.gov/infections.html" id="12">Infections</group>
<related-topic url="https://medlineplus.gov/mosquitobites.html" id="2">Mosquito Bites</related-topic>
</health-topic>
<health-topic title="Mosquito Bites" url="https://medlineplus.gov/mosquitobites.html" id="2" language="English">
<full-summary>&lt;p&gt;Mosquitoes can spread diseases.&lt;/p&gt;</full-summary>
<group url="https://medlineplus.gov/infections.html" id="12">Infections</group>
</health-topic>
<health-topic title="Fever" url="https://medlineplus.gov/fever.html" id="3" language="English">
<full-summary>&lt;p&gt;A fever is a body temperature above normal.&lt;/p&gt;</full-summary>
</health-topic>
<health-topic title="Dengue (Spanish)" url="https://medlineplus.gov/spanish/dengue.html" id="9" language="Spanish">
<full-summary>&lt;p&gt;x&lt;/p&gt;</full-summary>
</health-topic>
</health-topics>"""


def write(tmp_path: Path) -> Path:
    p = tmp_path / "t.xml"
    p.write_text(XML, encoding="utf-8")
    return p


def test_topic_slug():
    assert topic_slug("https://medlineplus.gov/mosquitobites.html") == "mosquitobites"


def test_parse_topics_english_only_with_fields(tmp_path):
    topics = parse_topics(write(tmp_path))
    assert [t["title"] for t in topics] == ["Dengue", "Mosquito Bites", "Fever"]
    d = topics[0]
    assert d["slug"] == "dengue" and d["also_called"] == ["Break-bone fever"]
    assert d["groups"] == ["Infections"] and d["related"] == ["mosquitobites"]


def test_build_medline_chunks_are_sections_with_titles(tmp_path):
    chunks, _, _, _ = build_medline(parse_topics(write(tmp_path)))
    ids = [c["id"] for c in chunks]
    assert ids[:2] == ["dengue:summary:0", "dengue:summary:1"]
    c0 = chunks[0]
    assert c0["drug"] == "dengue" and c0["section"] == "summary"
    assert c0["text"].startswith("Dengue — What is dengue?") and "<" not in c0["text"]


def test_build_medline_graph_edges_have_sentence_evidence(tmp_path):
    _, entities, relations, _ = build_medline(parse_topics(write(tmp_path)))
    types = {e["name"]: e["type"] for e in entities}
    assert types["dengue"] == "Topic" and types["infections"] == "Group"
    mention = [r for r in relations if r["type"] == "MENTIONS" and r["target"] == "mosquito bites"][0]
    assert mention["source"] == "dengue" and mention["chunk_id"] == "dengue:summary:0"
    assert "infected mosquito bites you" in mention["evidence"]
    assert {"source": "dengue", "target": "fever", "type": "MENTIONS"}.items() <= \
        [r for r in relations if r["target"] == "fever"][0].items()
    assert any(r["type"] == "RELATED_TO" and r["target"] == "mosquito bites" for r in relations)
    assert any(r["type"] == "IN_GROUP" and r["target"] == "infections" for r in relations)
    # no self loops, no duplicate (source, target, type, chunk) edges
    keys = [(r["source"], r["target"], r["type"], r["chunk_id"]) for r in relations]
    assert len(keys) == len(set(keys)) and all(r["source"] != r["target"] for r in relations)


def test_build_medline_aliases_include_synonyms_and_short_forms(tmp_path):
    _, _, _, aliases = build_medline(parse_topics(write(tmp_path)))
    assert aliases["break-bone fever"] == "dengue"
    assert aliases["mosquito"] == "mosquito bites"  # "X bites" is also found by "X"


def test_plural_variants():
    assert plural_variants("mosquitoes spread ticks and diseases") == "mosquito spread tick and disease"
