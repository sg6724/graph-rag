"""MedlinePlus health topics → cited chunks + a deterministic knowledge graph (no LLM needed).

Graph edges come from the data itself: links inside each summary (MENTIONS, with the sentence as evidence),
curated related topics (RELATED_TO) and body-system groups (IN_GROUP).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

from medgraph.ingest import chunk_text

_LINK = re.compile(r"^https?://medlineplus\.gov/([a-z0-9]+)\.html$")
_SENT_END = re.compile(r"(?<=[.!?])\s+|\n+")
SHORT_FORM_SUFFIXES = ("bites", "stings")  # "mosquito bites" is also what people mean by "mosquito"


def topic_slug(url: str) -> str | None:
    m = _LINK.match(url or "")
    return m.group(1) if m else None


def plural_variants(text: str) -> str:
    """Crude singularization applied to both questions and names, so 'mosquitoes' matches 'mosquito'."""
    out = []
    for w in text.lower().split():
        if len(w) > 4 and w.endswith("oes"):
            w = w[:-2]
        elif len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        out.append(w)
    return " ".join(out)


def parse_topics(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    topics = []
    for t in root.findall("health-topic"):
        if t.get("language") != "English":
            continue
        slug = topic_slug(t.get("url", ""))
        if not slug:
            continue
        topics.append({
            "slug": slug,
            "title": t.get("title", "").strip(),
            "also_called": [a.text.strip() for a in t.findall("also-called") if a.text],
            "groups": [g.text.strip() for g in t.findall("group") if g.text],
            "related": [s for s in (topic_slug(r.get("url", "")) for r in t.findall("related-topic")) if s],
            "summary_html": t.findtext("full-summary") or "",
        })
    return topics


class _Sections(HTMLParser):
    """Splits summary HTML at <h3> into (heading, text, links[(slug, char_offset)])."""

    def __init__(self) -> None:
        super().__init__()
        self.sections: list[dict] = [{"heading": "", "text": "", "links": []}]
        self._in_h3 = False
        self._href: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "h3":
            self.sections.append({"heading": "", "text": "", "links": []})
            self._in_h3 = True
        elif tag == "a":
            self._href = dict(attrs).get("href")
        elif tag in ("p", "li", "ul", "ol", "br"):
            self.sections[-1]["text"] += "\n"

    def handle_endtag(self, tag):
        if tag == "h3":
            self._in_h3 = False
        elif tag == "a":
            self._href = None

    def handle_data(self, data):
        data = re.sub(r"\s+", " ", data)  # source line breaks are formatting; only block tags end a line
        sec = self.sections[-1]
        if self._in_h3:
            sec["heading"] += data
            return
        slug = topic_slug(self._href or "")
        if slug:
            sec["links"].append((slug, len(sec["text"])))
        sec["text"] += data


def _sentence_at(text: str, offset: int) -> str:
    start = 0
    for m in _SENT_END.finditer(text):
        if m.end() > offset:
            return " ".join(text[start:m.start()].split())
        start = m.end()
    return " ".join(text[start:].split())


def build_medline(topics: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict[str, str]]:
    """Returns (chunks, entities, relations, aliases). Node names are lowercase topic titles."""
    name_of = {t["slug"]: t["title"].lower() for t in topics}
    chunks: list[dict] = []
    entities: dict[str, str] = {}
    relations: dict[tuple, dict] = {}
    aliases: dict[str, str] = {}

    def relate(source, target, type_, chunk_id, evidence):
        if source != target:
            relations.setdefault((source, target, type_, chunk_id),
                                 {"source": source, "target": target, "type": type_,
                                  "chunk_id": chunk_id, "evidence": evidence[:300]})

    for t in topics:
        node = name_of[t["slug"]]
        entities[node] = "Topic"
        for alt in t["also_called"]:
            aliases[plural_variants(alt)] = node
        words = node.split()
        if len(words) > 1 and words[-1] in SHORT_FORM_SUFFIXES:
            aliases[plural_variants(" ".join(words[:-1]))] = node

        parser = _Sections()
        parser.feed(t["summary_html"])
        topic_chunks: list[dict] = []
        for sec in parser.sections:
            body = " ".join(sec["text"].split())
            if not body:
                continue
            heading = " ".join(sec["heading"].split())
            prefix = f"{t['title']} — {heading}" if heading else t["title"]
            pieces = chunk_text(body)
            sec_chunks = []
            for piece in pieces:
                c = {"id": f"{t['slug']}:summary:{len(topic_chunks)}", "drug": t["slug"], "section": "summary",
                     "text": f"{prefix} {piece}"}
                topic_chunks.append(c)
                sec_chunks.append(c)
            for slug, offset in sec["links"]:
                if slug not in name_of:
                    continue
                sentence = _sentence_at(sec["text"], offset)
                home = next((c for c in sec_chunks if sentence[:60] in c["text"]), sec_chunks[0])
                relate(node, name_of[slug], "MENTIONS", home["id"], sentence)
        if not topic_chunks:
            continue
        chunks.extend(topic_chunks)
        first = topic_chunks[0]["id"]
        for slug in t["related"]:
            if slug in name_of:
                relate(node, name_of[slug], "RELATED_TO", first,
                       f"MedlinePlus lists {name_of[slug]} as related to {node}")
        for g in t["groups"]:
            group = g.lower()
            entities.setdefault(group, "Group")
            relate(node, group, "IN_GROUP", first, f"{t['title']} is in the MedlinePlus group '{g}'")

    ents = [{"name": n, "type": ty} for n, ty in entities.items()]
    return chunks, ents, list(relations.values()), aliases
