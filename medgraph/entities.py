"""Deterministic (no-LLM) recognition of drugs/enzymes/classes in user questions."""
from __future__ import annotations

import difflib
import re

from medgraph.aliases import BRAND_ALIASES, normalize_text

MATCH_TYPES = {"Drug", "Enzyme", "DrugClass"}


def _contains(text: str, name: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", text) is not None


def canonicalize(question: str) -> str:
    text = normalize_text(question)
    for brand, generic in BRAND_ALIASES.items():
        text = re.sub(rf"(?<![a-z0-9]){re.escape(brand)}(?![a-z0-9])", generic, text)
    return text


def match_entities(question: str, node_types: dict[str, str]) -> list[str]:
    text = canonicalize(question)
    found: set[str] = set()
    for name, t in node_types.items():
        if t in MATCH_TYPES and len(name) >= 3 and _contains(text, name):
            found.add(name)
    drugs = [n for n, t in node_types.items() if t == "Drug"]
    for token in re.findall(r"[a-z]{6,}", text):
        if token in found:
            continue
        # 0.92: catches "simvastatine" (0.96) / "clarithromicin" (0.93) but not "prednisolone"→prednisone (0.91)
        close = difflib.get_close_matches(token, drugs, n=1, cutoff=0.92)
        if close:
            found.add(close[0])
    return sorted(found)


_topic_index: dict[tuple, list[tuple[str, str]]] = {}


def match_topics(question: str, node_types: dict[str, str], aliases: dict[str, str]) -> list[str]:
    """MedlinePlus: find Topic nodes by title or synonym, plural-insensitive, longest match wins."""
    from medgraph.medline import plural_variants

    # keyed by content, not id(): callers pass a fresh dict each time and ids get reused across graphs
    key = (frozenset(n for n, t in node_types.items() if t == "Topic"), frozenset(aliases.items()))
    if key not in _topic_index:  # (normalized phrase, topic) sorted longest first; built once per graph
        pairs = [(plural_variants(n), n) for n, t in node_types.items() if t == "Topic" and len(n) >= 3]
        pairs += [(a, n) for a, n in aliases.items() if node_types.get(n) == "Topic" and len(a) >= 3]
        _topic_index[key] = sorted(pairs, key=lambda p: -len(p[0]))
    text = plural_variants(normalize_text(question))
    found: set[str] = set()
    for phrase, topic in _topic_index[key]:
        m = re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text)
        if m:
            found.add(topic)
            text = text[:m.start()] + " " * (m.end() - m.start()) + text[m.end():]  # consume: longest wins
    return sorted(found)


def drug_key(entities: list[str], node_types: dict[str, str]) -> str:
    return "|".join(sorted(e for e in entities if node_types.get(e) == "Drug"))
