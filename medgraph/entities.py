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


def drug_key(entities: list[str], node_types: dict[str, str]) -> str:
    return "|".join(sorted(e for e in entities if node_types.get(e) == "Drug"))
