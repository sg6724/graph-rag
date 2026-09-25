"""Fetch FDA drug labels from openFDA and split them into citable chunks."""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from medgraph import config


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def label_sections(label: dict) -> dict[str, str]:
    out = {}
    for section in config.SECTIONS:
        values = label.get(section)
        if not values:
            continue
        text = " ".join(" ".join(values).split())[: config.SECTION_CHAR_CAP]
        if text:
            out[section] = text
    return out


def build_chunks(drug: str, label: dict) -> list[dict]:
    chunks = []
    for section, text in label_sections(label).items():
        for i, piece in enumerate(chunk_text(text)):
            chunks.append({"id": f"{drug}:{section}:{i}", "drug": drug, "section": section, "text": piece})
    return chunks


def _score(drug: str, label: dict) -> tuple[bool, bool, int]:
    names = [n.lower() for n in label.get("openfda", {}).get("generic_name", [])]
    single = len(names) == 1 and drug in names[0] and " and " not in names[0] and "," not in names[0]
    has_interactions = bool(label.get("drug_interactions"))
    size = sum(len(" ".join(label.get(s, []))) for s in config.SECTIONS)
    return (single, has_interactions, size)


def pick_label(drug: str, results: list[dict]) -> dict | None:
    if not results:
        return None
    return max(results, key=lambda lbl: _score(drug, lbl))


def fetch_label(drug: str) -> dict | None:
    params = {"search": f'openfda.generic_name:"{drug}"', "limit": 20}
    try:
        r = httpx.get(config.OPENFDA_URL, params=params, timeout=30)
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    return pick_label(drug, r.json().get("results", []))


def save_chunks(chunks: list[dict], path: Path = config.CHUNKS_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_chunks(path: Path = config.CHUNKS_PATH) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                out[c["id"]] = c
    return out


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []
    for drug in config.DRUGS:
        label = fetch_label(drug)
        if label is None:
            print(f"SKIP {drug}: no label found")
            continue
        (config.RAW_DIR / f"{drug}.json").write_text(json.dumps(label), encoding="utf-8")
        chunks = build_chunks(drug, label)
        all_chunks.extend(chunks)
        print(f"{drug:15s} {len(chunks):3d} chunks  sections={sorted({c['section'] for c in chunks})}")
    save_chunks(all_chunks)
    print(f"TOTAL {len(all_chunks)} chunks from {len({c['drug'] for c in all_chunks})} drugs")


if __name__ == "__main__":
    main()
