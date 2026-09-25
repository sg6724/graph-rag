"""Verify each question's evidence terms appear in its drugs' labels (no LLM calls)."""
import json
from pathlib import Path

from medgraph.ingest import load_chunks

QUESTIONS = Path(__file__).with_name("questions.json")


def main() -> int:
    chunks = load_chunks()
    text_by_drug: dict[str, str] = {}
    for c in chunks.values():
        text_by_drug[c["drug"]] = text_by_drug.get(c["drug"], "") + " " + c["text"].lower().replace("cyp ", "cyp")
    bad = 0
    for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]:
        missing_drugs = [d for d in q["drugs"] if d not in text_by_drug]
        corpus = " ".join(text_by_drug.get(d, "") for d in q["drugs"])
        missing_terms = [t for t in q["evidence_terms"] if t.lower() not in corpus]
        if missing_drugs or missing_terms:
            bad += 1
            print(f"{q['id']}: missing drugs={missing_drugs} terms={missing_terms}")
    print(f"{bad} question(s) need fixing")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
