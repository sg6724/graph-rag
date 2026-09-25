"""Project-wide settings: paths, corpus, models, retrieval and cache knobs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
CHUNKS_PATH = DATA / "chunks.jsonl"
GRAPH_PATH = DATA / "graph.json"
EMB_PATH = DATA / "embeddings.npy"
EMB_IDS_PATH = DATA / "embedding_ids.json"
LLM_CACHE_DIR = DATA / "llm_cache"
CACHE_PATH = DATA / "cache.json"
MODEL_DIR = DATA / "models"
DEMO_UPDATES_PATH = DATA / "demo_updates.json"

OPENFDA_URL = "https://api.fda.gov/drug/label.json"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DRUGS = [
    "warfarin", "aspirin", "ibuprofen", "naproxen", "clopidogrel", "simvastatin",
    "atorvastatin", "clarithromycin", "erythromycin", "ketoconazole", "fluconazole",
    "sertraline", "fluoxetine", "tramadol", "metformin", "lisinopril", "spironolactone",
    "digoxin", "amiodarone", "omeprazole", "rifampin", "carbamazepine", "lithium",
    "methotrexate", "allopurinol", "sildenafil", "nitroglycerin", "levothyroxine",
    "prednisone", "acetaminophen",
]
SECTIONS = [
    "boxed_warning", "contraindications", "drug_interactions", "warnings_and_cautions",
    "warnings", "clinical_pharmacology", "do_not_use", "ask_doctor_or_pharmacist",
]
SECTION_CHAR_CAP = 15000
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# (provider, model) fallback chains per task. Free tiers: Gemini = 20 requests/day *per model*;
# OpenRouter = 50 free requests/day per account. Measured 2026-09-25 (see spec §6).
PROVIDERS = {
    # live answers: fast Gemini Flash models first, each with its own daily quota
    "answer": [
        ("gemini", "gemini-3.8-flash"),
        ("gemini", "gemini-3.6-flash"),
        ("gemini", "gemini-3.5-flash-lite"),
        ("gemini", "gemini-3.5-flash"),
        ("gemini", "gemini-3.7-flash"),
        ("openrouter", "qwen/qwen3.8-27b:free"),
        ("openrouter", "openrouter/free"),
    ],
    # offline graph extraction + judging: quality over speed
    "extract": [
        ("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
        ("gemini", "gemini-3.6-flash"),
        ("gemini", "gemini-3.5-flash-lite"),
    ],
}
# "extract_fast" (live label-update demo) has no chain of its own → uses the "answer" chain
TIMEOUTS = {"answer": 60.0, "extract": 600.0, "extract_fast": 90.0}
EXTRACT_WORKERS = 5  # parallel labels during graph build (OpenRouter free: 20 req/min)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K = 6
MAX_PATH_LEN = 3
MAX_PATHS = 8
NEIGHBOR_EDGE_LIMIT = 30
MAX_FACTS = 40
MAX_CONTEXT_CHUNKS = 8

CACHE_THRESHOLD = 0.90
REF_USD_PER_CALL = 0.01  # illustrative paid-model price per answer call, for "$ saved"
