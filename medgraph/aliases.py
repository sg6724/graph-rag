"""Name normalization shared by graph extraction and query matching."""
import re

BRAND_ALIASES = {
    "coumadin": "warfarin", "jantoven": "warfarin", "bayer": "aspirin", "advil": "ibuprofen",
    "motrin": "ibuprofen", "aleve": "naproxen", "naprosyn": "naproxen", "plavix": "clopidogrel",
    "zocor": "simvastatin", "lipitor": "atorvastatin", "biaxin": "clarithromycin",
    "ery-tab": "erythromycin", "nizoral": "ketoconazole", "diflucan": "fluconazole",
    "zoloft": "sertraline", "prozac": "fluoxetine", "ultram": "tramadol", "glucophage": "metformin",
    "zestril": "lisinopril", "prinivil": "lisinopril", "aldactone": "spironolactone",
    "lanoxin": "digoxin", "cordarone": "amiodarone", "pacerone": "amiodarone", "prilosec": "omeprazole",
    "rifadin": "rifampin", "rifampicin": "rifampin", "tegretol": "carbamazepine",
    "lithobid": "lithium", "trexall": "methotrexate", "zyloprim": "allopurinol",
    "viagra": "sildenafil", "revatio": "sildenafil", "nitrostat": "nitroglycerin",
    "synthroid": "levothyroxine", "levoxyl": "levothyroxine", "deltasone": "prednisone",
    "tylenol": "acetaminophen", "paracetamol": "acetaminophen",
}

_CYP = re.compile(r"\b(?:cyp|cytochrome\s*p-?\s*450)\s*-?\s*(\d)\s*([a-z])\s*(\d+)\b")
_PGP = {"p-glycoprotein", "p glycoprotein", "pgp", "p-gp", "p-gp transporter", "abcb1"}


def normalize_text(text: str) -> str:
    t = " ".join(text.lower().split())
    return _CYP.sub(lambda m: f"cyp{m.group(1)}{m.group(2)}{m.group(3)}", t)


def normalize_name(name: str) -> str:
    n = normalize_text(name).strip()
    if n in _PGP:
        return "p-gp"
    return BRAND_ALIASES.get(n, n)
