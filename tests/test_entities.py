from medgraph.entities import canonicalize, drug_key, match_entities

TYPES = {"warfarin": "Drug", "aspirin": "Drug", "ibuprofen": "Drug", "simvastatin": "Drug",
         "clarithromycin": "Drug", "cyp3a4": "Enzyme", "nsaids": "DrugClass",
         "bleeding": "SideEffect", "prednisone": "Drug"}


def test_generic_names():
    assert match_entities("Can I take aspirin with warfarin?", TYPES) == ["aspirin", "warfarin"]


def test_brand_names_map_to_generic():
    assert match_entities("Is Biaxin OK with Zocor?", TYPES) == ["clarithromycin", "simvastatin"]


def test_typo_tolerance():
    assert match_entities("simvastatine and clarithromicin together?", TYPES) == ["clarithromycin", "simvastatin"]


def test_enzyme_and_class_matched_but_not_side_effects():
    assert match_entities("Does CYP 3A4 matter for NSAIDs and bleeding?", TYPES) == ["cyp3a4", "nsaids"]


def test_word_boundaries():
    assert match_entities("prednisolone dosing", TYPES) == []


def test_drug_key_only_drugs_and_order_independent():
    a = drug_key(match_entities("warfarin + aspirin, cyp3a4?", TYPES), TYPES)
    b = drug_key(match_entities("aspirin with warfarin", TYPES), TYPES)
    c = drug_key(match_entities("warfarin with ibuprofen", TYPES), TYPES)
    assert a == b == "aspirin|warfarin"
    assert c == "ibuprofen|warfarin"


def test_canonicalize():
    assert canonicalize("Is Biaxin OK with  Zocor and CYP 3A4?") == "is clarithromycin ok with simvastatin and cyp3a4?"
