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


TOPICS = {"dengue": "Topic", "mosquito bites": "Topic", "tick bites": "Topic", "fever": "Topic",
          "yellow fever": "Topic", "rashes": "Topic", "infections": "Group", "diabetes": "Topic"}
ALIASES = {"break-bone fever": "dengue", "mosquito": "mosquito bites", "tick": "tick bites"}


def test_match_topics_plurals_aliases_and_longest_match():
    from medgraph.entities import match_topics

    assert match_topics("Which diseases spread through mosquitoes?", TOPICS, ALIASES) == ["mosquito bites"]
    assert match_topics("diseases spread by ticks", TOPICS, ALIASES) == ["tick bites"]
    assert match_topics("Is break-bone fever serious?", TOPICS, ALIASES) == ["dengue"]
    assert match_topics("yellow fever vaccine", TOPICS, ALIASES) == ["yellow fever"]
    assert match_topics("high fever with rashes", TOPICS, ALIASES) == ["fever", "rashes"]
    assert match_topics("I have diabetes", TOPICS, ALIASES) == ["diabetes"]
    assert match_topics("common infections", TOPICS, ALIASES) == []  # groups are not question entities
