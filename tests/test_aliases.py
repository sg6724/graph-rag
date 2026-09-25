from medgraph.aliases import normalize_name, normalize_text


def test_enzyme_spellings_collapse():
    for s in ["CYP3A4", "CYP 3A4", "cyp-3a4", "Cytochrome P450 3A4"]:
        assert normalize_name(s) == "cyp3a4"
    assert normalize_name("P-glycoprotein") == "p-gp"


def test_brand_to_generic_and_whitespace():
    assert normalize_name("  Coumadin ") == "warfarin"
    assert normalize_name("Heart   Failure") == "heart failure"


def test_normalize_text_rewrites_enzymes_inside_sentences():
    assert normalize_text("Is CYP 3A4 involved?") == "is cyp3a4 involved?"
