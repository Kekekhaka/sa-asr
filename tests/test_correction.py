from sa_asr.correction import build_messages, edit_distance, guard, parse_response

CANDS = [
    "Dimela di dira oksitšene yeo batho ba e hemago",
    "Dimela di dira oksitsene yeo batho ba e hemago",
    "Dimela di dira oxygen yeo batho ba hemago",
]


def test_prompt_names_language_lists_candidates_and_adds_hint():
    system, user = build_messages("Sepedi", "nso", CANDS)
    assert "Sepedi" in system["content"] and "š" in system["content"]
    assert "1. Dimela" in user["content"] and "3. Dimela" in user["content"]
    assert "š" not in build_messages("Sepedi", "nso", CANDS, hints=False)[0]["content"]


def test_parse_strips_labels_quotes_fences_and_numbering():
    assert parse_response('Corrected transcript: "Ke a leboga."') == "Ke a leboga."
    assert parse_response("```\n1. Ke a leboga.\n```") == "Ke a leboga."
    assert parse_response("\n\n  Ke a leboga.\nExplanation: fixed š") == "Ke a leboga."
    assert parse_response("") == ""


def test_edit_distance_stops_early():
    assert edit_distance("tša", "tsa", 2) == 1
    assert edit_distance("ajabulisayo", "ajabulishayo", 2) == 1
    assert edit_distance("abc", "xyzxyz", 2) == 3


def test_guard_accepts_candidate_words_and_small_spelling_fixes():
    assert guard("Dimela di dira oxygen yeo batho ba e hemago", CANDS) == (True, "ok")
    assert guard("Dimela di dira oksitšene yeo batho ba e hemelago", CANDS) == (True, "ok")


def test_guard_rejects_new_words_length_changes_and_empty():
    # A translation keeps the length but brings in words no candidate has.
    assert guard("Plants make the oxygen that people breathe in", CANDS) == (False, "new_word")
    assert guard("Dimela di dira", CANDS) == (False, "length")
    assert guard("Dimela di dira motsoko yeo batho ba e hemago", CANDS) == (False, "new_word")
    assert guard("", CANDS) == (False, "empty")
