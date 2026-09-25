import unicodedata

from sa_asr.text import clean_transcript, normalize_for_scoring


def test_clean_strips_annotation_tags_keeps_case_and_punctuation():
    assert clean_transcript("Sawubona [noise] baba, <unk> unjani?") == "Sawubona baba, unjani?"


def test_clean_normalizes_curly_apostrophes():
    assert clean_transcript("’n Man") == "'n Man"


def test_normalize_lowercases_and_drops_punctuation():
    assert normalize_for_scoring("Sawubona, Baba! Unjani?") == "sawubona baba unjani"


def test_normalize_keeps_afrikaans_leading_apostrophe():
    assert normalize_for_scoring("Hy is ’n man.") == "hy is 'n man"


def test_normalize_drops_quote_marks():
    assert normalize_for_scoring("'Hallo' se hy") == "'hallo se hy"
    assert normalize_for_scoring("\"Hallo\" se hy") == "hallo se hy"


def test_normalize_keeps_intra_word_hyphen():
    assert normalize_for_scoring("Ngiya e-Afrika - kusasa") == "ngiya e-afrika kusasa"


def test_normalize_keeps_tshivenda_and_sotho_diacritics():
    assert normalize_for_scoring("Ṱhuṱhuwedzo ya ḓuvha") == "ṱhuṱhuwedzo ya ḓuvha"
    assert normalize_for_scoring("Ke a leboha, ntšhwarele") == "ke a leboha ntšhwarele"


def test_normalize_treats_composed_and_decomposed_forms_the_same():
    decomposed = unicodedata.normalize("NFD", "ḓuvha")
    assert normalize_for_scoring(decomposed) == normalize_for_scoring("ḓuvha")


def test_normalize_only_tags_gives_empty():
    assert normalize_for_scoring("[noise] <sil>") == ""
