from sa_asr.config import load_languages
from sa_asr.data import CORPORA


def test_unscripted_swivuriso_is_a_filtered_view_of_swivuriso():
    full, view = CORPORA["swivuriso"], CORPORA["swivuriso_unscripted"]
    assert view.repo == full.repo and view.splits == full.splits and view.text_column == full.text_column
    assert view.row_filter == ("type", "unscripted") and full.row_filter is None


def test_every_swivuriso_language_has_an_unscripted_mapping_with_the_same_config():
    for lang in load_languages().values():
        assert ("swivuriso" in lang.corpora) == ("swivuriso_unscripted" in lang.corpora), lang.code
        if "swivuriso" in lang.corpora:
            assert lang.corpora["swivuriso"] == lang.corpora["swivuriso_unscripted"]
