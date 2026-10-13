from sa_asr.lid import FAMILIES, accuracy, family_of, report_markdown, top_confusions


def recs(cond, pairs, corpus="nchlt"):
    return [{"cond": cond, "corpus": corpus, "lang": t, "pred": p} for t, p in pairs]


def test_families_cover_all_eleven_languages_once():
    codes = [c for fam in FAMILIES.values() for c in fam]
    assert sorted(codes) == sorted(["eng", "afr", "zul", "xho", "ssw", "nbl", "nso", "sot", "tsn", "tso", "ven"])
    assert family_of("nso") == family_of("tsn") != family_of("zul")


def test_accuracy_counts_family_matches():
    r = recs("full", [("nso", "nso"), ("nso", "sot"), ("zul", "xho"), ("zul", "afr")])
    a = accuracy(r)
    assert a["n"] == 4 and a["accuracy"] == 0.25 and a["family_accuracy"] == 0.75


def test_report_lists_conditions_matrix_and_top_mistakes():
    r = recs("full", [("nso", "nso")] * 6 + [("nso", "sot")] * 4 + [("zul", "zul")] * 9 + [("zul", "xho")])
    r += recs("5s", [("nso", "tsn")] * 10 + [("zul", "zul")] * 10)
    assert top_confusions(r)[0][:3] == ("nso", "tsn", 10)
    md = report_markdown(r, {"nso": "Sepedi", "sot": "Sesotho", "zul": "isiZulu"}, "LID test")
    assert "| full | 20 | 75.0% | 100.0% |" in md and "| 5s | 20 | 50.0% | 100.0% |" in md
    assert "Sepedi heard as Sesotho: 4 clips (40%)" in md
    # full-clip matrix, columns grouped by family (zul, xho, nso, sot): Sepedi row
    assert "| Sepedi |  |  | **60%** | 40% |" in md
