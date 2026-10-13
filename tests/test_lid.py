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


def test_evaluate_lid_script_runs_end_to_end(tmp_path, monkeypatch):
    """Runs scripts/evaluate_lid.py main() with a fake model and fake cached eval sets."""
    import importlib.util
    import io
    import sys

    import numpy as np
    import soundfile as sf
    import yaml

    from sa_asr.config import PROJECT_ROOT

    spec = importlib.util.spec_from_file_location("evaluate_lid", PROJECT_ROOT / "scripts" / "evaluate_lid.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000 * 6, np.float32), 16000, format="FLAC")
    flac = buf.getvalue()

    class Rows:
        """Like a cached eval set: len(), and rows[a:b] -> dict of columns."""

        cols = {"flac": [flac] * 3, "utt_id": ["a", "b", "c"], "duration": [6.0] * 3, "text": ["x"] * 3}

        def __len__(self):
            return 3

        def __getitem__(self, key):
            return {k: v[key] for k, v in self.cols.items()}

    class FakeModel:
        proxy_tokens = {"nso": "ha", "sot": "so", "zul": "sn"}

        def detect_languages(self, clips, candidates):
            return [{"sot": 0.6, "nso": 0.3, "zul": 0.1} for _ in clips]

    monkeypatch.setattr(mod, "build_eval_set", lambda *a, **k: Rows())
    monkeypatch.setattr(mod, "resolve_model", lambda m: (m, ["nso", "sot", "zul"]))
    monkeypatch.setattr(mod, "load_transcriber", lambda ref, finetuned: FakeModel())
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(yaml.safe_dump({"cache_dir": str(tmp_path / "cache"), "eval_sets": {"nchlt": ["nso", "sot"]}}))
    out = tmp_path / "lid"
    monkeypatch.setattr(sys, "argv", ["evaluate_lid.py", "--config", str(cfg), "--output-dir", str(out), "--crops", "3"])
    mod.main()
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "| full | 6 | 50.0% | 100.0% |" in md and "| 3s | 6 |" in md
    assert "Sepedi heard as Sesotho: 3 clips (100%)" in md
    assert len((out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 12
