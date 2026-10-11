import numpy as np
import pyarrow.parquet as pq

from sa_asr.feedback import export_parquet, load_corrections, save_correction, summary
from sa_asr.translate import ROUTES, can_translate, translate


def test_corrections_saved_deduplicated_and_exported(tmp_path):
    wav = (np.sin(np.arange(16000) / 10) * 0.1).astype(np.float32)
    kw = dict(predicted_lang="zul", predicted_text="sawubona baba", model="m", root=tmp_path)
    save_correction(wav, "xho", "molo tata", **kw)
    save_correction(wav, "xho", "molo tata wam", **kw)  # same audio corrected again: replaces
    other = save_correction(wav * 0.5, "zul", "sawubona baba", **kw)

    recs = load_corrections(tmp_path)
    assert len(recs) == 2 and {r["text"] for r in recs} == {"molo tata wam", "sawubona baba"}
    assert not other["language_corrected"] and not other["text_corrected"]
    stats = summary(tmp_path)
    assert stats["count"] == 2 and stats["language_fixes"] == 1 and stats["by_language"] == {"xho": 1, "zul": 1}

    n = export_parquet(tmp_path / "out.parquet", root=tmp_path)
    table = pq.read_table(tmp_path / "out.parquet").to_pylist()
    assert n == 2 and {"audio", "utt_id", "speaker", "text", "lang"} <= set(table[0])
    assert table[0]["audio"]["bytes"][:4] == b"fLaC"


def test_translation_routes_cover_every_non_english_language():
    from sa_asr.config import load_languages

    langs = [c for c in load_languages() if c != "eng"]
    assert all(can_translate(c) for c in langs) and not can_translate("eng")
    assert ROUTES["nbl"].approximate and not ROUTES["zul"].approximate


def test_translate_sets_languages_and_keeps_empty_segments(monkeypatch):
    import torch

    import sa_asr.translate as tr

    class Tok:
        src_lang = None

        def convert_tokens_to_ids(self, token):
            return {"eng_Latn": 7}[token]

        def __call__(self, texts, **kw):
            Tok.seen = (self.src_lang, list(texts))
            return {"input_ids": torch.ones(len(texts), 2, dtype=torch.long)}

        def batch_decode(self, ids, **kw):
            return [f"en{i}" for i in range(len(ids))]

    class Model:
        def generate(self, input_ids, forced_bos_token_id=None, **kw):
            Model.bos = forced_bos_token_id
            return input_ids

    monkeypatch.setattr(tr, "_load", lambda model_id: (Tok(), Model()))
    assert translate(["ngiyabonga", "", "  "], "zul") == ["en0", "", ""]
    assert Tok.seen == ("zul_Latn", ["ngiyabonga"]) and Model.bos == 7
    assert translate([""], "zul") == [""]
