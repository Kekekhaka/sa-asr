import numpy as np
import pyarrow.parquet as pq

from sa_asr.feedback import export_parquet, load_corrections, report, save_correction, split_of, summary


def clip(seed: int) -> np.ndarray:
    return (0.1 * np.random.RandomState(seed).randn(16000)).astype(np.float32)


def fill(tmp_path, n=40):
    for i in range(n):
        save_correction(clip(i), "zul" if i % 2 else "sot", f"ukuthi {i}", predicted_lang="zul" if i % 4 else "xho",
                        predicted_text=f"ukuthi {i}" if i % 3 else "ukuthi x", model="m", root=tmp_path)


def test_split_depends_only_on_the_audio_and_is_about_thirty_percent_test(tmp_path):
    # the share is checked on many digests (60 saved clips is too few to pin down a 30% share)
    import hashlib

    digests = [hashlib.sha1(str(i).encode()).hexdigest()[:10] for i in range(3000)]
    share = sum(split_of(f"20260101-000000-{d}") == "test" for d in digests) / len(digests)
    assert 0.27 < share < 0.33
    fill(tmp_path, 60)
    recs = load_corrections(tmp_path)
    again = save_correction(clip(3), "zul", "fixed", predicted_lang="zul", predicted_text="x", model="m", root=tmp_path)
    first = next(r for r in recs if r["utt_id"].split("-", 2)[-1] == again["utt_id"].split("-", 2)[-1])
    assert split_of(first["utt_id"]) == split_of(again["utt_id"]) == again["split"]  # same audio, same split


def test_export_writes_disjoint_train_and_test_files(tmp_path):
    fill(tmp_path, 60)
    n_train = export_parquet(tmp_path / "train.parquet", tmp_path, split="train")
    n_test = export_parquet(tmp_path / "test.parquet", tmp_path, split="test")
    assert n_train + n_test == summary(tmp_path)["count"] and n_train > 0 and n_test > 0
    train_ids = set(pq.read_table(tmp_path / "train.parquet").column("utt_id").to_pylist())
    test_ids = set(pq.read_table(tmp_path / "test.parquet").column("utt_id").to_pylist())
    assert not train_ids & test_ids


def test_languages_present_always_includes_the_corrected_language(tmp_path):
    rec = save_correction(clip(1), "zul", "t", predicted_lang="xho", predicted_text="t", model="m", root=tmp_path,
                          languages_present=["eng"])
    assert rec["languages_present"] == ["eng", "zul"] and rec["language_corrected"]
    plain = save_correction(clip(2), "sot", "t", predicted_lang="sot", predicted_text="t", model="m", root=tmp_path)
    assert plain["languages_present"] == ["sot"]


def test_word_languages_are_kept_and_added_to_the_languages_present(tmp_path):
    rec = save_correction(clip(1), "nso", "ke a leboga thank you", predicted_lang="sot", predicted_text="x",
                          model="m", root=tmp_path, word_langs=["nso", "nso", "nso", "eng", "eng"])
    assert rec["word_langs"] == ["nso", "nso", "nso", "eng", "eng"] and rec["languages_present"] == ["eng", "nso"]
    plain = save_correction(clip(2), "nso", "ke a leboga", predicted_lang="nso", predicted_text="x", model="m",
                            root=tmp_path)
    assert plain["word_langs"] == ["nso"] * 3
    import pytest

    with pytest.raises(ValueError):
        save_correction(clip(3), "nso", "two words", predicted_lang="nso", predicted_text="x", model="m",
                        root=tmp_path, word_langs=["nso"])
    export_parquet(tmp_path / "all.parquet", tmp_path)
    rows = pq.read_table(tmp_path / "all.parquet").to_pylist()
    assert sorted(len(r["word_langs"]) for r in rows) == [3, 5]


def test_segments_are_kept_with_their_languages(tmp_path):
    parts = [{"start": 0.0, "end": 5.0, "lang": "nso", "text": "Thobela ka mošate"},
             {"start": 5.0, "end": 11.0, "lang": "xho", "text": "Sivukile"}]
    rec = save_correction(clip(1), "xho", "Thobela ka mošate Sivukile", predicted_lang="sot", predicted_text="x",
                          model="m", root=tmp_path, segments=parts)
    assert rec["segments"] == parts and rec["languages_present"] == ["nso", "xho"]
    plain = save_correction(clip(2), "nso", "ke a leboga", predicted_lang="nso", predicted_text="x", model="m",
                            root=tmp_path)
    assert plain["segments"] == [{"start": 0.0, "end": 1.0, "lang": "nso", "text": "ke a leboga"}]
    export_parquet(tmp_path / "all.parquet", tmp_path)
    rows = pq.read_table(tmp_path / "all.parquet").to_pylist()
    assert sorted(len(r["segments"]) for r in rows) == [1, 2]


def test_report_measures_language_and_word_accuracy(tmp_path):
    save_correction(clip(1), "zul", "ngiyabonga kakhulu", predicted_lang="zul", predicted_text="ngiyabonga kakhulu",
                    model="m", root=tmp_path)
    save_correction(clip(2), "zul", "sawubona baba", predicted_lang="xho", predicted_text="sawubona mama", model="m",
                    root=tmp_path)
    out = report(tmp_path)
    assert out["all"]["count"] == 2 and out["all"]["language_right"] == 50.0
    assert out["all"]["wer"] == 25.0  # 1 wrong word out of 4
    assert out["language_mistakes"] == {"zul heard as xho": 1}
    assert out["by_language"]["zul"]["count"] == 2
