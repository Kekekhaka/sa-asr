import math

import numpy as np
import pytest
import soundfile as sf

from sa_asr.audio import decode_audio
from sa_asr.config import Language, load_languages, select_languages, validate_languages
from sa_asr.data import CORPORA
from sa_asr.metrics import score

WHISPER_LANGUAGE_CODES = {
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id",
    "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg",
    "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br",
    "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so",
    "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt",
    "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
}


def test_registry_has_all_eleven_spoken_official_languages():
    langs = load_languages()
    assert set(langs) == {"eng", "afr", "zul", "xho", "nso", "sot", "tsn", "tso", "ven", "ssw", "nbl"}


def test_proxy_tokens_are_real_whisper_tokens():
    for lang in load_languages().values():
        assert lang.whisper_token in WHISPER_LANGUAGE_CODES, lang.code


def test_registry_only_references_known_corpora():
    for lang in load_languages().values():
        assert set(lang.corpora) <= set(CORPORA), lang.code


def test_duplicate_proxy_tokens_rejected():
    a = Language("aaa", "A", "sn", False, None)
    b = Language("bbb", "B", "sn", False, None)
    with pytest.raises(ValueError, match="sn"):
        validate_languages({"aaa": a, "bbb": b})


def test_language_is_hashable():
    lang = load_languages()["zul"]
    assert {("nchlt", lang): 1}[("nchlt", lang)] == 1


def test_select_languages():
    langs = load_languages()
    assert len(select_languages(langs, None)) == 11
    assert len(select_languages(langs, ["all"])) == 11
    assert [l.code for l in select_languages(langs, ["zul", "xho"])] == ["zul", "xho"]
    with pytest.raises(ValueError):
        select_languages(langs, ["xx"])


def test_score_is_corpus_level_and_normalized():
    s = score(["Sawubona baba", "Unjani?"], ["sawubona", "unjani"])
    assert s["n_utts"] == 2 and s["n_ref_words"] == 3
    assert s["wer"] == pytest.approx(1 / 3)


def test_score_skips_empty_references_and_handles_empty_hypothesis():
    s = score(["[noise]", "molo"], ["anything", ""])
    assert s["n_utts"] == 1
    assert s["wer"] == pytest.approx(1.0)


def test_score_empty():
    assert math.isnan(score([], [])["wer"])


def test_decode_audio_resamples_and_downmixes(tmp_path):
    sr = 48_000
    t = np.linspace(0, 1, sr, endpoint=False)
    stereo = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32) * 0.5
    path = tmp_path / "a.wav"
    sf.write(path, stereo, sr)
    wav = decode_audio({"bytes": path.read_bytes(), "path": None})
    assert wav.dtype == np.float32 and wav.ndim == 1
    assert abs(len(wav) - 16_000) <= 1


def test_telephone_channel_removes_energy_above_4khz():
    from sa_asr.audio import telephone_channel

    sr = 16_000
    t = np.arange(sr) / sr
    wav = (0.4 * np.sin(2 * np.pi * 1000 * t) + 0.4 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    out = telephone_channel(wav, np.random.default_rng(0))
    spec = np.abs(np.fft.rfft(out))
    assert out.shape == wav.shape
    assert spec[6000] < 0.01 * spec[1000]


def test_augment_is_noop_at_zero_probability():
    from sa_asr.audio import augment

    wav = np.random.default_rng(0).normal(size=1600).astype(np.float32)
    assert np.array_equal(augment(wav, np.random.default_rng(1)), wav)


def test_validation_holdout_is_stable_and_roughly_sized():
    from sa_asr.data import in_validation_holdout

    speakers = [str(i) for i in range(4000)]
    held = [s for s in speakers if in_validation_holdout(s, 0.05)]
    assert 120 < len(held) < 280
    assert held == [s for s in speakers if in_validation_holdout(s, 0.05)]


def test_nchlt_eng_utterance_ids_include_speaker():
    assert CORPORA["nchlt_eng"].id_column == ("speaker_id", "utterance_id")


def test_select_languages_accepts_plain_all_string():
    # train_lora.yaml says `languages: all` (a string, not a list)
    assert len(select_languages(load_languages(), "all")) == 11


def test_eval_set_path_changes_with_every_sampling_setting():
    from sa_asr.data import eval_set_path

    base = dict(cache_dir="c", corpus="nchlt", lang="zul", split="test", max_samples=200, seed=42)
    names = {
        eval_set_path(**base).name,
        eval_set_path(**{**base, "max_samples": 5}).name,
        eval_set_path(**{**base, "seed": 1}).name,
        eval_set_path(**base, max_duration_s=20).name,
        eval_set_path(**base, shuffle_buffer=100).name,
    }
    assert len(names) == 5


def test_split_long_audio_cuts_at_silence_and_keeps_everything():
    from sa_asr.audio import split_long_audio

    sr = 16_000
    rng = np.random.default_rng(0)
    speech = lambda s: rng.normal(0, 0.3, int(s * sr)).astype(np.float32)  # noqa: E731
    silence = lambda s: np.zeros(int(s * sr), np.float32)  # noqa: E731
    wav = np.concatenate([speech(20), silence(1), speech(20), silence(1), speech(10)])  # 52 s
    chunks = split_long_audio(wav, sr)
    assert sum(len(c) for _, c in chunks) == len(wav)
    assert all(len(c) <= 28 * sr for _, c in chunks)
    # first cut lands inside the first silence (20-21 s)
    assert 20 <= chunks[1][0] <= 21


def test_split_long_audio_short_input_is_one_chunk():
    from sa_asr.audio import split_long_audio

    wav = np.zeros(16_000 * 5, np.float32)
    assert [s for s, _ in split_long_audio(wav)] == [0.0]


def test_localize_model_leaves_local_paths_and_plain_ids_alone(tmp_path):
    from sa_asr.backends import localize_model

    assert localize_model(str(tmp_path)) == tmp_path
    assert str(localize_model("not-a-repo-id")) == "not-a-repo-id"


def test_hub_sync_paths_mirror_project_layout():
    from sa_asr.config import PROJECT_ROOT
    from sa_asr.hub import HubSync

    assert HubSync.rel("results/baseline") == "results/baseline"
    assert HubSync.rel(PROJECT_ROOT / "data" / "eval_cache") == "data/eval_cache"


def test_split_by_padded_length_bounds_memory_and_keeps_order():
    from sa_asr.backends import split_by_padded_length

    sr = 16_000
    lengths = [30 * sr] * 16  # the batch that ran a T4 out of memory
    groups = split_by_padded_length(lengths, 120 * sr)
    assert [i for g in groups for i in g] == list(range(16))
    assert all(len(g) * max(lengths[i] for i in g) <= 120 * sr for g in groups)
    # short clips stay together; an oversize clip gets its own group
    assert split_by_padded_length([sr] * 16, 120 * sr) == [list(range(16))]
    assert split_by_padded_length([200 * sr, sr], 120 * sr) == [[0], [1]]


def test_header_duration_reads_length_without_decoding(tmp_path):
    import io

    from sa_asr.data import header_duration

    for fmt in ("WAV", "FLAC"):
        buf = io.BytesIO()
        sf.write(buf, np.zeros(48_000 * 3, np.float32), 48_000, format=fmt)
        assert header_duration({"bytes": buf.getvalue()}) == pytest.approx(3.0)
    assert header_duration({"bytes": b"not audio"}) is None


def test_t4_trains_in_fp16_not_emulated_bf16(monkeypatch):
    # torch says a T4 "supports" bf16 (by emulation); training in it ran at ~130 s/step.
    import torch

    from sa_asr.backends import native_bf16, pick_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))  # T4
    assert not native_bf16()
    assert pick_dtype(torch.device("cuda")) == torch.float16
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 0))  # A100
    assert native_bf16()


def test_files_containing_picks_files_by_language_statistics(tmp_path):
    # Lwazi's files are sorted by language; each language stream should only read its own.
    import pyarrow as pa
    import pyarrow.parquet as pq
    from fsspec.implementations.dirfs import DirFileSystem
    from fsspec.implementations.local import LocalFileSystem

    from sa_asr.data import files_containing

    root = tmp_path / "datasets" / "org" / "repo" / "data"
    root.mkdir(parents=True)
    for i, langs in enumerate([["afr", "eng"], ["eng", "nbl"], ["zul", "zul"]]):
        pq.write_table(pa.table({"language": langs, "text": ["x", "y"]}), root / f"train-0000{i}-of-00003.parquet",
                       row_group_size=1)
    fs = DirFileSystem(path=str(tmp_path), fs=LocalFileSystem())
    pick = lambda value: [u.rsplit("/", 1)[-1][:11] for u in files_containing("org/repo", "train", "language", value, fs=fs)]
    assert pick("eng") == ["train-00000", "train-00001"]
    assert pick("zul") == ["train-00002"]
    assert files_containing("org/repo", "train", "language", "xho", fs=fs) is None  # unknown: stream everything
