"""Label construction against the real Whisper tokenizer (downloads whisper-tiny's tokenizer)."""

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")

from sa_asr.config import load_languages  # noqa: E402
from sa_asr.training import LabelBuilder, WhisperCollator  # noqa: E402


@pytest.fixture(scope="module")
def processor():
    try:
        return transformers.WhisperProcessor.from_pretrained("openai/whisper-tiny")
    except OSError as e:  # offline
        pytest.skip(f"cannot download tokenizer: {e}")


@pytest.fixture(scope="module")
def labels(processor):
    return LabelBuilder(processor, list(load_languages().values()))


def test_every_proxy_token_exists_in_tokenizer(labels):
    assert len(set(labels.lang_ids.values())) == 11


def test_label_layout(processor, labels):
    ids = labels("Sawubona [noise] baba", "zul")
    tok = processor.tokenizer
    assert tok.convert_ids_to_tokens(ids[:3]) == ["<|sn|>", "<|transcribe|>", "<|notimestamps|>"]
    assert ids[-1] == tok.eos_token_id
    assert tok.decode(ids, skip_special_tokens=True) == "Sawubona baba"


def test_labels_do_not_start_with_sot(processor, labels):
    # The model prepends <|startoftranscript|> itself; a second one would shift everything.
    assert labels("molo", "xho")[0] != processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")


def test_collator_pads_with_ignore_index():
    batch = WhisperCollator()(
        [
            {"input_features": np.zeros((80, 3000), np.float32), "labels": [1, 2, 3]},
            {"input_features": np.ones((80, 3000), np.float32), "labels": [4]},
        ]
    )
    assert batch["input_features"].shape == (2, 80, 3000)
    assert batch["labels"].tolist() == [[1, 2, 3], [4, -100, -100]]


def test_find_resume_checkpoint_picks_highest_step_numerically(tmp_path):
    from sa_asr.training import find_resume_checkpoint

    for step in (500, 1000, 5000):
        (tmp_path / f"checkpoint-{step}").mkdir()
    assert find_resume_checkpoint(tmp_path, None).name == "checkpoint-5000"


def test_find_resume_checkpoint_none_without_checkpoints(tmp_path):
    from sa_asr.training import find_resume_checkpoint

    assert find_resume_checkpoint(tmp_path, None) is None


def test_training_stream_shuffles_one_file_at_a_time(monkeypatch):
    # datasets >= 5 otherwise reads up to 10 files per stream at once, which ran a 30 GB
    # machine out of memory; see build_train_stream.
    import inspect

    from datasets import IterableDataset

    import sa_asr.training as training
    from sa_asr.data import TRAIN_FEATURES

    def fake_corpus(corpus, lang, split, max_duration_s=None, seed=0):
        rows = [{"audio": {"bytes": b"", "path": None}, "utt_id": str(i), "speaker": "", "text": "x", "lang": lang.code,
                 "corpus": corpus} for i in range(3)]
        return IterableDataset.from_generator(lambda: iter(rows), features=TRAIN_FEATURES)

    import functools

    calls = []
    original = IterableDataset.shuffle

    @functools.wraps(original)  # keeps the real signature visible to inspect.signature
    def spy(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(training, "stream_corpus", fake_corpus)
    monkeypatch.setattr(IterableDataset, "shuffle", spy)
    zul = load_languages()["zul"]
    training.build_train_stream([("nchlt", zul, 1.0), ("lwazi", zul, 1.0)], lambda ex: ex, seed=0, shuffle_buffer=4)
    assert len(calls) == 2
    if "max_buffer_input_shards" in inspect.signature(original).parameters:
        assert all(c.get("max_buffer_input_shards") == 1 for c in calls)


def test_time_limit_callback_saves_and_stops_after_deadline():
    import time

    from transformers import TrainerControl, TrainerState

    from sa_asr.training import TimeLimitCallback

    state, control = TrainerState(), TrainerControl()
    state.global_step = 7
    assert not TimeLimitCallback(time.time() + 3600).on_step_end(None, state, control).should_training_stop
    cb = TimeLimitCallback(time.time() - 1)
    out = cb.on_step_end(None, state, TrainerControl())
    assert out.should_save and out.should_training_stop and cb.hit


def test_fold_diacritics_for_diagnosis():
    from sa_asr.training import fold_diacritics

    assert fold_diacritics("vha ḓo ṱoḓa ṋamusi") == "vha do toda namusi"
    assert fold_diacritics("ke šomiša") == "ke somisa"


def test_poor_set_explained_and_predictions_saved(tmp_path, caplog):
    import json
    import logging
    from types import SimpleNamespace

    from sa_asr.training import WhisperWERTrainer

    with caplog.at_level(logging.INFO, logger="sa_asr.training"):
        WhisperWERTrainer._explain_poor_set("nchlt", "ven", 0.8, ["vha do ya"], ["vha ḓo ya"])
    assert "0.0% ignoring diacritics" in caplog.text and "vha do ya  =>  vha ḓo ya" in caplog.text

    fake = SimpleNamespace(args=SimpleNamespace(output_dir=str(tmp_path), push_to_hub=False, hub_model_id=None),
                           state=SimpleNamespace(global_step=500))
    WhisperWERTrainer._save_predictions(fake, [{"corpus": "nchlt", "lang": "ven", "ref": "ḓo", "hyp": "do"}])
    saved = (tmp_path / "eval_predictions" / "step-00500.jsonl").read_text(encoding="utf-8")
    assert json.loads(saved) == {"corpus": "nchlt", "lang": "ven", "ref": "ḓo", "hyp": "do"}
