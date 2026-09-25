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
