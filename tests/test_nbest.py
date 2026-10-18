import numpy as np
import pytest

from sa_asr.backends import WhisperTranscriber
from sa_asr.config import load_languages


def test_nbest_returns_distinct_ranked_candidates_per_clip():
    try:
        model = WhisperTranscriber("tiny", "openai/whisper-tiny", language_mode="native", device="cpu")
    except Exception:
        pytest.skip("openai/whisper-tiny not cached locally")
    t = np.arange(16000 * 3) / 16000
    rng = np.random.default_rng(0)
    clips = [(0.3 * np.sin(2 * np.pi * f * t) + 0.05 * rng.standard_normal(len(t))).astype(np.float32) for f in (300, 900)]
    out = model.transcribe_nbest(clips, load_languages()["eng"], 4)
    assert len(out) == 2 and all(len(c) == 4 for c in out)
    for cands in out:
        scores = [s for _, s in cands]
        assert scores == sorted(scores, reverse=True)  # best first
    assert any(len({text for text, _ in cands}) > 1 for cands in out)  # the beams are not all the same text
