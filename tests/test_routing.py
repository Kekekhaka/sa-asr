import numpy as np

from sa_asr.audio import SAMPLE_RATE
from sa_asr.config import load_languages
from sa_asr.inference import transcribe_audio
from sa_asr.routing import MIN_HIGH_BAND_DB, eligible, high_band_db, reroute

SR = SAMPLE_RATE
RNG = np.random.RandomState(0)


def wideband(seconds: float) -> np.ndarray:
    return (0.1 * RNG.randn(int(seconds * SR))).astype(np.float32)  # white noise: plenty above 4 kHz


def telephone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.1 * np.sin(2 * np.pi * 600 * t) + 0.05 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)  # nothing above 4 kHz


class Second:
    def __init__(self):
        self.calls = []

    def transcribe(self, audios, lang):
        self.calls.append((len(audios), lang))
        return ["second"] * len(audios)


def test_bandwidth_separates_wideband_from_telephone_audio():
    assert high_band_db(wideband(3)) >= MIN_HIGH_BAND_DB > high_band_db(telephone(3))
    assert high_band_db(np.zeros(100, np.float32)) < MIN_HIGH_BAND_DB  # too short to measure


def test_only_long_wideband_audio_in_the_six_languages_is_eligible():
    assert eligible("zul", wideband(6))
    assert not eligible("zul", wideband(3))        # too short
    assert not eligible("zul", telephone(8))       # narrow-band
    assert not eligible("nso", wideband(8))        # not a dsfsi language
    assert not eligible(None, wideband(8))


def test_reroute_changes_only_eligible_segments():
    wav = np.concatenate([wideband(6), wideband(6), telephone(6)])
    segs = [
        {"start": 0.0, "end": 6.0, "text": "a", "language": "zul"},
        {"start": 6.0, "end": 12.0, "text": "b", "language": "eng"},
        {"start": 12.0, "end": 18.0, "text": "c", "language": "zul"},
    ]
    second = Second()
    assert reroute(segs, wav, second) == 1
    assert [s["text"] for s in segs] == ["second", "b", "c"] and segs[0]["engine"] == "dsfsi-anv"
    assert second.calls == [(1, None)]  # the second model picks its own language token


def test_an_empty_answer_keeps_the_original_text():
    class Empty(Second):
        def transcribe(self, audios, lang):
            return [""] * len(audios)

    segs = [{"start": 0.0, "end": 6.0, "text": "kept", "language": "xho"}]
    assert reroute(segs, wideband(6), Empty()) == 0 and segs[0]["text"] == "kept" and "engine" not in segs[0]


def test_transcribe_audio_uses_the_second_model_for_clear_speech_only():
    class Main:
        proxy_tokens = {"zul": "sn", "eng": "en"}

        def detect_languages(self, audios, candidates):
            return [{c.code: 0.98 if c.code == "zul" else 0.01 for c in candidates} for _ in audios]

        def transcribe(self, audios, lang):
            return ["main"] * len(audios)

    registry = load_languages()
    clear = transcribe_audio(Main(), wideband(8), registry, "zul", second_model=Second())
    assert clear["rerouted"] >= 1 and "second" in clear["text"] and any(s.get("engine") for s in clear["segments"])
    phone = transcribe_audio(Main(), telephone(8), registry, "zul", second_model=Second())
    assert phone["rerouted"] == 0 and phone["text"] == "main"
    none = transcribe_audio(Main(), wideband(8), registry, "zul")
    assert none["rerouted"] == 0 and none["text"] == "main"
