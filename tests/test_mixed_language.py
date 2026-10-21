import numpy as np

from sa_asr.audio import SAMPLE_RATE
from sa_asr.config import load_languages
from sa_asr.inference import detection_windows, smooth_languages, transcribe_audio

SR = SAMPLE_RATE


def tone(seconds: float, amp: float) -> np.ndarray:
    return (amp * np.sin(np.arange(int(seconds * SR)) / 5)).astype(np.float32)


def test_smoothing_ignores_a_weak_flip_but_follows_a_clear_switch():
    zul = {"zul": 0.8, "xho": 0.15, "afr": 0.05}
    weak_xho = {"zul": 0.42, "xho": 0.53, "afr": 0.05}  # related language edges ahead once
    afr = {"zul": 0.02, "xho": 0.02, "afr": 0.96}
    w = [2.0] * 6
    assert smooth_languages([zul, weak_xho, zul, zul, zul, zul], w) == ["zul"] * 6
    assert smooth_languages([zul, zul, zul, afr, afr, afr], w) == ["zul"] * 3 + ["afr"] * 3


def test_detection_windows_cover_the_recording_and_can_be_sampled():
    wav = np.concatenate([tone(7, 0.1), np.zeros(SR // 2, np.float32)] * 6)  # 45 s with pauses
    bounds = detection_windows(wav)
    assert bounds[0][0] == 0 and bounds[-1][1] == len(wav)
    assert all(a == prev_b for (a, _), (_, prev_b) in zip(bounds[1:], bounds))  # contiguous
    assert all(4 * SR <= b - a <= 12 * SR for a, b in bounds)
    assert len(detection_windows(wav, max_windows=2)) == 2


class FakeModel:
    """Detects 'afr' for loud audio and 'zul' for quiet audio; transcribes as '<lang>'."""

    proxy_tokens = {"zul": "sn", "afr": "af", "eng": "en"}

    def __init__(self):
        self.calls = []

    def detect_languages(self, audios, candidates):
        out = []
        for a in audios:
            loud = np.abs(a).mean() > 0.12
            p = {"afr": 0.97 if loud else 0.01, "zul": 0.01 if loud else 0.97}
            p["eng"] = 1 - p["afr"] - p["zul"]
            out.append({c.code: p[c.code] for c in candidates})
        return out

    def transcribe(self, audios, lang):
        self.calls.append((lang.code, len(audios)))
        return [lang.code for _ in audios]


def test_mixed_recording_is_transcribed_part_by_part():
    pause = np.zeros(SR // 2, np.float32)
    wav = np.concatenate([tone(9, 0.1), pause, tone(9, 0.1), pause, tone(9, 0.3), pause, tone(9, 0.3)])
    registry = load_languages()

    one = transcribe_audio(FakeModel(), wav, registry)
    assert len(one["languages"]) == 1  # one-language mode never mixes

    model = FakeModel()
    res = transcribe_audio(model, wav, registry, mixed=True)
    langs = [s["language"] for s in res["segments"]]
    assert langs[0] == "zul" and langs[-1] == "afr" and set(langs) == {"zul", "afr"}
    assert langs == sorted(langs, key=lambda c: c == "afr")  # one switch, no flip-flopping
    assert set(res["languages"]) == {"zul", "afr"} and abs(sum(res["languages"].values()) - 1) < 1e-6
    assert res["text"] == " ".join(langs)
    assert res["segments"][-1]["end"] == round(len(wav) / SR, 2)


def test_progress_is_reported_and_a_caller_can_cancel_between_batches():
    import pytest

    wav = np.concatenate([tone(9, 0.1), np.zeros(SR // 2, np.float32), tone(9, 0.1)])
    registry = load_languages()
    seen = []
    transcribe_audio(FakeModel(), wav, registry, mixed=True, on_progress=lambda what, done, total: seen.append((what, done, total)))
    assert seen and all(total >= 1 and 0 <= done <= total for _, done, total in seen)
    assert {what for what, _, _ in seen} == {"Detecting the language", "Transcribing"}

    class Stop(Exception):
        pass

    def stop_when_transcribing(what, done, total):
        if what == "Transcribing":
            raise Stop

    model = FakeModel()
    with pytest.raises(Stop):
        transcribe_audio(model, wav, registry, mixed=True, on_progress=stop_when_transcribing)
    assert model.calls == []  # cancelled before the first transcription batch ran


def test_translate_segments_uses_each_segments_language(monkeypatch):
    import sa_asr.translate as tr

    seen = []
    monkeypatch.setattr(tr, "translate", lambda texts, lang: seen.append((lang, texts)) or [f"EN({t})" for t in texts])
    segs = [{"text": "sawubona", "language": "zul"}, {"text": "hello", "language": "eng"},
            {"text": "goeie môre", "language": "afr"}, {"text": "unjani", "language": "zul"}]
    assert tr.translate_segments(segs) == ["EN(sawubona)", "hello", "EN(goeie môre)", "EN(unjani)"]
    assert sorted(seen) == [("afr", ["goeie môre"]), ("zul", ["sawubona", "unjani"])]
