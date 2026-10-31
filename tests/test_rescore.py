import numpy as np

from sa_asr.config import load_languages
from sa_asr.rescore import RescoringTranscriber, pick, sentence_case

LANGS = load_languages()


class FakeWhisper:
    device = "cpu"
    proxy_tokens = ["zul", "ven"]

    def __init__(self):
        self.calls = []

    def transcribe(self, audios, lang):
        self.calls.append(("plain", lang.code if lang else None))
        return ["plain"] * len(audios)

    def transcribe_nbest(self, audios, lang, n):
        self.calls.append(("nbest", lang.code))
        # Whisper prefers "a" slightly; the language model will prefer "b"
        return [[("a", -0.10), ("b", -0.12)] for _ in audios]


class FakeLM:
    def scores(self, texts, device):
        return [-10.0 if t == "a" else -5.0 for t in texts]


def test_pick_weighs_asr_and_lm():
    cands = [("a", -0.10), ("b", -0.12)]
    assert pick(cands, [-10.0, -5.0], 0.0) == "a"  # no LM weight: Whisper's best
    assert pick(cands, [-10.0, -5.0], 0.02) == "b"  # -0.12 - 0.1 > -0.10 - 0.2


def test_rescoring_only_in_weighted_languages_and_delegates_the_rest():
    w = FakeWhisper()
    r = RescoringTranscriber(w, FakeLM(), lambdas={"zul": 0.02})
    clips = [np.zeros(16000, dtype=np.float32)] * 2
    assert r.transcribe(clips, LANGS["zul"]) == ["b", "b"]
    assert r.transcribe(clips, LANGS["ven"]) == ["plain", "plain"]  # no weight: transcribed as usual
    assert r.transcribe(clips, None) == ["plain", "plain"]
    assert w.calls == [("nbest", "zul"), ("plain", "ven"), ("plain", None)]
    assert r.proxy_tokens == ["zul", "ven"]  # everything else comes from the wrapped model


def test_sentence_case():
    assert sentence_case("sawubona baba") == "Sawubona baba."
    assert sentence_case("Unjani?") == "Unjani?"
    assert sentence_case("  ") == ""
