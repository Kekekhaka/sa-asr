import numpy as np

from sa_asr.audio import SAMPLE_RATE
from sa_asr.config import load_languages
from sa_asr.inference import transcribe_audio
from sa_asr.textlid import TextLID, combine

SR = SAMPLE_RATE
ZUL = ["ngiyabonga kakhulu baba wami", "sawubona nkosikazi ngiyaphila", "ubaba uyahamba ekhaya manje"]
XHO = ["enkosi kakhulu tata wam", "molo nkosikazi ndiphilile", "utata uyahamba ekhaya ngoku"]
ENG = ["thank you very much father", "hello madam i am well", "father is going home now"]


def model() -> TextLID:
    return TextLID.train({"zul": ZUL * 4, "xho": XHO * 4, "eng": ENG * 4})


def test_spelling_identifies_the_language_and_roundtrips(tmp_path):
    m = model()
    xho, zul = m.posterior("enkosi tata ndiyahamba"), m.posterior("ngiyabonga baba ngiyahamba")
    assert max(xho, key=xho.get) == "xho" and max(zul, key=zul.get) == "zul"
    m.save(tmp_path / "t.json.gz")
    again = TextLID.load(tmp_path / "t.json.gz")
    assert again.posterior("enkosi tata ndiyahamba") == m.posterior("enkosi tata ndiyahamba")


def test_text_flips_a_close_audio_call_but_not_a_short_text_one():
    audio = {"zul": 0.55, "xho": 0.45}
    text = {"zul": 0.02, "xho": 0.98}
    assert next(iter(combine(audio, text, 1.0, 60))) == "xho"
    assert next(iter(combine(audio, text, 1.0, 0))) == "zul"  # no text: the audio decides
    sure_audio = {"zul": 0.99, "xho": 0.01}
    assert next(iter(combine(sure_audio, {"zul": 0.3, "xho": 0.7}, 1.0, 60))) == "zul"


def test_text_weight_per_family_of_the_audio_top_language():
    audio = {"nso": 0.8, "tsn": 0.2}  # the audio leans Sepedi, the spelling says Setswana
    text = {"nso": 0.05, "tsn": 0.95}
    assert next(iter(combine(audio, text, 0.25, 60))) == "nso"
    assert next(iter(combine(audio, text, {"Sotho-Tswana": 3.0, "default": 0.25}, 60))) == "tsn"
    assert next(iter(combine(audio, text, {"Nguni": 3.0, "default": 0.25}, 60))) == "nso"  # other family: default
    nguni = {"zul": 0.8, "xho": 0.2}
    assert next(iter(combine(nguni, {"zul": 0.05, "xho": 0.95}, {"Sotho-Tswana": 3.0}, 60))) == "xho"  # default 1.0


class FakeModel:
    """The audio detector is unsure (zul 0.55 vs xho 0.45); the speech is really isiXhosa, which
    is what it writes whatever language it is told (as Whisper mostly does)."""

    proxy_tokens = {"zul": "sn", "xho": "sw", "eng": "en"}

    def __init__(self):
        self.calls = []

    def detect_languages(self, audios, candidates):
        return [{c.code: {"zul": 0.55, "xho": 0.45, "eng": 0.0}[c.code] for c in candidates} for _ in audios]

    def transcribe(self, audios, lang):
        self.calls.append(lang.code)
        return ["enkosi kakhulu tata wam molo nkosikazi ndiphilile" for _ in audios]


def test_text_check_corrects_the_label_and_transcribes_again_in_that_language():
    wav = (0.1 * np.sin(np.arange(12 * SR) / 5)).astype(np.float32)
    registry = load_languages()
    plain = transcribe_audio(FakeModel(), wav, registry, mixed=True)
    assert plain["language"] == "zul"  # audio alone: the weak favourite

    fake = FakeModel()
    res = transcribe_audio(fake, wav, registry, mixed=True, text_lid=model())
    assert res["language"] == "xho" and list(res["languages"]) == ["xho"]
    assert all(s["language"] == "xho" for s in res["segments"])
    assert fake.calls[0] == "zul" and fake.calls[-1] == "xho"  # first pass, then redone in the new language
    assert res["text"].startswith("enkosi")
