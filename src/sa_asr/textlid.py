"""Language identification from spelling: a second opinion on the audio detector.

Whisper's language token barely changes what it writes (the decoder follows the audio), so
transcribing a clip under each candidate language does not tell close relatives apart. The
*spelling* of what it wrote does: isiZulu, isiXhosa, siSwati and isiNdebele, or Sepedi, Sesotho
and Setswana, each have their own letter patterns. A character n-gram model trained on
transcripts of each language reads that off the transcript, and its verdict is combined with
the audio detector's probabilities (see inference.transcribe_audio).

Measured on held-out sentences (3100 test transcripts, text model trained on only ~1500
sentences): audio alone 76.3% exact language, text alone 81.4%, combined 84.6%.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path

from .config import PROJECT_ROOT

NGRAMS = (1, 2, 3, 4)
SMOOTHING = 0.1
DEFAULT_PATH = PROJECT_ROOT / "models" / "text_lid.json.gz"
# Below this many characters the text says little; its weight shrinks proportionally.
FULL_WEIGHT_CHARS = 30


def normalize(text: str) -> str:
    return " " + re.sub(r"\s+", " ", re.sub(r"[^\w\s']", "", text.lower())).strip() + " "


def grams(text: str) -> list[str]:
    s = normalize(text)
    return [s[i : i + n] for n in NGRAMS for i in range(len(s) - n + 1)]


class TextLID:
    def __init__(self, counts: dict[str, Counter]):
        self.counts = counts
        self.totals = {c: sum(cnt.values()) for c, cnt in counts.items()}
        self.vocab = len(set().union(*(cnt.keys() for cnt in counts.values()))) if counts else 0

    @classmethod
    def train(cls, texts_by_lang: dict[str, list[str]]) -> "TextLID":
        counts: dict[str, Counter] = {}
        for code, texts in texts_by_lang.items():
            c: Counter = Counter()
            for t in texts:
                c.update(grams(t))
            counts[code] = c
        return cls(counts)

    @property
    def languages(self) -> list[str]:
        return sorted(self.counts)

    def scores(self, text: str) -> dict[str, float]:
        """Mean log-probability per n-gram under each language (length-neutral, so texts of
        different lengths are comparable)."""
        g = grams(text)
        if not g:
            return {c: 0.0 for c in self.counts}
        return {
            c: sum(math.log((cnt[x] + SMOOTHING) / (self.totals[c] + SMOOTHING * self.vocab)) for x in g) / len(g)
            for c, cnt in self.counts.items()
        }

    def posterior(self, text: str, temperature: float = 0.05) -> dict[str, float]:
        """Probability of each language from the spelling alone (a softmax over the mean scores)."""
        s = self.scores(text)
        top = max(s.values())
        e = {c: math.exp((v - top) / temperature) for c, v in s.items()}
        z = sum(e.values())
        return {c: v / z for c, v in e.items()}

    def save(self, path: str | Path = DEFAULT_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({c: dict(cnt) for c, cnt in self.counts.items()}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "TextLID":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return cls({c: Counter(d) for c, d in json.load(f).items()})


def family_weight(audio: dict[str, float], text_weight: float | dict[str, float]) -> float:
    """The text weight for this clip: a number, or {family: weight} looked up by the family of the audio's top
    language (e.g. the spelling counts more when the audio hesitates among Sepedi, Sesotho and Setswana);
    families not in the dict use its "default" entry (1.0 if absent)."""
    if not isinstance(text_weight, dict):
        return text_weight
    from .lid import family_of

    return text_weight.get(family_of(max(audio, key=audio.get)), text_weight.get("default", 1.0))


def combine(audio: dict[str, float], text: dict[str, float], text_weight: float | dict[str, float],
            n_chars: int) -> dict[str, float]:
    """Audio probabilities times text probabilities^weight, renormalised. The weight shrinks for
    very short texts, and can depend on the audio's top language family (see family_weight).
    Languages missing from `text` keep only their audio evidence."""
    w = family_weight(audio, text_weight) * min(1.0, n_chars / FULL_WEIGHT_CHARS)
    logp = {
        c: math.log(max(p, 1e-6)) + (w * math.log(max(text[c], 1e-6)) if c in text else 0.0)
        for c, p in audio.items()
    }
    top = max(logp.values())
    e = {c: math.exp(v - top) for c, v in logp.items()}
    z = sum(e.values())
    return dict(sorted(((c, v / z) for c, v in e.items()), key=lambda kv: -kv[1]))
