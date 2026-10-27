"""Optionally hand clear speech in six languages to a second model (dsfsi-anv, MIT licence).

dsfsi-anv/za-anv-multilingual-whisper-v3-turbo was fine-tuned on Swivuriso for isiZulu, isiXhosa, Sesotho,
Setswana, Xitsonga and Tshivenda. On wideband, continuous speech it is clearly more accurate than our model
(scripts/route_sim.py: 18.5% vs 23.9% WER over the six languages' test sets), but far worse on telephone
speech and on short read prompts. So a part of a recording goes to it only when all three hold:

    its language (as detected by our model) is one of the six,
    it is at least MIN_SECONDS long (short clips are where it falls behind),
    the audio is wideband: energy above 4 kHz is not negligible (telephone audio has almost none).

Everything else keeps our model's transcript. The thresholds sit on a plateau: any minimum length from 3 to
6 s, and any bandwidth threshold from -70 to -45 dB, gave the same result in the simulation.
"""

from __future__ import annotations

import numpy as np

from .audio import SAMPLE_RATE

DSFSI_MODEL = "dsfsi-anv/za-anv-multilingual-whisper-v3-turbo"
DSFSI_PROCESSOR = "openai/whisper-large-v3-turbo"  # the repo has no tokenizer files of its own
DSFSI_LANGS = frozenset({"zul", "xho", "sot", "tsn", "tso", "ven"})
MIN_SECONDS = 5.0
MIN_HIGH_BAND_DB = -60.0
BATCH = 4


def high_band_db(wav: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Energy above 4 kHz relative to 0.3-4 kHz, in dB (about -80 for telephone audio, -30 or more otherwise)."""
    n = 512
    if len(wav) < n:
        return -100.0
    frames = np.lib.stride_tricks.sliding_window_view(wav.astype(np.float64), n)[:: n // 2]
    spec = (np.abs(np.fft.rfft(frames * np.hanning(n), axis=1)) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    low, high = spec[(freqs >= 300) & (freqs < 4000)].sum(), spec[freqs >= 4000].sum()
    return float(10 * np.log10((high + 1e-12) / (low + 1e-12)))


def eligible(lang_code: str | None, audio: np.ndarray) -> bool:
    return (lang_code in DSFSI_LANGS and len(audio) >= MIN_SECONDS * SAMPLE_RATE
            and high_band_db(audio) >= MIN_HIGH_BAND_DB)


def load_second_model(device: str | None = None):
    """dsfsi-anv as a Whisper transcriber that picks its own language token (the way it was evaluated)."""
    from .backends import WhisperTranscriber

    return WhisperTranscriber(name="dsfsi-anv", model=DSFSI_MODEL, processor=DSFSI_PROCESSOR, language_mode="auto",
                              device=device)


def reroute(segments: list[dict], wav: np.ndarray, second, on_progress=None) -> int:
    """Re-transcribe the eligible segments with `second`, in place. A segment whose new text is empty keeps the
    old one. Each rerouted segment gets "engine": "dsfsi-anv". Returns how many segments changed engine."""
    todo = []
    for i, seg in enumerate(segments):
        a, b = int(seg["start"] * SAMPLE_RATE), int(seg["end"] * SAMPLE_RATE)
        if eligible(seg.get("language"), wav[a:b]):
            todo.append((i, wav[a:b]))
    changed = 0
    for k in range(0, len(todo), BATCH):
        part = todo[k : k + BATCH]
        if on_progress:
            on_progress("Re-transcribing clear parts with the second model", k, len(todo))
        for (i, _), text in zip(part, second.transcribe([audio for _, audio in part], None)):
            if text.strip():
                segments[i]["text"], segments[i]["engine"] = text.strip(), "dsfsi-anv"
                changed += 1
    return changed
