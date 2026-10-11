"""End-to-end transcription of arbitrary-length audio (used by the app; reusable by an API)."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, decode_audio, duration_seconds, split_long_audio
from .backends import META_FILE, WhisperTranscriber, localize_model
from .config import PROJECT_ROOT, Language

CHUNK_BATCH = 8
# Language detection runs on windows of 4-10 s cut at pauses: long enough to identify a
# language, short enough to notice a switch. One-language detection samples at most
# DETECT_MAX_WINDOWS of them; mixed-language detection reads every window.
DETECT_MIN_S, DETECT_MAX_S = 4.0, 10.0
DETECT_MAX_WINDOWS = 12
SWITCH_PENALTY = 2.0  # in log-probability units; see smooth_languages


def resolve_model(model: str) -> tuple[str, list[str] | None]:
    """(model id or local path, languages it was trained on, or None if not trained here).

    Accepts a project-relative folder, an absolute folder, or a Hugging Face repo id
    (a repo pushed by train.py --hub-repo is downloaded).
    """
    path = Path(model)
    if not path.is_absolute() and (PROJECT_ROOT / path).exists():
        path = PROJECT_ROOT / path
    else:
        path = localize_model(model)
    meta = path / META_FILE
    if meta.exists():
        return str(path), json.loads(meta.read_text(encoding="utf-8"))["languages"]
    return (str(path) if path.exists() else model), None


def load_transcriber(model: str, finetuned: bool, num_beams: int = 1) -> WhisperTranscriber:
    # A zero-shot model is told the language only when Whisper knows it (en, af) = "native".
    return WhisperTranscriber(
        name=Path(model).name, model=model, language_mode="proxy" if finetuned else "native", num_beams=num_beams
    )


def load_audio(data: bytes, filename: str = "audio") -> np.ndarray:
    """Decode with soundfile; fall back to ffmpeg for formats it can't read (m4a, webm, ...)."""
    try:
        return decode_audio({"bytes": data})
    except Exception as e:
        if not shutil.which("ffmpeg"):
            raise ValueError(
                f"Can't read '{filename}' without ffmpeg. Install ffmpeg, or convert the file to WAV, MP3 or FLAC."
            ) from e
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", "pipe:1"],
        input=data,
        capture_output=True,
        check=True,
    )
    return decode_audio({"bytes": proc.stdout})


def detection_windows(wav: np.ndarray, max_windows: int | None = None) -> list[tuple[int, int]]:
    """(start, end) sample ranges of 4-10 s, cut at pauses, covering the whole recording.

    A final scrap under 2 s joins the window before it (too short to identify a language).
    With max_windows, an evenly spread subset (enough to identify one language for the
    whole recording without scanning every second of a long one).
    """
    starts = [int(round(s * SAMPLE_RATE)) for s, _ in split_long_audio(wav, max_s=DETECT_MAX_S, min_s=DETECT_MIN_S)]
    bounds = list(zip(starts, starts[1:] + [len(wav)]))
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < 2 * SAMPLE_RATE:
        bounds[-2:] = [(bounds[-2][0], bounds[-1][1])]
    if max_windows and len(bounds) > max_windows:
        bounds = [bounds[round(i * (len(bounds) - 1) / max(max_windows - 1, 1))] for i in range(max_windows)]
    return bounds


def smooth_languages(probs: list[dict[str, float]], weights: list[float], switch_penalty: float = SWITCH_PENALTY) -> list[str]:
    """Most likely language per window, with a cost for every change of language (Viterbi).

    Closely related languages (isiZulu/isiXhosa/siSwati, Sepedi/Sesotho/Setswana) can trade
    places on a single short window; the switch cost keeps the language steady unless the
    evidence for a change is strong. Weights scale each window's evidence (by duration).
    """
    codes = list(probs[0])
    score = {c: weights[0] * math.log(max(probs[0][c], 1e-6)) for c in codes}
    back: list[dict[str, str]] = []
    for p, w in zip(probs[1:], weights[1:]):
        best_prev = max(score, key=score.get)
        new, ptr = {}, {}
        for c in codes:
            stay, switch = score[c], score[best_prev] - switch_penalty
            new[c], ptr[c] = (stay, c) if stay >= switch else (switch, best_prev)
            new[c] += w * math.log(max(p.get(c, 0.0), 1e-6))
        back.append(ptr)
        score = new
    path = [max(score, key=score.get)]
    for ptr in reversed(back):
        path.append(ptr[path[-1]])
    return path[::-1]


def _detect(model: WhisperTranscriber, wav: np.ndarray, bounds: list[tuple[int, int]], candidates: list[Language]):
    probs: list[dict[str, float]] = []
    for i in range(0, len(bounds), CHUNK_BATCH):
        probs += model.detect_languages([wav[a:b] for a, b in bounds[i : i + CHUNK_BATCH]], candidates)
    return probs


def _weighted_mean(probs: list[dict[str, float]], weights: list[float]) -> dict[str, float]:
    total = sum(weights)
    mean = {c: sum(w * p[c] for p, w in zip(probs, weights)) / total for c in probs[0]}
    return dict(sorted(mean.items(), key=lambda kv: -kv[1]))


def transcribe_audio(
    model: WhisperTranscriber,
    wav: np.ndarray,
    languages: dict[str, Language],
    lang_code: str | None = None,
    mixed: bool = False,
) -> dict:
    """Transcribe audio of any length.

    lang_code given: transcribe everything in that language.
    lang_code None (a model trained here): detect the language across the whole recording,
      from 4-10 s windows. mixed=False uses the single most likely language; mixed=True
      lets the language change along the recording (e.g. the national anthem), and each
      stretch is transcribed in its own language.
    A zero-shot model with no language lets Whisper choose.
    """
    t0 = time.time()
    detected = None
    runs: list[tuple[int, int, Language | None]]  # (start, end, language) sample ranges
    if lang_code is not None:
        runs = [(0, len(wav), languages[lang_code])]
    elif model.proxy_tokens is not None:
        candidates = [languages[c] for c in model.proxy_tokens if c in languages]
        bounds = detection_windows(wav, max_windows=None if mixed else DETECT_MAX_WINDOWS)
        probs = _detect(model, wav, bounds, candidates)
        weights = [(b - a) / SAMPLE_RATE / 5 for a, b in bounds]  # a 10 s window counts twice a 5 s one
        detected = _weighted_mean(probs, weights)
        if mixed:
            labels = smooth_languages(probs, weights)
            runs = []
            for (a, b), code in zip(bounds, labels):
                if runs and runs[-1][2].code == code:
                    runs[-1] = (runs[-1][0], b, runs[-1][2])
                else:
                    runs.append((a, b, languages[code]))
        else:
            runs = [(0, len(wav), languages[next(iter(detected))])]
    else:
        runs = [(0, len(wav), None)]

    segments = []
    for a, b, lang in runs:
        chunks = split_long_audio(wav[a:b])
        texts: list[str] = []
        for i in range(0, len(chunks), CHUNK_BATCH):
            texts += model.transcribe([c for _, c in chunks[i : i + CHUNK_BATCH]], lang)
        offset = a / SAMPLE_RATE
        segments += [
            {"start": round(offset + start, 2), "end": round(offset + start + duration_seconds(chunk), 2),
             "text": text, "language": lang.code if lang else None}
            for (start, chunk), text in zip(chunks, texts)
        ]

    share: dict[str, float] = {}
    for a, b, lang in runs:
        if lang is not None:
            share[lang.code] = share.get(lang.code, 0.0) + (b - a) / max(len(wav), 1)
    share = dict(sorted(share.items(), key=lambda kv: -kv[1]))
    return {
        "text": " ".join(s["text"] for s in segments if s["text"]).strip(),
        "segments": segments,
        "language": next(iter(share), None),  # the main language (most audio)
        "languages": share,  # share of the audio per language; more than one = mixed
        "detected": detected,
        "duration": duration_seconds(wav),
        "elapsed": time.time() - t0,
    }
