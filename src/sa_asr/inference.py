"""End-to-end transcription of arbitrary-length audio (used by the app; reusable by an API)."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, decode_audio, duration_seconds, split_long_audio
from .backends import META_FILE, WhisperTranscriber, localize_model
from .config import PROJECT_ROOT, Language
from .routing import reroute
from .textlid import TextLID, combine

CHUNK_BATCH = 8
# Language detection runs on windows of 4-10 s cut at pauses: long enough to identify a
# language, short enough to notice a switch. One-language detection samples at most
# DETECT_MAX_WINDOWS of them; mixed-language detection reads every window.
DETECT_MIN_S, DETECT_MAX_S = 4.0, 10.0
DETECT_MAX_WINDOWS = 12
# In log-probability units; see smooth_languages. Chosen with scripts/lid_long.py on simulated recordings: 6 instead
# of 2 raised the share of correctly labelled windows in one-language recordings from 90% to 93% and made
# two-language recordings come out right far more often (stretches of 2+ windows), at a small cost for a
# switch that lasts a single window.
SWITCH_PENALTY = 6.0
TEXT_WEIGHT: float | dict[str, float] = 1.0  # how much the spelling of a window's text counts against the audio
# detector; a dict gives one weight per family of the audio's top language (textlid.family_weight, tuned by lid_combined.py)
TEXT_TEMPERATURE = 0.05  # softness of the spelling model's verdict (textlid.TextLID.posterior)


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


Progress = Callable[[str, int, int], None]  # (what is being done, parts done, parts in total)


def _detect(model: WhisperTranscriber, wav: np.ndarray, bounds: list[tuple[int, int]], candidates: list[Language],
            on_progress: Progress | None = None):
    probs: list[dict[str, float]] = []
    for i in range(0, len(bounds), CHUNK_BATCH):
        if on_progress:
            on_progress("Detecting the language", i, len(bounds))
        probs += model.detect_languages([wav[a:b] for a, b in bounds[i : i + CHUNK_BATCH]], candidates)
    return probs


def _weighted_mean(probs: list[dict[str, float]], weights: list[float]) -> dict[str, float]:
    total = sum(weights)
    mean = {c: sum(w * p[c] for p, w in zip(probs, weights)) / total for c in probs[0]}
    return dict(sorted(mean.items(), key=lambda kv: -kv[1]))


def _transcribe_windows(model: WhisperTranscriber, wav: np.ndarray, bounds: list[tuple[int, int]],
                        indices: list[int], labels: list[str], languages: dict[str, Language],
                        on_progress: Progress | None = None) -> dict[int, tuple[str, str]]:
    """{window index: (language code, text)}: each window (up to 10 s) transcribed in its language."""
    out: dict[int, tuple[str, str]] = {}
    for code in dict.fromkeys(labels[i] for i in indices):
        todo = [i for i in indices if labels[i] == code]
        for k in range(0, len(todo), CHUNK_BATCH):
            part = todo[k : k + CHUNK_BATCH]
            if on_progress:
                on_progress("Transcribing", len(out), len(indices))
            for i, text in zip(part, model.transcribe([wav[bounds[i][0] : bounds[i][1]] for i in part], languages[code])):
                out[i] = (code, text)
    return out


def transcribe_audio(
    model: WhisperTranscriber,
    wav: np.ndarray,
    languages: dict[str, Language],
    lang_code: str | None = None,
    mixed: bool = False,
    text_lid: TextLID | None = None,
    text_weight: float | dict[str, float] = TEXT_WEIGHT,
    on_progress: Progress | None = None,
    second_model=None,
) -> dict:
    """Transcribe audio of any length.

    lang_code given: transcribe everything in that language.
    lang_code None (a model trained here): detect the language across the whole recording,
      from 4-10 s windows. mixed=False uses the single most likely language; mixed=True
      lets the language change along the recording (e.g. the national anthem), and each
      stretch is transcribed in its own language.
    text_lid (mixed only): a second opinion from the spelling of each window's transcript. Each
      window is transcribed once in its audio-detected language, the spelling is read by the
      text model, and the two are combined before the final choice; windows whose language
      changes are transcribed again in the new language.
    A zero-shot model with no language lets Whisper choose.
    on_progress(what, done, total) is called before each batch of windows; a caller that raises
    from it (a Stop button) cancels the run between batches.
    second_model: an optional second transcriber (routing.load_second_model). Parts of the result in its six
      languages that are long and wideband are transcribed again by it (see routing.py).
    """
    t0 = time.time()
    detected = None
    runs: list[tuple[int, int, Language | None]]  # (start, end, language) sample ranges
    window_texts: dict[int, tuple[str, str]] = {}  # window index -> (language, text), when already transcribed
    bounds: list[tuple[int, int]] = []
    if lang_code is not None:
        runs = [(0, len(wav), languages[lang_code])]
    elif model.proxy_tokens is not None:
        candidates = [languages[c] for c in model.proxy_tokens if c in languages]
        bounds = detection_windows(wav, max_windows=None if mixed else DETECT_MAX_WINDOWS)
        probs = _detect(model, wav, bounds, candidates, on_progress)
        weights = [(b - a) / SAMPLE_RATE / 5 for a, b in bounds]  # a 10 s window counts twice a 5 s one
        if mixed:
            labels = smooth_languages(probs, weights)
            if text_lid is not None and probs:
                window_texts = _transcribe_windows(model, wav, bounds, list(range(len(bounds))), labels, languages, on_progress)
                probs = [combine(p, text_lid.posterior(window_texts[i][1], TEXT_TEMPERATURE), text_weight, len(window_texts[i][1]))
                         for i, p in enumerate(probs)]
                labels = smooth_languages(probs, weights)
                changed = [i for i, (code, _) in window_texts.items() if code != labels[i]]
                window_texts.update(_transcribe_windows(model, wav, bounds, changed, labels, languages, on_progress))
            detected = _weighted_mean(probs, weights)
            runs = []
            for (a, b), code in zip(bounds, labels):
                if runs and runs[-1][2].code == code:
                    runs[-1] = (runs[-1][0], b, runs[-1][2])
                else:
                    runs.append((a, b, languages[code]))
        else:
            detected = _weighted_mean(probs, weights)
            runs = [(0, len(wav), languages[next(iter(detected))])]
    else:
        runs = [(0, len(wav), None)]

    segments = []
    if window_texts:  # the text check already transcribed every window
        segments = [
            {"start": round(a / SAMPLE_RATE, 2), "end": round(b / SAMPLE_RATE, 2),
             "text": window_texts[i][1], "language": window_texts[i][0]}
            for i, (a, b) in enumerate(bounds)
        ]
        runs = []
        for seg, (a, b) in zip(segments, bounds):
            lang = languages[seg["language"]]
            if runs and runs[-1][2].code == lang.code:
                runs[-1] = (runs[-1][0], b, lang)
            else:
                runs.append((a, b, lang))
    for a, b, lang in ([] if window_texts else runs):
        chunks = split_long_audio(wav[a:b])
        texts: list[str] = []
        for i in range(0, len(chunks), CHUNK_BATCH):
            if on_progress:
                on_progress("Transcribing", i, len(chunks))
            texts += model.transcribe([c for _, c in chunks[i : i + CHUNK_BATCH]], lang)
        offset = a / SAMPLE_RATE
        segments += [
            {"start": round(offset + start, 2), "end": round(offset + start + duration_seconds(chunk), 2),
             "text": text, "language": lang.code if lang else None}
            for (start, chunk), text in zip(chunks, texts)
        ]

    rerouted = reroute(segments, wav, second_model, on_progress) if second_model is not None else 0

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
        "rerouted": rerouted,  # segments re-transcribed by the second model
        "duration": duration_seconds(wav),
        "elapsed": time.time() - t0,
    }
