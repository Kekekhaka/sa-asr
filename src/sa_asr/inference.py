"""End-to-end transcription of arbitrary-length audio (used by the app; reusable by an API)."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, decode_audio, duration_seconds, split_long_audio
from .backends import META_FILE, WhisperTranscriber, localize_model
from .config import PROJECT_ROOT, Language

CHUNK_BATCH = 8


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


def transcribe_audio(
    model: WhisperTranscriber,
    wav: np.ndarray,
    languages: dict[str, Language],
    lang_code: str | None = None,
) -> dict:
    """Transcribe audio of any length.

    lang_code None = automatic: a model trained here detects among its own languages
    (with probabilities); a zero-shot model lets Whisper choose.
    """
    t0 = time.time()
    detected = None
    if lang_code is not None:
        lang = languages[lang_code]
    elif model.proxy_tokens is not None:
        candidates = [languages[c] for c in model.proxy_tokens if c in languages]
        detected = model.detect_language(wav, candidates)
        lang = languages[next(iter(detected))]
    else:
        lang = None

    chunks = split_long_audio(wav)
    texts: list[str] = []
    for i in range(0, len(chunks), CHUNK_BATCH):
        texts += model.transcribe([c for _, c in chunks[i : i + CHUNK_BATCH]], lang)
    return {
        "text": " ".join(t for t in texts if t).strip(),
        "segments": [
            {"start": round(start, 2), "end": round(start + duration_seconds(chunk), 2), "text": text}
            for (start, chunk), text in zip(chunks, texts)
        ],
        "language": lang.code if lang else None,
        "detected": detected,
        "duration": duration_seconds(wav),
        "elapsed": time.time() - t0,
    }
