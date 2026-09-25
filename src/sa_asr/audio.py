"""Audio decoding.

We decode audio ourselves (soundfile + soxr) instead of letting `datasets` do it. Recent
`datasets` versions decode through torchcodec, which needs system FFmpeg libraries and is
awkward on Windows. Reading the raw bytes keeps the pipeline working the same on a laptop
and on Colab.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import soundfile as sf
import soxr

SAMPLE_RATE = 16_000


def decode_audio(audio: dict[str, Any], target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Turn a `datasets` audio cell (decoded or not) into mono float32 at target_sr."""
    if audio.get("array") is not None:
        wav = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
    else:
        source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
        wav, sr = sf.read(source, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != target_sr:
        wav = soxr.resample(wav, sr, target_sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def duration_seconds(wav: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    return len(wav) / sr


def split_long_audio(
    wav: np.ndarray, sr: int = SAMPLE_RATE, max_s: float = 28.0, min_s: float = 15.0, frame_s: float = 0.1
) -> list[tuple[float, np.ndarray]]:
    """Cut audio longer than Whisper's window into (start_seconds, chunk) pieces.

    Each cut is placed at the quietest 100 ms frame between min_s and max_s into the
    remaining audio, so words are rarely split. Returns one piece for short audio.
    """
    chunks, start = [], 0
    max_n, min_n, frame = int(max_s * sr), int(min_s * sr), int(frame_s * sr)
    while len(wav) - start > max_n:
        window = wav[start + min_n : start + max_n]
        n_frames = len(window) // frame
        energy = (window[: n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1)
        cut = start + min_n + int(np.argmin(energy)) * frame + frame // 2
        chunks.append((start / sr, wav[start:cut]))
        start = cut
    chunks.append((start / sr, wav[start:]))
    return chunks


# ---------------------------------------------------------------------------------------
# Augmentation: make clean studio/phone-app recordings look like other channels.
# ---------------------------------------------------------------------------------------


def mu_law(wav: np.ndarray, mu: int = 255) -> np.ndarray:
    """G.711 mu-law companding round trip at 8 bits (the quantisation noise of a phone call)."""
    x = np.clip(wav, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round((y + 1) / 2 * mu) / mu * 2 - 1
    return (np.sign(y) * np.expm1(np.abs(y) * np.log1p(mu)) / mu).astype(np.float32)


def telephone_channel(wav: np.ndarray, rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Narrowband phone line: 8 kHz band limit, optional mu-law codec, back to sr."""
    peak = float(np.abs(wav).max()) or 1.0
    x = wav / peak * rng.uniform(0.3, 0.9)
    x = soxr.resample(x, sr, 8_000)
    if rng.random() < 0.7:
        x = mu_law(x)
    return soxr.resample(x, 8_000, sr).astype(np.float32)


def add_noise(wav: np.ndarray, rng: np.random.Generator, snr_db: tuple[float, float] = (10.0, 30.0)) -> np.ndarray:
    """White noise at a random SNR. A stand-in until a real noise corpus (e.g. MUSAN) is wired in."""
    power = float(np.mean(wav**2)) or 1e-8
    snr = rng.uniform(*snr_db)
    noise = rng.normal(0.0, np.sqrt(power / 10 ** (snr / 10)), size=wav.shape)
    return (wav + noise).astype(np.float32)


def augment(wav: np.ndarray, rng: np.random.Generator, telephone_prob: float = 0.0, noise_prob: float = 0.0) -> np.ndarray:
    if noise_prob and rng.random() < noise_prob:
        wav = add_noise(wav, rng)
    if telephone_prob and rng.random() < telephone_prob:
        wav = telephone_channel(wav, rng)
    return wav
