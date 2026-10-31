"""Audio decoding.

We decode audio ourselves (soundfile + soxr) instead of letting `datasets` do it. Recent
`datasets` versions decode through torchcodec, which needs system FFmpeg libraries and is
awkward on Windows. Reading the raw bytes keeps the pipeline working the same on a laptop
and on Colab.
"""

from __future__ import annotations

import io
from pathlib import Path
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


def packet_loss(wav: np.ndarray, rng: np.random.Generator, rate: tuple[float, float] = (0.01, 0.05),
                frame_ms: int = 20, burst: tuple[int, int] = (1, 3), sr: int = SAMPLE_RATE) -> np.ndarray:
    """Dropouts of a mobile or internet call: 20 ms frames go silent in bursts of 1-3 (20-60 ms gaps) until
    1-5% of the audio is lost. Short enough that every word is still partly heard, so the transcript holds."""
    frame = sr * frame_ms // 1000
    n = len(wav) // frame
    if n < 10:
        return wav
    out = wav.copy()
    lost, target = 0, int(round(rng.uniform(*rate) * n))
    while lost < target:
        start, length = int(rng.integers(0, n)), int(rng.integers(burst[0], burst[1] + 1))
        out[start * frame : (start + length) * frame] = 0.0
        lost += length
    return out


def add_noise(wav: np.ndarray, rng: np.random.Generator, snr_db: tuple[float, float] = (10.0, 30.0)) -> np.ndarray:
    """White noise at a random SNR. A stand-in until a real noise corpus (e.g. MUSAN) is wired in."""
    power = float(np.mean(wav**2)) or 1e-8
    snr = rng.uniform(*snr_db)
    noise = rng.normal(0.0, np.sqrt(power / 10 ** (snr / 10)), size=wav.shape)
    return (wav + noise).astype(np.float32)


def speed_perturb(wav: np.ndarray, rng: np.random.Generator, factors: tuple[float, ...] = (0.9, 1.1),
                  max_seconds: float = 30.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Speak faster or slower (changes pitch too, as in Kaldi's speed perturbation). Left alone if the slowed
    audio would pass max_seconds, because Whisper would cut the end off and the text would no longer match."""
    f = float(rng.choice(factors))
    if len(wav) / f / sr > max_seconds:
        return wav
    return soxr.resample(wav, int(round(sr * f)), sr).astype(np.float32)


def reverberate(wav: np.ndarray, rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A simulated room: convolve with an exponentially decaying noise burst (reverberation time 0.2-0.8 s)."""
    rt60 = rng.uniform(0.2, 0.8)
    t = np.arange(int(rt60 * sr)) / sr
    rir = rng.normal(size=t.size) * np.exp(-6.9 * t / rt60)  # -60 dB at rt60
    rir[0] = np.abs(rir).max()  # direct sound
    rir /= np.sqrt(np.sum(rir**2))
    n = len(wav) + len(rir) - 1
    out = np.fft.irfft(np.fft.rfft(wav, n) * np.fft.rfft(rir, n), n)[: len(wav)]
    return (out * (np.abs(wav).max() / (np.abs(out).max() or 1.0))).astype(np.float32)


def noise_files(noise_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(noise_dir).rglob("*") if p.suffix.lower() in {".wav", ".flac", ".ogg"})


def add_real_noise(wav: np.ndarray, rng: np.random.Generator, files: list[Path],
                   snr_db: tuple[float, float] = (0.0, 20.0), sr: int = SAMPLE_RATE) -> np.ndarray:
    """Mix in a random stretch of a noise recording (e.g. MUSAN noise/music/speech babble) at a random SNR."""
    noise, nsr = sf.read(str(files[int(rng.integers(len(files)))]), dtype="float32", always_2d=True)
    noise = noise.mean(axis=1)
    if nsr != sr:
        noise = soxr.resample(noise, nsr, sr)
    if len(noise) < len(wav):
        noise = np.tile(noise, int(np.ceil(len(wav) / max(len(noise), 1))))
    start = int(rng.integers(0, len(noise) - len(wav) + 1))
    noise = noise[start : start + len(wav)]
    p_noise = float(np.mean(noise**2)) or 1e-8
    p_wav = float(np.mean(wav**2)) or 1e-8
    scale = np.sqrt(p_wav / (p_noise * 10 ** (rng.uniform(*snr_db) / 10)))
    return (wav + scale * noise).astype(np.float32)


def augment(wav: np.ndarray, rng: np.random.Generator, telephone_prob: float = 0.0, noise_prob: float = 0.0,
            speed_prob: float = 0.0, reverb_prob: float = 0.0, noise_files_: list[Path] | None = None,
            packet_loss_prob: float = 0.0) -> np.ndarray:
    """Speed, then room, then noise (real recordings when noise_files_ is given, else white), then phone line,
    whose audio then loses packets with probability packet_loss_prob."""
    if speed_prob and rng.random() < speed_prob:
        wav = speed_perturb(wav, rng)
    if reverb_prob and rng.random() < reverb_prob:
        wav = reverberate(wav, rng)
    if noise_prob and rng.random() < noise_prob:
        wav = add_real_noise(wav, rng, noise_files_) if noise_files_ else add_noise(wav, rng)
    if telephone_prob and rng.random() < telephone_prob:
        wav = telephone_channel(wav, rng)
        if packet_loss_prob and rng.random() < packet_loss_prob:
            wav = packet_loss(wav, rng)
    return wav
