"""Corrections made in the app, kept as a small training corpus.

Each saved correction is the audio (16 kHz FLAC) plus the right language and transcript,
alongside what the model predicted. They live in data/feedback/ (git-ignored):

    data/feedback/corrections.jsonl   one JSON object per correction
    data/feedback/audio/<utt_id>.flac

export_parquet() writes them in the same row layout as the training corpora (audio
{bytes, path}, utt_id, speaker, text, lang), so they can be pushed to a private Hub dataset
(scripts/feedback.py push) and mixed into a later fine-tuning run.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import SAMPLE_RATE, duration_seconds
from .config import PROJECT_ROOT

FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"


def _paths(root: Path) -> tuple[Path, Path]:
    return root / "corrections.jsonl", root / "audio"


def save_correction(
    wav: np.ndarray,
    lang: str,
    text: str,
    *,
    predicted_lang: str | None,
    predicted_text: str,
    model: str,
    root: Path = FEEDBACK_DIR,
) -> dict:
    """Store one corrected example; returns the stored record."""
    log_file, audio_dir = _paths(root)
    audio_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(np.ascontiguousarray(wav, dtype=np.float32).tobytes()).hexdigest()[:10]
    utt_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{digest}"
    sf.write(audio_dir / f"{utt_id}.flac", wav, SAMPLE_RATE, format="FLAC")
    record = {
        "utt_id": utt_id,
        "lang": lang,
        "text": text.strip(),
        "predicted_lang": predicted_lang,
        "predicted_text": predicted_text,
        "language_corrected": predicted_lang is not None and predicted_lang != lang,
        "text_corrected": predicted_text.strip() != text.strip(),
        "duration": round(duration_seconds(wav), 2),
        "model": model,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_corrections(root: Path = FEEDBACK_DIR) -> list[dict]:
    """All corrections; a later correction of the same audio replaces an earlier one."""
    log_file, _ = _paths(root)
    if not log_file.exists():
        return []
    latest: dict[str, dict] = {}
    for line in log_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[rec["utt_id"].split("-", 2)[-1]] = rec  # keyed by audio digest
    return list(latest.values())


def summary(root: Path = FEEDBACK_DIR) -> dict:
    recs = load_corrections(root)
    by_lang: dict[str, int] = {}
    for r in recs:
        by_lang[r["lang"]] = by_lang.get(r["lang"], 0) + 1
    return {
        "count": len(recs),
        "hours": round(sum(r["duration"] for r in recs) / 3600, 3),
        "language_fixes": sum(r["language_corrected"] for r in recs),
        "text_fixes": sum(r["text_corrected"] for r in recs),
        "by_language": dict(sorted(by_lang.items())),
    }


def export_parquet(out_file: Path, root: Path = FEEDBACK_DIR) -> int:
    """Write all corrections as one Parquet file in the training-corpus layout."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _, audio_dir = _paths(root)
    rows = []
    for r in load_corrections(root):
        flac = audio_dir / f"{r['utt_id']}.flac"
        if not flac.exists():
            continue
        rows.append({
            "audio": {"bytes": flac.read_bytes(), "path": flac.name},
            "utt_id": r["utt_id"],
            "speaker": "",
            "text": r["text"],
            "lang": r["lang"],
            "predicted_lang": r["predicted_lang"],
            "predicted_text": r["predicted_text"],
            "duration": r["duration"],
        })
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_file, row_group_size=100)
    return len(rows)
