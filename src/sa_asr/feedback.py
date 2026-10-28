"""Corrections made in the app, kept as a small training corpus.

Each saved correction is the audio (16 kHz FLAC) plus the right language and transcript,
alongside what the model predicted. They live in data/feedback/ (git-ignored):

    data/feedback/corrections.jsonl   one JSON object per correction
    data/feedback/audio/<utt_id>.flac

export_parquet() writes them in the same row layout as the training corpora (audio
{bytes, path}, utt_id, speaker, text, lang), so they can be pushed to a private Hub dataset
(scripts/feedback.py push) and mixed into a later fine-tuning run.

Every correction belongs to a fixed split decided by its audio alone: about 30% are "test" and are never
trained on (so the model's accuracy on them stays honest), the rest are "train". report() measures how the
model did on the corrections: how often its language and its words were right.
"""

from __future__ import annotations

import hashlib
import json
import time
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import SAMPLE_RATE, duration_seconds
from .config import PROJECT_ROOT

FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
TEST_SHARE = 0.3


def split_of(utt_id: str) -> str:
    """"test" for about TEST_SHARE of the recordings, "train" for the rest; decided by the audio's digest, so a
    recording keeps its split however often it is corrected."""
    digest = utt_id.split("-", 2)[-1]
    return "test" if zlib.crc32(digest.encode()) % 100 < TEST_SHARE * 100 else "train"


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
    languages_present: list[str] | None = None,
    word_langs: list[str] | None = None,
    segments: list[dict] | None = None,
    root: Path = FEEDBACK_DIR,
) -> dict:
    """Store one corrected example; returns the stored record. word_langs gives the language of each word of
    text (split on spaces) for code-switched speech; without it every word is in `lang`. segments are the
    corrected parts of a mixed recording ({start, end, lang, text}); without them the whole clip is one part."""
    words = text.split()
    if word_langs is not None and len(word_langs) != len(words):
        raise ValueError(f"word_langs has {len(word_langs)} labels for {len(words)} words")
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
        # every language heard in the recording (more than one for code-switched or mixed audio)
        "languages_present": sorted(set(languages_present or [lang]) | {lang} | set(word_langs or [])
                                    | {s["lang"] for s in segments or []}),
        "word_langs": word_langs or [lang] * len(words),
        "segments": [{"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2), "lang": s["lang"],
                      "text": s["text"].strip()} for s in segments or []]
                    or [{"start": 0.0, "end": round(duration_seconds(wav), 2), "lang": lang, "text": text.strip()}],
        "language_corrected": predicted_lang is not None and predicted_lang != lang,
        "text_corrected": predicted_text.strip() != text.strip(),
        "split": split_of(utt_id),
        "duration": round(duration_seconds(wav), 2),
        "model": model,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def upload_correction(record: dict, repo: str, root: Path = FEEDBACK_DIR) -> None:
    """Copy one saved correction to a private Hub dataset (raw/<utt_id>.json + raw/<utt_id>.flac). Used where the
    local disk does not last, such as a Hugging Face Space; pull() brings them back into a local folder."""
    from huggingface_hub import CommitOperationAdd, HfApi

    _, audio_dir = _paths(root)
    utt = record["utt_id"]
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.create_commit(repo, repo_type="dataset", commit_message=f"correction {utt}", operations=[
        CommitOperationAdd(path_in_repo=f"raw/{utt}.json",
                           path_or_fileobj=json.dumps(record, ensure_ascii=False).encode()),
        CommitOperationAdd(path_in_repo=f"raw/{utt}.flac", path_or_fileobj=str(audio_dir / f"{utt}.flac")),
    ])


def pull(repo: str, root: Path = FEEDBACK_DIR) -> int:
    """Add the corrections uploaded to `repo` (by upload_correction) that are not in `root` yet; returns how many."""
    import shutil

    from huggingface_hub import snapshot_download

    log_file, audio_dir = _paths(root)
    audio_dir.mkdir(parents=True, exist_ok=True)
    have = set()
    if log_file.exists():
        have = {json.loads(x)["utt_id"] for x in log_file.read_text(encoding="utf-8").splitlines() if x.strip()}
    raw = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=["raw/*"])) / "raw"
    new = 0
    with log_file.open("a", encoding="utf-8") as f:
        for p in sorted(raw.glob("*.json")) if raw.exists() else []:
            rec = json.loads(p.read_text(encoding="utf-8"))
            flac = p.with_suffix(".flac")
            if rec["utt_id"] in have or not flac.exists():
                continue
            shutil.copyfile(flac, audio_dir / flac.name)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            new += 1
    return new


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


def report(root: Path = FEEDBACK_DIR) -> dict:
    """How the model did on the corrections, from what it predicted at the time: language accuracy, and the word
    error rate of its text against the corrected text (lowercased, punctuation removed), overall and per language,
    for the "test" split and for everything."""
    import jiwer

    from .text import normalize_for_scoring

    def measure(recs: list[dict]) -> dict:
        with_lang = [r for r in recs if r.get("predicted_lang")]
        pairs = [(normalize_for_scoring(r["text"]), normalize_for_scoring(r["predicted_text"])) for r in recs]
        pairs = [(a, b) for a, b in pairs if a]
        wer = None
        if pairs:
            out = jiwer.process_words([a for a, _ in pairs], [b if b else "-" for _, b in pairs])
            wer = round(100 * (out.substitutions + out.deletions + out.insertions)
                        / max(out.substitutions + out.deletions + out.hits, 1), 1)
        cer = round(100 * jiwer.cer([a.replace(" ", "") for a, _ in pairs], [b.replace(" ", "") or "-" for _, b in pairs]), 1) if pairs else None
        right = sum(r["predicted_lang"] == r["lang"] for r in with_lang)
        return {"count": len(recs), "language_right": round(100 * right / len(with_lang), 1) if with_lang else None,
                "wer": wer, "cer_nospace": cer}

    recs = load_corrections(root)
    for r in recs:
        r.setdefault("split", split_of(r["utt_id"]))
    out = {"all": measure(recs), "test": measure([r for r in recs if r["split"] == "test"])}
    by_lang: dict[str, list[dict]] = {}
    for r in recs:
        by_lang.setdefault(r["lang"], []).append(r)
    out["by_language"] = {c: measure(v) for c, v in sorted(by_lang.items())}
    confusions: dict[str, int] = {}
    for r in recs:
        if r.get("predicted_lang") and r["predicted_lang"] != r["lang"]:
            key = f"{r['lang']} heard as {r['predicted_lang']}"
            confusions[key] = confusions.get(key, 0) + 1
    out["language_mistakes"] = dict(sorted(confusions.items(), key=lambda kv: -kv[1]))
    return out


def export_parquet(out_file: Path, root: Path = FEEDBACK_DIR, split: str | None = None) -> int:
    """Write corrections as one Parquet file in the training-corpus layout; split="train" or "test" writes only
    that split (None writes all)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _, audio_dir = _paths(root)
    rows = []
    for r in load_corrections(root):
        flac = audio_dir / f"{r['utt_id']}.flac"
        if not flac.exists() or (split and split_of(r["utt_id"]) != split):
            continue
        rows.append({
            "audio": {"bytes": flac.read_bytes(), "path": flac.name},
            "utt_id": r["utt_id"],
            "speaker": "",
            "text": r["text"],
            "lang": r["lang"],
            "predicted_lang": r["predicted_lang"],
            "predicted_text": r["predicted_text"],
            "languages_present": r.get("languages_present") or [r["lang"]],
            "word_langs": r.get("word_langs") or [r["lang"]] * len(r["text"].split()),
            "segments": r.get("segments") or [{"start": 0.0, "end": r["duration"], "lang": r["lang"],
                                               "text": r["text"]}],
            "duration": r["duration"],
        })
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_file, row_group_size=100)
    return len(rows)
