"""Corpus loading.

Every corpus is streamed from the Hugging Face Hub and mapped to one common schema:

    audio    {"bytes", "path"}  undecoded; use audio.decode_audio()
    utt_id   str
    speaker  str               "" when the corpus has no speaker ids
    text     str               raw transcript
    lang     str               ISO 639-3 code
    corpus   str

Evaluation sets are small, fixed samples of a split. They are cached locally as 16 kHz
FLAC so every model is scored on exactly the same audio without re-streaming.
"""

from __future__ import annotations

import gc
import io
import json
import logging
import os
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
from datasets import Audio, Dataset, Features, IterableDataset, Value, load_dataset, load_from_disk

from .audio import SAMPLE_RATE, decode_audio, duration_seconds
from .config import Language
from .text import normalize_for_scoring

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusSpec:
    """How to read one corpus.

    Per-language corpora have one HF config per language; languages.yaml gives the config
    name. Multilingual corpora (language_column set) have a single config, and
    languages.yaml gives the value of language_column for each language.
    """

    name: str
    repo: str
    text_column: str
    id_column: str | tuple[str, ...]  # a tuple is joined with "_" (for ids unique only per speaker)
    speaker_column: str | None
    # canonical split name (train/validation/test) -> split name in the repo
    splits: dict[str, str]
    license: str
    gated: bool = False
    language_column: str | None = None
    # For corpora without a validation split: this share of train SPEAKERS is held out as
    # validation (and removed from train), so validation speakers are never trained on.
    validation_holdout: float | None = None
    channel: str = "wideband"  # "telephone" for corpora recorded over phone lines


CORPORA: dict[str, CorpusSpec] = {
    "nchlt": CorpusSpec(
        name="nchlt",
        repo="dsfsi-anv/multilingual-nchlt-dataset",
        text_column="text",
        id_column="filename",
        speaker_column="speaker",
        splits={"train": "train", "validation": "validation", "test": "test"},
        license="CC-BY-4.0",
    ),
    # Swivuriso / African Next Voices: scripted + spontaneous speech, 48 kHz.
    # Gated: accept the terms on the dataset page and log in with `hf auth login`.
    "swivuriso": CorpusSpec(
        name="swivuriso",
        repo="dsfsi-anv/za-african-next-voices",
        text_column="transcript",
        id_column="audio_id",
        speaker_column="recorder_uuid",
        splits={"train": "train", "validation": "dev", "test": "dev_test"},
        license="CC-BY-4.0",
        gated=True,
    ),
    # FLEURS: read Wikipedia sentences. Kept out of training by default so it stays an
    # out-of-domain benchmark.
    "fleurs": CorpusSpec(
        name="fleurs",
        repo="google/fleurs",
        text_column="raw_transcription",
        id_column="id",
        speaker_column=None,
        splits={"train": "train", "validation": "validation", "test": "test"},
        license="CC-BY-4.0",
    ),
    # NCHLT South African English (the dsfsi-anv mirror above has no English config).
    "nchlt_eng": CorpusSpec(
        name="nchlt_eng",
        repo="danielshaps/nchlt_speech_eng",
        text_column="text",
        id_column=("speaker_id", "utterance_id"),
        speaker_column="speaker_id",
        splits={"train": "train", "test": "test"},
        license="CC-BY-3.0",
        validation_holdout=0.05,
    ),
    # Lwazi: telephone speech, ~200 speakers per language. This mirror has 9 languages
    # (no Afrikaans or English) in one config.
    # The dataset card states no licence; check the original (SADiLaR) terms before
    # using a model trained on it commercially.
    "lwazi": CorpusSpec(
        name="lwazi",
        repo="dsfsi/multilingual-lwazi-dataset",
        text_column="text",
        id_column="filename",
        speaker_column="speaker",
        splits={"train": "train", "validation": "validation", "test": "test"},
        license="unstated (check SADiLaR terms)",
        language_column="language",
        channel="telephone",
    ),
}

COMMON_FEATURES = Features(
    {
        "audio": Audio(decode=False),
        "utt_id": Value("string"),
        "speaker": Value("string"),
        "text": Value("string"),
        "lang": Value("string"),
        "corpus": Value("string"),
    }
)


def check_access(corpus: str) -> bool:
    """True if the current HF credentials can read the corpus (matters for gated repos)."""
    from huggingface_hub import auth_check
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    spec = CORPORA[corpus]
    try:
        auth_check(spec.repo, repo_type="dataset")
        return True
    except (GatedRepoError, RepositoryNotFoundError) as e:
        log.warning(
            "No access to %s (%s). %s",
            spec.repo,
            type(e).__name__,
            "Accept the terms at https://huggingface.co/datasets/%s and run `hf auth login`."
            % spec.repo
            if spec.gated
            else "",
        )
        return False


def corpus_languages(corpus: str, languages: list[Language]) -> list[Language]:
    return [lang for lang in languages if corpus in lang.corpora]


def in_validation_holdout(speaker: str, share: float) -> bool:
    """Stable speaker-level assignment (same answer on every machine and run)."""
    return zlib.crc32(speaker.encode("utf-8")) % 10_000 < share * 10_000


def header_duration(audio: dict) -> float | None:
    """Clip length in seconds from the file header, without decoding; None if unreadable."""
    try:
        return sf.info(io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]).duration
    except Exception:
        return None


def load_corpus(corpus: str, lang: Language, split: str, max_duration_s: float | None = None) -> IterableDataset:
    """Stream one (corpus, language, split) in the common schema.

    For multilingual corpora this filters one language out of the shared stream. That
    reads the other languages' rows too, but a mixed stream is worse: Lwazi's shards are
    sorted by language, so a mixed stream would feed the model one language at a time.

    max_duration_s drops longer clips here, from the file header, so they never reach a
    shuffle buffer downstream. Swivuriso has many multi-minute recordings; buffering
    those ran a 30 GB machine out of RAM.
    """
    spec = CORPORA[corpus]
    if corpus not in lang.corpora:
        raise ValueError(f"{corpus} has no data for {lang.code} ({lang.name})")

    repo_split = split
    if split == "validation" and "validation" not in spec.splits:
        if spec.validation_holdout is None:
            raise ValueError(f"{corpus} has no validation split")
        repo_split = "train"
    repo_split = spec.splits[repo_split]

    config = None if spec.language_column else lang.corpora[corpus]
    # PyArrow pre-buffers Parquet reads by default and keeps every byte it has read from a
    # file until the reader is garbage-collected. When streaming, that is the whole file
    # per open stream, and it piles up across files: on Kaggle the training data pipeline
    # alone went from 1 to 26 GB of RAM in 100 s. Reading without pre-buffering keeps it flat.
    import pyarrow.dataset as pads

    extra = {}
    if os.environ.get("SA_ASR_PARQUET_PREBUFFER", "0") != "1":
        extra["fragment_scan_options"] = pads.ParquetFragmentScanOptions(pre_buffer=False)
    if os.environ.get("SA_ASR_PARQUET_BATCH_SIZE"):  # experiment knob: rows per Arrow batch
        extra["batch_size"] = int(os.environ["SA_ASR_PARQUET_BATCH_SIZE"])
    ds = load_dataset(spec.repo, config, split=repo_split, streaming=True, **extra)
    # Before any filter/map: otherwise `datasets` decodes audio itself (needs torchcodec).
    ds = ds.cast_column("audio", Audio(decode=False))
    if spec.language_column:
        value = lang.corpora[corpus]
        ds = ds.filter(lambda ex: ex[spec.language_column] == value)

    if spec.validation_holdout is not None and "validation" not in spec.splits and split in ("train", "validation"):
        want_holdout = split == "validation"
        share = spec.validation_holdout
        ds = ds.filter(lambda ex: in_validation_holdout(str(ex[spec.speaker_column]), share) == want_holdout)

    if max_duration_s is not None:
        seen = [0]

        def short_enough(ex):
            # Safety net: PyArrow read buffers can sit in reference cycles that Python's
            # garbage collector rarely visits while few objects are created, so sweep
            # regularly. Costs milliseconds per few hundred rows.
            seen[0] += 1
            if seen[0] % 300 == 0:
                gc.collect()
            d = header_duration(ex["audio"])
            return d is None or d <= max_duration_s  # unreadable headers are decided after decoding

        ds = ds.filter(short_enough)

    id_cols = spec.id_column if isinstance(spec.id_column, tuple) else (spec.id_column,)

    def to_common(ex):
        return {
            "utt_id": "_".join(str(ex[c]) for c in id_cols),
            "speaker": str(ex[spec.speaker_column]) if spec.speaker_column else "",
            "text": ex[spec.text_column] or "",
            "lang": lang.code,
            "corpus": corpus,
        }

    drop = [c for c in ds.column_names if c != "audio"]
    return ds.map(to_common, remove_columns=drop, features=COMMON_FEATURES)


# ---------------------------------------------------------------------------------------
# Cached evaluation sets
# ---------------------------------------------------------------------------------------


def eval_set_path(
    cache_dir: str | Path,
    corpus: str,
    lang: str,
    split: str,
    max_samples: int | None,
    seed: int,
    max_duration_s: float = 30.0,
    shuffle_buffer: int = 500,
) -> Path:
    """Every setting that changes which utterances are picked is part of the name."""
    n = "all" if max_samples is None else str(max_samples)
    return Path(cache_dir) / f"{corpus}_{lang}_{split}_n{n}_s{seed}_d{max_duration_s:g}_b{shuffle_buffer}"


def build_eval_set(
    corpus: str,
    lang: Language,
    split: str,
    cache_dir: str | Path,
    max_samples: int | None = 200,
    seed: int = 42,
    max_duration_s: float = 30.0,
    shuffle_buffer: int = 500,
) -> Dataset:
    """Sample up to max_samples utterances once, store them as 16 kHz FLAC, reuse afterwards.

    Utterances longer than max_duration_s (Whisper's 30 s window) or with an empty
    transcript are skipped; the counts are saved in stats.json next to the data.
    """
    path = eval_set_path(cache_dir, corpus, lang.code, split, max_samples, seed, max_duration_s, shuffle_buffer)
    if (path / "dataset_info.json").exists():
        return load_from_disk(str(path))

    stream = load_corpus(corpus, lang, split)
    if max_samples is not None:
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)

    rows, skipped = [], Counter()
    for ex in stream:
        if not normalize_for_scoring(ex["text"]):
            skipped["empty_text"] += 1
            continue
        wav = decode_audio(ex["audio"])
        dur = duration_seconds(wav)
        if dur > max_duration_s:
            skipped["too_long"] += 1
            continue
        if dur < 0.3:
            skipped["too_short"] += 1
            continue
        buf = io.BytesIO()
        sf.write(buf, wav, SAMPLE_RATE, format="FLAC")
        rows.append(
            {
                "utt_id": ex["utt_id"],
                "speaker": ex["speaker"],
                "text": ex["text"],
                "lang": ex["lang"],
                "corpus": ex["corpus"],
                "duration": dur,
                "flac": buf.getvalue(),
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break

    if not rows:
        raise RuntimeError(f"No usable utterances in {corpus}/{lang.code}/{split}: {dict(skipped)}")
    ds = Dataset.from_list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))
    stats = {
        "corpus": corpus,
        "lang": lang.code,
        "split": split,
        "kept": len(rows),
        "skipped": dict(skipped),
        "hours": round(sum(r["duration"] for r in rows) / 3600, 3),
        "speakers": len({r["speaker"] for r in rows if r["speaker"]}),
    }
    (path / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("Cached eval set %s: %s", path.name, stats)
    return ds


def eval_row_audio(row: dict):
    return decode_audio({"bytes": row["flac"]})
