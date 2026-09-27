"""Corpus loading.

Every corpus is streamed from the Hugging Face Hub and mapped to one common schema:

    audio    {"bytes", "path"}  undecoded; use audio.decode_audio()
    utt_id   str
    speaker  str               "" when the corpus has no speaker ids
    text     str               raw transcript
    lang     str               ISO 639-3 code
    corpus   str

Training reads the Parquet files itself (stream_corpus): `datasets`' streaming reader
held ~3 GB per Swivuriso stream, which ran a 30 GB machine out of memory with all 11
languages. Evaluation sets are small, fixed samples of a split, read with load_corpus and
cached locally as 16 kHz FLAC so every model is scored on exactly the same audio.
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import shutil
import tempfile
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache, partial
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

# The training stream's audio column: the same {bytes, path} struct as Audio(decode=False),
# declared plainly because `datasets` 5 needs torchcodec to encode Audio in a generator.
TRAIN_AUDIO = {"bytes": Value("binary"), "path": Value("string")}

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


def _group_may_contain(row_group, column: str, value: str) -> bool | None:
    """From a row group's min/max statistics: False if column == value cannot occur in it,
    True if it can, None if the file has no statistics for the column."""
    cols = [row_group.column(j) for j in range(row_group.num_columns) if row_group.column(j).path_in_schema == column]
    st = cols[0].statistics if cols else None
    if st is None or not st.has_min_max:
        return None
    return st.min <= value <= st.max


def header_duration(audio: dict) -> float | None:
    """Clip length in seconds from the file header, without decoding; None if unreadable."""
    try:
        return sf.info(io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]).duration
    except Exception:
        return None


@lru_cache(maxsize=None)
def files_containing(repo: str, split: str, column: str, value: str, fs=None) -> tuple[str, ...] | None:
    """The Parquet files of a split that can hold rows with column == value, as hf:// URLs.

    Decided from the row-group min/max statistics in each file's footer, so no audio is
    downloaded. Lwazi's 16 files are sorted by language and each language sits in one or
    two of them; without this every language stream downloads all 16 and discards ~90%.
    Returns None when it cannot be decided (then the caller streams everything).
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    try:
        fs = fs or HfFileSystem()
        files = sorted(fs.glob(f"datasets/{repo}/**/{split}-*.parquet"))
        keep = []
        for f in files:
            with fs.open(f, block_size=2**16) as fh:
                md = pq.ParquetFile(fh).metadata
            for i in range(md.num_row_groups):
                hit = _group_may_contain(md.row_group(i), column, value)
                if hit is None:
                    return None
                if hit:
                    keep.append(f)
                    break
        if not keep:
            return None
        log.info("%s %s: %s=%s is in %d of %d files", repo, split, column, value, len(keep), len(files))
        return tuple(f"hf://{f}" for f in keep)
    except Exception as e:  # an optimisation only; never stop a run over it
        log.warning("%s: could not pick files by %s (%s); streaming all files", repo, column, e)
        return None


def _repo_split(spec: CorpusSpec, split: str) -> str:
    """Canonical split -> the repo's split name (validation may be carved out of train)."""
    if split == "validation" and "validation" not in spec.splits:
        if spec.validation_holdout is None:
            raise ValueError(f"{spec.name} has no validation split")
        split = "train"
    return spec.splits[split]


def _holdout_wanted(spec: CorpusSpec, split: str) -> bool | None:
    """For corpora whose validation speakers are held out of train: which side to keep."""
    if spec.validation_holdout is not None and "validation" not in spec.splits and split in ("train", "validation"):
        return split == "validation"
    return None


def load_corpus(corpus: str, lang: Language, split: str, max_duration_s: float | None = None) -> IterableDataset:
    """Stream one (corpus, language, split) in the common schema.

    For multilingual corpora this filters one language out of the shared stream, reading
    only the files that hold that language (see files_containing). A mixed stream would be
    worse: Lwazi's shards are sorted by language, so it would feed one language at a time.

    max_duration_s drops longer clips here, from the file header, before they are decoded
    or buffered. Over a third of Swivuriso clips exceed Whisper's 30 s window.
    """
    spec = CORPORA[corpus]
    if corpus not in lang.corpora:
        raise ValueError(f"{corpus} has no data for {lang.code} ({lang.name})")
    repo_split = _repo_split(spec, split)

    config = None if spec.language_column else lang.corpora[corpus]
    files = files_containing(spec.repo, repo_split, spec.language_column, lang.corpora[corpus]) if spec.language_column else None
    if files:
        ds = load_dataset("parquet", data_files={repo_split: list(files)}, split=repo_split, streaming=True)
    else:
        ds = load_dataset(spec.repo, config, split=repo_split, streaming=True)
    # Before any filter/map: otherwise `datasets` decodes audio itself (needs torchcodec).
    ds = ds.cast_column("audio", Audio(decode=False))
    if spec.language_column:
        value = lang.corpora[corpus]
        ds = ds.filter(lambda ex: ex[spec.language_column] == value)

    want_holdout = _holdout_wanted(spec, split)
    if want_holdout is not None:
        share = spec.validation_holdout
        ds = ds.filter(lambda ex: in_validation_holdout(str(ex[spec.speaker_column]), share) == want_holdout)

    if max_duration_s is not None:
        def short_enough(ex):
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
# Training stream: our own Parquet reader
# ---------------------------------------------------------------------------------------


def corpus_files(corpus: str, lang: Language, split: str) -> list[str]:
    """hf:// URLs of the Parquet files holding (corpus, language, split)."""
    from datasets import load_dataset_builder

    spec = CORPORA[corpus]
    repo_split = _repo_split(spec, split)
    if spec.language_column:
        picked = files_containing(spec.repo, repo_split, spec.language_column, lang.corpora[corpus])
        if picked:
            return list(picked)
    config = None if spec.language_column else lang.corpora[corpus]
    return [str(f) for f in load_dataset_builder(spec.repo, config).config.data_files[repo_split]]


def parse_hf_url(url: str) -> tuple[str, str | None, str]:
    """hf://datasets/owner/name[@revision]/path -> (repo_id, revision, path_in_repo)."""
    from urllib.parse import unquote

    rest = url.removeprefix("hf://").removeprefix("datasets/")
    owner, name, path = rest.split("/", 2)
    revision = None
    if "@" in name:
        name, revision = name.split("@", 1)
        revision = unquote(revision)
    return f"{owner}/{name}", revision, path


def download_file(url: str, dest_dir: Path) -> Path:
    """Download one repo file to local disk (with retries: a long run must not die on a blip)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()  # one bar per file otherwise floods the notebook log
    repo_id, revision, path = parse_hf_url(url)
    for attempt in range(4):
        try:
            return Path(hf_hub_download(repo_id, path, repo_type="dataset", revision=revision, local_dir=dest_dir))
        except Exception as e:
            if attempt == 3:
                raise
            log.warning("Download of %s failed (%s); retrying", path, e)
            time.sleep(10 * (attempt + 1))
    raise AssertionError("unreachable")


def read_parquet_rows(files: list[str], *, corpus: str, lang_code: str, lang_value: str, split: str,
                      max_duration_s: float | None, seed: int):
    """Rows of the given files in the common schema, one file on local disk at a time.

    Each file is downloaded, read in small batches (pyarrow keeps about one row group in
    memory: ~0.3-0.7 GB for Swivuriso, whose rows are ~3 MB), then deleted. Row groups are
    read in a seeded random order; Lwazi row groups without the language are skipped.
    """
    import pyarrow.parquet as pq

    spec = CORPORA[corpus]
    id_cols = spec.id_column if isinstance(spec.id_column, tuple) else (spec.id_column,)
    columns = sorted({"audio", spec.text_column, *id_cols}
                     | ({spec.speaker_column} if spec.speaker_column else set())
                     | ({spec.language_column} if spec.language_column else set()))
    want_holdout = _holdout_wanted(spec, split)
    root = os.environ.get("SA_ASR_STREAM_DIR") or None
    for url in files:
        tmp = Path(tempfile.mkdtemp(prefix="sa_asr_", dir=root))
        try:
            pf = pq.ParquetFile(download_file(url, tmp), buffer_size=4 << 20, pre_buffer=False)
            groups = [g for g in range(pf.metadata.num_row_groups)
                      if not spec.language_column
                      or _group_may_contain(pf.metadata.row_group(g), spec.language_column, lang_value) is not False]
            random.Random(f"{seed}:{url}").shuffle(groups)
            for g in groups:
                for batch in pf.iter_batches(batch_size=8, row_groups=[g], columns=columns):
                    for ex in batch.to_pylist():
                        if spec.language_column and ex[spec.language_column] != lang_value:
                            continue
                        if want_holdout is not None and in_validation_holdout(
                            str(ex[spec.speaker_column]), spec.validation_holdout
                        ) != want_holdout:
                            continue
                        raw = ex["audio"] or {}
                        audio = {"bytes": raw.get("bytes"), "path": raw.get("path")}
                        if max_duration_s is not None:
                            d = header_duration(audio)
                            if d is not None and d > max_duration_s:  # unreadable headers are decided after decoding
                                continue
                        yield {
                            "audio": audio,
                            "utt_id": "_".join(str(ex[c]) for c in id_cols),
                            "speaker": str(ex[spec.speaker_column]) if spec.speaker_column else "",
                            "text": ex[spec.text_column] or "",
                            "lang": lang_code,
                            "corpus": corpus,
                        }
            pf.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


TRAIN_FEATURES = Features({**COMMON_FEATURES, "audio": TRAIN_AUDIO})


def stream_corpus(corpus: str, lang: Language, split: str = "train", max_duration_s: float | None = None,
                  seed: int = 0) -> IterableDataset:
    """Training stream for one (corpus, language), in the common schema.

    One shard per file, so IterableDataset.shuffle() shuffles the file order.
    """
    if corpus not in lang.corpora:
        raise ValueError(f"{corpus} has no data for {lang.code} ({lang.name})")
    files = corpus_files(corpus, lang, split)
    if not files or not all(f.endswith(".parquet") for f in files):
        raise ValueError(f"{corpus}/{lang.code}/{split}: expected Parquet files, got {files[:3]}")
    gen = partial(read_parquet_rows, corpus=corpus, lang_code=lang.code, lang_value=lang.corpora[corpus],
                  split=split, max_duration_s=max_duration_s, seed=seed)
    return IterableDataset.from_generator(gen, gen_kwargs={"files": files}, features=TRAIN_FEATURES)


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
