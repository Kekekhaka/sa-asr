"""Pieces for fine-tuning Whisper: label building, the training stream, collator, trainer."""

from __future__ import annotations

import inspect
import json
import logging
import unicodedata
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, IterableDataset, interleave_datasets
from transformers import Seq2SeqTrainer, TrainerCallback, WhisperProcessor

from .audio import SAMPLE_RATE, augment, decode_audio, duration_seconds
from .config import Language
from .data import CORPORA, eval_row_audio, stream_corpus
from .metrics import score
from .text import clean_transcript, normalize_for_scoring

log = logging.getLogger(__name__)


class LabelBuilder:
    """Builds Whisper decoder targets with a per-language (proxy) token.

    Labels are  <|lang|> <|transcribe|> <|notimestamps|> text <|endoftext|>.
    The model prepends <|startoftranscript|> itself when it shifts labels right, and the
    loss on <|lang|> is what teaches it to identify the language.
    """

    def __init__(self, processor: WhisperProcessor, languages: list[Language], text_mode: str = "clean"):
        if text_mode not in ("clean", "normalized"):
            raise ValueError("text_mode must be 'clean' or 'normalized'")
        tok = processor.tokenizer
        self.tokenizer = tok
        self.text_mode = text_mode
        self.lang_ids = {}
        for lang in languages:
            tid = tok.convert_tokens_to_ids(f"<|{lang.whisper_token}|>")
            if tid is None or tid == tok.unk_token_id:
                raise ValueError(f"Tokenizer has no <|{lang.whisper_token}|> token (for {lang.code})")
            self.lang_ids[lang.code] = tid
        self.transcribe_id = tok.convert_tokens_to_ids("<|transcribe|>")
        self.notimestamps_id = tok.convert_tokens_to_ids("<|notimestamps|>")
        self.eot_id = tok.eos_token_id

    def target_text(self, text: str) -> str:
        return normalize_for_scoring(text) if self.text_mode == "normalized" else clean_transcript(text)

    def __call__(self, text: str, lang: str) -> list[int]:
        body = self.tokenizer.encode(self.target_text(text), add_special_tokens=False)
        return [self.lang_ids[lang], self.transcribe_id, self.notimestamps_id, *body, self.eot_id]


def make_example_fn(
    processor: WhisperProcessor,
    labels: LabelBuilder,
    max_duration_s: float,
    max_label_tokens: int,
    telephone_prob: float = 0.0,
    noise_prob: float = 0.0,
    seed: int = 0,
):
    """Row in the common schema -> {input_features, labels, keep}.

    Wideband audio is turned into simulated phone audio with probability telephone_prob,
    so one model serves both phone calls and app/microphone recordings. Corpora that were
    recorded over the phone (Lwazi) are left alone. The augmentation is seeded per
    utterance, so a run is reproducible.
    """
    fe = processor.feature_extractor

    def fn(ex: dict[str, Any]) -> dict[str, Any]:
        wav = decode_audio(ex["audio"])
        ids = labels(ex["text"], ex["lang"])
        dur = duration_seconds(wav)
        keep = 0.3 <= dur <= max_duration_s and len(ids) > 4 and len(ids) <= max_label_tokens
        feats = None
        if keep:
            if telephone_prob or noise_prob:
                rng = np.random.default_rng(zlib.crc32(f"{seed}:{ex['corpus']}:{ex['utt_id']}".encode()))
                wideband = CORPORA[ex["corpus"]].channel != "telephone"
                wav = augment(wav, rng, telephone_prob=telephone_prob if wideband else 0.0, noise_prob=noise_prob)
            feats = fe(wav, sampling_rate=SAMPLE_RATE).input_features[0]
        return {"input_features": feats, "labels": ids, "keep": keep}

    return fn


def build_train_stream(
    streams: list[tuple[str, Language, float]],
    example_fn,
    seed: int,
    shuffle_buffer: int,
    max_duration_s: float | None = None,
) -> IterableDataset:
    """Mix (corpus, language, weight) streams by weight into one endless training stream.

    Clips longer than max_duration_s are dropped before shuffling (see read_parquet_rows).
    """
    # datasets >= 5 fills a shuffle buffer from up to 10 files of a stream at once, each
    # open file holding its read buffers. With ~23 streams, and Swivuriso files of ~700 MB,
    # that took a 30 GB Kaggle machine out of memory in under two minutes. One file at a
    # time per stream keeps RAM flat; the file order is still shuffled, and mixing many
    # streams already interleaves sources.
    one_file = {}
    if "max_buffer_input_shards" in inspect.signature(IterableDataset.shuffle).parameters:
        one_file["max_buffer_input_shards"] = 1
    import datasets

    log.info("datasets %s: shuffle buffers read %s", datasets.__version__,
             "one file at a time" if one_file else "one file at a time (library default)")
    parts, weights = [], []
    for corpus, lang, weight in streams:
        stream = stream_corpus(corpus, lang, "train", max_duration_s=max_duration_s, seed=seed)
        parts.append(stream.shuffle(seed=seed, buffer_size=shuffle_buffer, **one_file))
        weights.append(weight)
    probs = [w / sum(weights) for w in weights]
    mixed = interleave_datasets(parts, probabilities=probs, seed=seed, stopping_strategy="all_exhausted")
    mixed = mixed.map(example_fn, remove_columns=["audio", "text", "utt_id", "speaker"])
    return mixed.filter(lambda ex: ex["keep"])


class EvalFeatureDataset(torch.utils.data.Dataset):
    """Cached eval rows -> model inputs, computed lazily (nothing large written to disk)."""

    def __init__(self, rows: Dataset, processor: WhisperProcessor, labels: LabelBuilder, max_label_tokens: int):
        self.rows = rows
        self.fe = processor.feature_extractor
        self.labels = labels
        self.max_label_tokens = max_label_tokens

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        # Cached eval sets are already filtered on duration, so only labels need a cap.
        row = self.rows[i]
        wav = eval_row_audio(row)
        return {
            "input_features": self.fe(wav, sampling_rate=SAMPLE_RATE).input_features[0],
            "labels": self.labels(row["text"], row["lang"])[: self.max_label_tokens],
        }


def find_resume_checkpoint(out_dir: Path, hub_repo: str | None) -> Path | None:
    """Newest local checkpoint-N in out_dir, else the last-checkpoint pushed to hub_repo."""
    local = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if local:
        return local[-1]
    if hub_repo:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError

        try:
            snapshot_download(hub_repo, local_dir=str(out_dir), allow_patterns=["last-checkpoint/*"])
        except RepositoryNotFoundError:
            return None
        if (out_dir / "last-checkpoint" / "trainer_state.json").exists():
            return out_dir / "last-checkpoint"
    return None


def memory_stats() -> dict[str, float]:
    """RAM used by this process and its dataloader workers, RAM still free, and GPU peak (GB)."""
    import psutil

    proc = psutil.Process()
    rss = proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True))
    stats = {"ram_used_gb": round(rss / 2**30, 1), "ram_free_gb": round(psutil.virtual_memory().available / 2**30, 1)}
    if torch.cuda.is_available():
        stats["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    return stats


def start_memory_monitor(interval_s: float = 30.0) -> None:
    """Log memory every interval_s from a background thread, from startup to exit.

    The per-step log only starts after the first logging_steps; a run killed for using too
    much RAM before that point would otherwise leave no trace of where the memory went.
    """
    import threading
    import time

    def loop():
        while True:
            log.info("memory: %s", memory_stats())
            time.sleep(interval_s)

    threading.Thread(target=loop, name="memory-monitor", daemon=True).start()


class TimeLimitCallback(TrainerCallback):
    """Save and stop cleanly before a platform's hard runtime limit kills the process.

    Kaggle stops a run at 12 h without warning; anything not yet pushed is lost. Stopping
    at the deadline saves a checkpoint (pushed to the Hub by the Trainer), so the next
    session resumes with --resume.
    """

    def __init__(self, deadline: float):
        self.deadline = deadline  # time.time() value
        self.hit = False

    def on_step_end(self, args, state, control, **kwargs):
        import time

        if time.time() >= self.deadline:
            log.warning("Runtime limit reached at step %d: saving a checkpoint and stopping", state.global_step)
            self.hit = True
            control.should_save = True
            control.should_training_stop = True
        return control


class PeriodicSaveCallback(TrainerCallback):
    """Save (and so push to the Hub) at least every `every_s` seconds.

    Step-based saves come every ~4.5 h on a T4. If the platform kills the run in between
    (Kaggle's weekly GPU quota running out, a disconnect), everything since the last save
    is lost; this caps the loss at `every_s`.
    """

    def __init__(self, every_s: float):
        import time

        self.every_s = every_s
        self.last = time.time()

    def on_save(self, args, state, control, **kwargs):
        import time

        self.last = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        import time

        if time.time() - self.last >= self.every_s:
            log.info("%.1f h since the last checkpoint: saving at step %d", self.every_s / 3600, state.global_step)
            control.should_save = True
        return control


class WhisperCollator:
    """Stacks log-mel features and pads labels with -100 (ignored by the loss)."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = torch.tensor(np.stack([np.asarray(f["input_features"], dtype=np.float32) for f in features]))
        max_len = max(len(f["labels"]) for f in features)
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        for i, f in enumerate(features):
            labels[i, : len(f["labels"])] = torch.tensor(f["labels"], dtype=torch.long)
        return {"input_features": input_features, "labels": labels}


class WhisperWERTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer that also reports WER/CER per (corpus, language) at every evaluation.

    Eval loss alone can keep improving while transcripts stop improving; WER is what we
    select checkpoints on (metric_for_best_model: wer).
    """

    def __init__(
        self,
        *args,
        wer_sets: dict[tuple[str, Language], Dataset] | None = None,
        wer_processor: WhisperProcessor | None = None,
        wer_batch_size: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.wer_sets = wer_sets or {}
        self.wer_processor = wer_processor
        self.wer_batch_size = wer_batch_size
        self._timing = {"data_s": 0.0, "steps": 0, "since": None}

    def get_batch_samples(self, epoch_iterator, num_batches, *args, **kwargs):
        # Time spent waiting for the data stream, so a slow step can be pinned on data or GPU.
        import time

        t = time.time()
        if self._timing["since"] is None:
            self._timing["since"] = t
        out = super().get_batch_samples(epoch_iterator, num_batches, *args, **kwargs)
        self._timing["data_s"] += time.time() - t
        self._timing["steps"] += 1
        return out

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        # Memory next to the loss: a run killed by the OS (exit code -9) otherwise leaves no clue.
        # sec_per_step vs data_wait_s: if they are close, the data stream is the bottleneck.
        import time

        logs = {**logs, **memory_stats()}
        tm = self._timing
        if "loss" in logs and tm["steps"] and tm["since"] is not None:
            logs["sec_per_step"] = round((time.time() - tm["since"]) / tm["steps"], 1)
            logs["data_wait_s"] = round(tm["data_s"] / tm["steps"], 1)
            self._timing = {"data_s": 0.0, "steps": 0, "since": None}
        super().log(logs, *args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **gen_kwargs):
        metrics = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        if self.wer_sets:
            wer_metrics = self._wer_metrics(metric_key_prefix)
            self.log(wer_metrics)
            metrics.update(wer_metrics)
        return metrics

    @torch.inference_mode()
    def _wer_metrics(self, prefix: str) -> dict[str, float]:
        model = self.model
        was_training = model.training
        model.eval()
        device = self.args.device
        amp_dtype = torch.bfloat16 if self.args.bf16 else (torch.float16 if self.args.fp16 else None)
        fe = self.wer_processor.feature_extractor
        all_refs, all_hyps, out, records = [], [], {}, []
        for (corpus, lang), rows in self.wer_sets.items():
            refs, hyps = [], []
            for start in range(0, len(rows), self.wer_batch_size):
                batch = rows[start : start + self.wer_batch_size]
                wavs = [eval_row_audio({"flac": b}) for b in batch["flac"]]
                feats = fe(wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    ids = model.generate(
                        input_features=feats.input_features.to(device),
                        attention_mask=feats.attention_mask.to(device),
                        language=lang.whisper_token,
                        task="transcribe",
                        max_new_tokens=225,
                    )
                hyps += self.wer_processor.batch_decode(ids, skip_special_tokens=True)
                refs += batch["text"]
            s = score(refs, hyps)
            out[f"{prefix}_wer_{corpus}_{lang.code}"] = round(s["wer"], 4)
            records += [{"corpus": corpus, "lang": lang.code, "ref": r, "hyp": h} for r, h in zip(refs, hyps)]
            if s["wer"] > 0.6:
                self._explain_poor_set(corpus, lang.code, s["wer"], refs, hyps)
            all_refs += refs
            all_hyps += hyps
        overall = score(all_refs, all_hyps)
        out[f"{prefix}_wer"] = round(overall["wer"], 4)
        out[f"{prefix}_cer"] = round(overall["cer"], 4)
        self._save_predictions(records)
        if was_training:
            model.train()
        return out

    @staticmethod
    def _explain_poor_set(corpus: str, lang: str, wer: float, refs: list[str], hyps: list[str]) -> None:
        """Log what goes wrong on a poor set: WER without diacritics (a spelling-convention
        mismatch shows up as a big drop) and a few reference/output pairs."""
        folded = score([fold_diacritics(r) for r in refs], [fold_diacritics(h) for h in hyps])["wer"]
        log.info("%s/%s: WER %.1f%%, %.1f%% ignoring diacritics. Examples (reference => model):",
                 corpus, lang, 100 * wer, 100 * folded)
        pairs = [(normalize_for_scoring(r), normalize_for_scoring(h)) for r, h in zip(refs, hyps)]
        for r, h in [p for p in pairs if p[0] != p[1]][:4]:
            log.info("    %s  =>  %s", r, h)

    def _save_predictions(self, records: list[dict]) -> None:
        """Every reference/output pair of this evaluation: to output_dir/eval_predictions and,
        when pushing to the Hub, to eval_predictions/ in the model repo. Never fatal."""
        name = f"step-{self.state.global_step:05d}.jsonl"
        try:
            path = Path(self.args.output_dir) / "eval_predictions" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
            if self.args.push_to_hub and self.args.hub_model_id:
                from huggingface_hub import HfApi

                HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=f"eval_predictions/{name}",
                                    repo_id=self.args.hub_model_id, commit_message=f"Eval predictions {name}")
        except Exception as e:
            log.warning("Could not save eval predictions (%s)", e)


def fold_diacritics(text: str) -> str:
    """Strip combining marks (ḓ -> d, š -> s). For diagnosis only: in these languages
    diacritics distinguish words, so scoring keeps them."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if unicodedata.category(c) != "Mn"))
