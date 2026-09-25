"""Pieces for fine-tuning Whisper: label building, the training stream, collator, trainer."""

from __future__ import annotations

import logging
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, IterableDataset, interleave_datasets
from transformers import Seq2SeqTrainer, WhisperProcessor

from .audio import SAMPLE_RATE, augment, decode_audio, duration_seconds
from .config import Language
from .data import CORPORA, eval_row_audio, load_corpus
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
) -> IterableDataset:
    """Mix (corpus, language, weight) streams by weight into one endless training stream."""
    parts, weights = [], []
    for corpus, lang, weight in streams:
        parts.append(load_corpus(corpus, lang, "train").shuffle(seed=seed, buffer_size=shuffle_buffer))
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
        all_refs, all_hyps, out = [], [], {}
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
            all_refs += refs
            all_hyps += hyps
        overall = score(all_refs, all_hyps)
        out[f"{prefix}_wer"] = round(overall["wer"], 4)
        out[f"{prefix}_cer"] = round(overall["cer"], 4)
        if was_training:
            model.train()
        return out
