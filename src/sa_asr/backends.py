"""Speech recognisers behind one interface, so evaluation treats every model the same.

A model spec (from an eval config) looks like:

    name: whisper-large-v3-turbo
    type: whisper | mms
    model: <hub id or local dir>        # a local LoRA adapter dir works too
    processor: <hub id>                 # optional; defaults to `model` (or the adapter's base)
    language_mode: auto | proxy | native
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .audio import SAMPLE_RATE
from .config import Language

log = logging.getLogger(__name__)

META_FILE = "sa_asr_meta.json"


def localize_model(model: str) -> Path:
    """Local folder for models trained by this project, wherever they live.

    A local folder is returned as is. A Hugging Face repo containing sa_asr_meta.json (as
    pushed by train.py --hub-repo) is downloaded, minus its checkpoints. Anything else is
    returned unchanged for from_pretrained to handle.
    """
    path = Path(model)
    if path.exists() or model.count("/") != 1:
        return path
    from huggingface_hub import hf_hub_download, snapshot_download

    try:
        hf_hub_download(model, META_FILE)
    except Exception:
        return path
    return Path(snapshot_download(model, ignore_patterns=["last-checkpoint/*", "checkpoint-*/*"]))


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class Transcriber:
    name: str

    def supports(self, lang: Language) -> bool:
        return True

    def transcribe(self, audios: list[np.ndarray], lang: Language) -> list[str]:
        raise NotImplementedError

    def close(self) -> None:
        """Free GPU memory before the next model is loaded."""
        for attr in ("model",):
            if hasattr(self, attr):
                delattr(self, attr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class WhisperTranscriber(Transcriber):
    """Whisper via transformers.

    language_mode:
      auto    let Whisper detect the language (the fair zero-shot baseline, and the right
              choice for models whose language-token scheme is unknown)
      proxy   force the proxy token from languages.yaml (models trained by this project)
      native  force the real token when Whisper knows the language, else auto
    """

    def __init__(
        self,
        name: str,
        model: str,
        processor: str | None = None,
        language_mode: str = "auto",
        device: str | None = None,
        num_beams: int = 1,
        max_new_tokens: int = 225,
    ):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        if language_mode not in ("auto", "proxy", "native"):
            raise ValueError(f"language_mode must be auto, proxy or native, not {language_mode!r}")
        self.name = name
        self.language_mode = language_mode
        self.device = pick_device(device)
        self.dtype = pick_dtype(self.device)
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

        model_path = localize_model(model)
        if model_path.exists():
            model = str(model_path)
        adapter_cfg = model_path / "adapter_config.json"
        if adapter_cfg.exists():
            from peft import PeftModel

            base_id = json.loads(adapter_cfg.read_text(encoding="utf-8"))["base_model_name_or_path"]
            log.info("[%s] loading base %s + LoRA adapter %s", name, base_id, model)
            base = WhisperForConditionalGeneration.from_pretrained(base_id, dtype=self.dtype)
            self.model = PeftModel.from_pretrained(base, str(model_path)).merge_and_unload()
            default_processor = str(model_path) if (model_path / "preprocessor_config.json").exists() else base_id
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(model, dtype=self.dtype)
            default_processor = model
        self.processor = WhisperProcessor.from_pretrained(processor or default_processor)
        self.model.to(self.device).eval()
        # We always pass max_new_tokens; a stored max_length only triggers a warning per batch.
        self.model.generation_config.max_length = None
        self.proxy_tokens = self._load_meta_tokens(model_path)

    @staticmethod
    def _load_meta_tokens(model_path: Path) -> dict[str, str] | None:
        meta = model_path / META_FILE
        if meta.exists():
            return json.loads(meta.read_text(encoding="utf-8"))["whisper_tokens"]
        return None

    def _language_for(self, lang: Language) -> str | None:
        if self.language_mode == "auto":
            return None
        if self.language_mode == "native":
            return lang.whisper_token if lang.whisper_native else None
        # proxy: the model's own mapping wins over the registry, and they must agree
        if self.proxy_tokens is not None:
            trained = self.proxy_tokens.get(lang.code)
            if trained is None:
                raise ValueError(f"{self.name} was not trained on {lang.code}")
            if trained != lang.whisper_token:
                raise ValueError(
                    f"{self.name} was trained with proxy token '{trained}' for {lang.code}, "
                    f"but languages.yaml says '{lang.whisper_token}'"
                )
        return lang.whisper_token

    def supports(self, lang: Language) -> bool:
        if self.language_mode == "proxy" and self.proxy_tokens is not None:
            return lang.code in self.proxy_tokens
        return True

    @torch.inference_mode()
    def detect_language(self, audio: np.ndarray, candidates: list[Language]) -> dict[str, float]:
        """Probability of each candidate language, from Whisper's first decoding step.

        Only meaningful for models trained by this project (language_mode proxy), where each
        proxy token stands for a South African language; the softmax is restricted to the
        candidates, so the answer is always one of them.
        """
        tok = self.processor.tokenizer
        feats = self.processor.feature_extractor(audio[: 30 * SAMPLE_RATE], sampling_rate=SAMPLE_RATE, return_tensors="pt")
        sot = torch.tensor([[tok.convert_tokens_to_ids("<|startoftranscript|>")]], device=self.device)
        logits = self.model(input_features=feats.input_features.to(self.device, self.dtype), decoder_input_ids=sot).logits
        ids = [tok.convert_tokens_to_ids(f"<|{self._language_for(lang)}|>") for lang in candidates]
        probs = torch.softmax(logits[0, -1, ids].float(), dim=-1).tolist()
        return dict(sorted(zip((lang.code for lang in candidates), probs), key=lambda kv: -kv[1]))

    @torch.inference_mode()
    def transcribe(self, audios: list[np.ndarray], lang: Language | None) -> list[str]:
        """Transcribe clips of up to 30 s. lang=None lets Whisper pick the language."""
        feats = self.processor.feature_extractor(
            audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True
        )
        kwargs: dict[str, Any] = {
            "task": "transcribe",
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "attention_mask": feats.attention_mask.to(self.device),
        }
        language = self._language_for(lang) if lang is not None else None
        if language is not None:
            kwargs["language"] = language
        out = self.model.generate(feats.input_features.to(self.device, self.dtype), **kwargs)
        return [t.strip() for t in self.processor.batch_decode(out, skip_special_tokens=True)]


class MMSTranscriber(Transcriber):
    """Meta MMS (wav2vec2 CTC with per-language adapters)."""

    def __init__(self, name: str, model: str = "facebook/mms-1b-all", device: str | None = None, **_):
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self.name = name
        self.device = pick_device(device)
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = Wav2Vec2ForCTC.from_pretrained(model).to(self.device).eval()
        self.adapters = set(getattr(self.processor.tokenizer, "vocab", {}).keys())
        self._loaded: str | None = None

    def supports(self, lang: Language) -> bool:
        return lang.mms_adapter is not None and (not self.adapters or lang.mms_adapter in self.adapters)

    @torch.inference_mode()
    def transcribe(self, audios: list[np.ndarray], lang: Language) -> list[str]:
        if self._loaded != lang.mms_adapter:
            self.processor.tokenizer.set_target_lang(lang.mms_adapter)
            self.model.load_adapter(lang.mms_adapter)
            self._loaded = lang.mms_adapter
        inputs = self.processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits
        ids = torch.argmax(logits, dim=-1)
        return [t.strip() for t in self.processor.batch_decode(ids)]


BACKENDS = {"whisper": WhisperTranscriber, "mms": MMSTranscriber}


def build_transcriber(spec: dict[str, Any], device: str | None = None) -> Transcriber:
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in BACKENDS:
        raise ValueError(f"Unknown model type {kind!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[kind](device=device, **spec)
