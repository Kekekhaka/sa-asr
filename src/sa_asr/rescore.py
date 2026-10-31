"""MzansiLM rescoring: Whisper writes its N best candidates, a text-only language model picks among them.

    lm = TextLM.load()                              # uctnlp/mzansilm-125m, ~0.5 GB
    model = RescoringTranscriber(whisper, lm)       # a drop-in for the transcriber in inference.transcribe_audio

A candidate's score is  asr + LAMBDAS[language] * LM log-probability of its sentence-cased text (summed). The
weights were chosen per language on half of round 4's test clips and measured on the other half (lm_v4 stage,
2026-10-31): 22.95 -> 22.10 WER over 38 test sets, no language worse. Tshivenda and Xitsonga got worse with any
weight and English did not change, so they keep Whisper's own best (0): those are transcribed as usual.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import Language

MZANSILM = "uctnlp/mzansilm-125m"
NBEST = 5
LAMBDAS = {"afr": 0.05, "nbl": 0.02, "nso": 0.02, "sot": 0.05, "ssw": 0.05, "tsn": 0.02, "xho": 0.05, "zul": 0.02}


def load_lm(model_id: str = MZANSILM, device: str = "cpu"):
    """MzansiLM needs hidden_size 512 with 9 attention heads (head_dim 56 set explicitly). Transformers
    5 rejects that combination in its config check although the model is fine, so drop that one check."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

    validators = getattr(LlamaConfig, "__class_validators__", None)
    if validators is not None:
        validators[:] = [v for v in validators if v.__name__ != "validate_architecture"]
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "right"
    lm = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    return tok, lm


@torch.inference_mode()
def lm_logprob(tok, lm, texts: list[str], device: str, batch: int = 32) -> list[tuple[float, int]]:
    """(total log-probability, number of scored tokens) of each text, including the end-of-text token."""
    out: list[tuple[float, int]] = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i : i + batch], return_tensors="pt", padding=True).to(device)
        logits = lm(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits[:, :-1].float()
        target, mask = enc.input_ids[:, 1:], enc.attention_mask[:, 1:]
        lp = torch.log_softmax(logits, dim=-1).gather(2, target.unsqueeze(-1)).squeeze(-1) * mask
        out += list(zip(lp.sum(1).tolist(), mask.sum(1).tolist()))
    return out


def sentence_case(text: str) -> str:
    """Capitalise the first letter and end with a full stop: the form MzansiLM saw in web text."""
    t = text.strip()
    if not t:
        return t
    t = t[0].upper() + t[1:]
    return t if t[-1] in ".!?" else t + "."


class TextLM:
    """MzansiLM kept on the CPU and moved to the transcriber's device for each call (ZeroGPU lends a GPU per call)."""

    def __init__(self, tok, lm):
        self.tok, self.lm = tok, lm

    @classmethod
    def load(cls, model_id: str = MZANSILM) -> "TextLM":
        return cls(*load_lm(model_id, "cpu"))

    def scores(self, texts: list[str], device: str) -> list[float]:
        self.lm.to(device)
        return [lp for lp, _ in lm_logprob(self.tok, self.lm, [sentence_case(t) for t in texts], device)]


def pick(cands: list[tuple[str, float]], lm_scores: list[float], lam: float) -> str:
    """The candidate with the best  asr + lam * LM  (candidates as (text, asr score), best-first)."""
    best = max(range(len(cands)), key=lambda i: cands[i][1] + lam * lm_scores[i])
    return cands[best][0]


class RescoringTranscriber:
    """Wraps a WhisperTranscriber: transcribe() rescores the n best per clip in the languages with a weight; all
    else (language detection, proxy tokens, beams setting) is the wrapped model's."""

    def __init__(self, model, lm: TextLM, n: int = NBEST, lambdas: dict[str, float] | None = None):
        self.model, self.lm, self.n = model, lm, n
        self.lambdas = LAMBDAS if lambdas is None else lambdas

    def __getattr__(self, name):
        return getattr(self.model, name)

    def transcribe(self, audios: list[np.ndarray], lang: Language | None) -> list[str]:
        lam = self.lambdas.get(lang.code, 0.0) if lang is not None else 0.0
        if not lam or not audios:
            return self.model.transcribe(audios, lang)
        nbest = self.model.transcribe_nbest(audios, lang, self.n)
        flat = [t for cands in nbest for t, _ in cands]
        scores = iter(self.lm.scores(flat, self.model.device))
        return [pick(cands, [next(scores) for _ in cands], lam) for cands in nbest]
