"""Translate transcripts into English with open text-translation models (run locally).

NLLB-200 (distilled 600M) covers 8 of the 10 non-English languages. It has no Tshivenda or
isiNdebele: Tshivenda uses a dedicated Marian model, and isiNdebele (a Nguni language
close to isiZulu) is translated as isiZulu and flagged as approximate.

Licences: NLLB-200 is CC-BY-NC-4.0 (non-commercial); opus-mt-ve-en is Apache-2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

NLLB = "facebook/nllb-200-distilled-600M"


@dataclass(frozen=True)
class Route:
    model: str
    src_code: str | None  # NLLB source-language code; None for single-pair Marian models
    approximate: bool = False


ROUTES: dict[str, Route] = {
    "afr": Route(NLLB, "afr_Latn"),
    "zul": Route(NLLB, "zul_Latn"),
    "xho": Route(NLLB, "xho_Latn"),
    "nso": Route(NLLB, "nso_Latn"),
    "sot": Route(NLLB, "sot_Latn"),
    "tsn": Route(NLLB, "tsn_Latn"),
    "tso": Route(NLLB, "tso_Latn"),
    "ssw": Route(NLLB, "ssw_Latn"),
    "nbl": Route(NLLB, "zul_Latn", approximate=True),
    "ven": Route("Helsinki-NLP/opus-mt-ve-en", None),
}


def can_translate(lang: str | None) -> bool:
    return lang in ROUTES


@lru_cache(maxsize=2)
def _load(model_id: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()
    return tok, model


def translate(texts: list[str], lang: str, max_new_tokens: int = 400, num_beams: int = 2) -> list[str]:
    """English translations of `texts` (one per input; empty inputs stay empty)."""
    import torch

    route = ROUTES[lang]
    todo = [(i, t.strip()) for i, t in enumerate(texts) if t and t.strip()]
    out = [""] * len(texts)
    if not todo:
        return out
    tok, model = _load(route.model)
    gen_kwargs = {"max_new_tokens": max_new_tokens, "num_beams": num_beams}
    if route.src_code:
        tok.src_lang = route.src_code
        gen_kwargs["forced_bos_token_id"] = tok.convert_tokens_to_ids("eng_Latn")
    for start in range(0, len(todo), 8):
        batch = todo[start : start + 8]
        enc = tok([t for _, t in batch], return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.inference_mode():
            ids = model.generate(**enc, **gen_kwargs)
        for (i, _), text in zip(batch, tok.batch_decode(ids, skip_special_tokens=True)):
            out[i] = text.strip()
    return out
