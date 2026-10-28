"""WER / CER on normalized text."""

from __future__ import annotations

import jiwer

from .text import normalize_for_scoring


def score(references: list[str], hypotheses: list[str]) -> dict[str, float | int]:
    """Corpus-level WER and CER (total errors / total reference words or characters).

    Pairs whose reference is empty after normalization are dropped; they cannot be scored.
    """
    refs, hyps = [], []
    for ref, hyp in zip(references, hypotheses, strict=True):
        ref_n = normalize_for_scoring(ref)
        if ref_n:
            refs.append(ref_n)
            hyps.append(normalize_for_scoring(hyp))
    if not refs:
        return {"n_utts": 0, "n_ref_words": 0, "wer": float("nan"), "cer": float("nan"), "cer_nospace": float("nan")}
    return {
        "n_utts": len(refs),
        "n_ref_words": sum(len(r.split()) for r in refs),
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
        # spaces ignored: where words are split or joined differently (Sotho-Tswana writes verbs disjunctively)
        # WER moves but this does not, so WER far above it points to spelling convention, not wrong sounds
        "cer_nospace": jiwer.cer([r.replace(" ", "") for r in refs], [h.replace(" ", "") for h in hyps]),
    }
