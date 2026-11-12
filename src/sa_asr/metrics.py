"""WER / CER on normalized text, and bootstrap confidence intervals for comparing models."""

from __future__ import annotations

import jiwer
import numpy as np

from .text import normalize_for_scoring


def utterance_errors(references: list[str], hypotheses: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Word errors and reference words per utterance (normalized as in `score`), so WER = errors.sum() / words.sum().
    Utterances whose reference is empty after normalization count 0 errors and 0 words."""
    errors, words = [], []
    for ref, hyp in zip(references, hypotheses, strict=True):
        r, h = normalize_for_scoring(ref), normalize_for_scoring(hyp)
        n = len(r.split()) if r else 0
        if not n:
            e = 0
        elif not h:
            e = n
        else:
            o = jiwer.process_words(r, h)
            e = o.substitutions + o.deletions + o.insertions
        errors.append(e)
        words.append(n)
    return np.asarray(errors, dtype=float), np.asarray(words, dtype=float)


def bootstrap_wer(
    sets: list[tuple[np.ndarray, ...]], n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> dict[str, list[float] | float]:
    """Bootstrap the mean WER over sets, for one or more models scored on the same utterances (paired).

    `sets` holds one tuple per test set: (words, errors_model_1, errors_model_2, ...), arrays over that set's
    utterances. Each round resamples every set's utterances with replacement (the same draw for all models, so
    differences are paired), computes each set's WER and averages over sets, as the reports do.
    Returns per model the mean WER and its interval, and for every model after the first its difference from the
    first model with the interval and the share of rounds in which it was lower (WER in %).
    """
    rng = np.random.default_rng(seed)
    n_models = len(sets[0]) - 1
    draws = np.zeros((n_boot, n_models))
    point = np.zeros(n_models)
    for words, *errs in sets:
        n = len(words)
        idx = rng.integers(0, n, size=(n_boot, n))
        w = np.maximum(words[idx].sum(axis=1), 1e-9)
        for m, e in enumerate(errs):
            draws[:, m] += 100 * e[idx].sum(axis=1) / w
            point[m] += 100 * e.sum() / max(words.sum(), 1e-9)
    draws /= len(sets)
    point /= len(sets)
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    out: dict[str, list[float] | float] = {
        "wer": point.tolist(),
        "low": np.percentile(draws, lo_q, axis=0).tolist(),
        "high": np.percentile(draws, hi_q, axis=0).tolist(),
    }
    diff = draws[:, 1:] - draws[:, :1]
    out["diff"] = (point[1:] - point[0]).tolist()
    out["diff_low"] = np.percentile(diff, lo_q, axis=0).tolist() if n_models > 1 else []
    out["diff_high"] = np.percentile(diff, hi_q, axis=0).tolist() if n_models > 1 else []
    out["p_better"] = (diff < 0).mean(axis=0).tolist() if n_models > 1 else []
    return out


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
