"""Compare models on their saved predictions, with 95% bootstrap confidence intervals. No GPU needed.

    python scripts/compare_models.py --sync-repo KekeCode/sa-asr-results \
        --models sa-turbo-lora-v5 sa-turbo-lora-v4 sa-turbo-lora-v6

The first model is the reference: every other model is reported as its difference from it. All hypotheses go
through the loop guard (text.collapse_repeats), as the app and current evaluations do, so older saved runs
compare fairly. Only utterances every model was scored on are used, so differences are paired.

Each test set has ~100 utterances, so one set's WER moves by a few points from the luck of which clips were
drawn. A difference is marked **real** when its 95% interval excludes zero (the bootstrap resamples each
set's utterances, keeping the pairing). Sets with fewer than --min-utts utterances (e.g. soap_eng, 1 clip)
are listed but left out of the suite averages.

Writes results/compare/summary.md.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np

from sa_asr.config import PROJECT_ROOT
from sa_asr.hub import HubSync
from sa_asr.metrics import bootstrap_wer, utterance_errors
from sa_asr.text import collapse_repeats

SUITES = ("baseline", "natural", "sadilar")


def load_predictions(path: Path) -> dict[str, tuple[str, str]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "reference" in r:
            out[str(r["utt_id"])] = (r["reference"], collapse_repeats(r["hypothesis"]))
    return out


def collect(pred_dir: Path, models: list[str]) -> dict[str, tuple[np.ndarray, ...]]:
    """{set name: (words, errors per model...)} on the utterances all models share."""
    sets = {}
    names = sorted({p.name for m in models for p in (pred_dir / m).glob("*.jsonl")})
    for name in names:
        paths = [pred_dir / m / name for m in models]
        if not all(p.exists() for p in paths):
            continue
        preds = [load_predictions(p) for p in paths]
        ids = sorted(set.intersection(*(set(p) for p in preds)))
        if not ids:
            continue
        refs = [preds[0][i][0] for i in ids]
        words, errs = None, []
        for p in preds:
            e, w = utterance_errors(refs, [p[i][1] for i in ids])
            words, errs = w, errs + [e]
        sets[name.split("_test_")[0]] = (words, *errs)
    return sets


def fmt_diff(d: float, lo: float, hi: float) -> str:
    real = hi < 0 or lo > 0
    s = f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]"
    return f"**{s}**" if real else s


def report(suite: str, sets: dict, models: list[str], min_utts: int, n_boot: int) -> list[str]:
    ref, others = models[0], models[1:]
    lines = [f"## {suite}", "",
             "| set | utts | " + f"{ref} WER [95%]" + "".join(f" | {m} WER | {m} vs {ref}" for m in others) + " |",
             "|---|---|---|" + "---|---|" * len(others)]
    kept, per_set = [], {}
    for name, arrays in sets.items():
        n = len(arrays[0])
        r = per_set[name] = bootstrap_wer([arrays], n_boot=n_boot, seed=zlib.crc32(name.encode()))
        cells = [f"{r['wer'][0]:.1f} [{r['low'][0]:.1f}, {r['high'][0]:.1f}]"]
        for k in range(len(others)):
            cells += [f"{r['wer'][k + 1]:.1f}", fmt_diff(r["diff"][k], r["diff_low"][k], r["diff_high"][k])]
        small = n < min_utts
        lines.append(f"| {name}{' (too small, not averaged)' if small else ''} | {n} | " + " | ".join(cells) + " |")
        if not small:
            kept.append(arrays)
    if not kept:
        return lines + [""]
    r = bootstrap_wer(kept, n_boot=n_boot, seed=1)
    lines += ["", f"**Average over {len(kept)} sets:**", ""]
    lines.append(f"- {ref}: {r['wer'][0]:.1f} [{r['low'][0]:.1f}, {r['high'][0]:.1f}]")
    for k, m in enumerate(others):
        lines.append(f"- {m}: {r['wer'][k + 1]:.1f} [{r['low'][k + 1]:.1f}, {r['high'][k + 1]:.1f}], "
                     f"vs {ref} {fmt_diff(r['diff'][k], r['diff_low'][k], r['diff_high'][k])} "
                     f"(lower in {100 * r['p_better'][k]:.0f}% of resamples)")
    real_better = [[], []]
    for name, arrays in sets.items():
        if len(arrays[0]) < min_utts:
            continue
        rr = per_set[name]
        for k, m in enumerate(others):
            if rr["diff_high"][k] < 0:
                real_better[0].append(f"{name} ({m})")
            elif rr["diff_low"][k] > 0:
                real_better[1].append(f"{name} ({m})")
    lines += [f"- real gains vs {ref}: {', '.join(real_better[0]) or 'none'}",
              f"- real losses vs {ref}: {', '.join(real_better[1]) or 'none'}", ""]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="the first one is the reference")
    ap.add_argument("--suites", nargs="+", default=list(SUITES))
    ap.add_argument("--min-utts", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output", default="results/compare/summary.md")
    ap.add_argument("--sync-repo", help="pull the saved predictions from this HF dataset repo first")
    args = ap.parse_args()

    if args.sync_repo:
        HubSync(args.sync_repo).pull(*(f"results/{s}/predictions" for s in args.suites))
    lines = [f"# Model comparison with 95% confidence intervals (reference: {args.models[0]})", "",
             "WER % with the loop guard on every model. `[low, high]` is the 95% bootstrap interval; a difference in "
             "**bold** is real (its interval excludes zero), the others are within the noise of ~100 test clips.", ""]
    for suite in args.suites:
        sets = collect(PROJECT_ROOT / "results" / suite / "predictions", args.models)
        if sets:
            lines += report(suite, sets, args.models, args.min_utts, args.n_boot)
    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
