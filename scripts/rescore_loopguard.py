"""Re-score saved predictions with the loop guard (text.collapse_repeats), next to the raw scores.

The guard only edits the decoded text, so applying it to the predictions every evaluation already
saved gives exactly what a fresh run with the guard would score; no GPU is needed.

    python scripts/rescore_loopguard.py --sync-repo KekeCode/sa-asr-results \
        --models sa-turbo-lora-v4 sa-turbo-lora-v5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sa_asr.config import PROJECT_ROOT
from sa_asr.hub import HubSync
from sa_asr.metrics import score
from sa_asr.text import collapse_repeats

SUITES = ("baseline", "natural", "sadilar")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["sa-turbo-lora-v4", "sa-turbo-lora-v5"])
    ap.add_argument("--suites", nargs="+", default=list(SUITES))
    ap.add_argument("--sync-repo", help="pull the saved predictions from this HF dataset repo first")
    args = ap.parse_args()

    if args.sync_repo:
        HubSync(args.sync_repo).pull(*(f"results/{s}/predictions" for s in args.suites))
    lines = ["# Loop guard: WER of saved predictions, raw and with repeats cut back", "",
             "| suite | set | " + " | ".join(f"{m} raw | {m} guarded" for m in args.models) + " |",
             "|---|---|" + "---|---|" * len(args.models)]
    totals: dict[tuple[str, str], list[float]] = {}
    for suite in args.suites:
        pred_dir = PROJECT_ROOT / "results" / suite / "predictions"
        sets = sorted({p.name for m in args.models for p in (pred_dir / m).glob("*.jsonl")})
        for name in sets:
            cells = []
            for m in args.models:
                path = pred_dir / m / name
                if not path.exists():
                    cells += ["", ""]
                    continue
                rows = [r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if "reference" in r]
                refs = [r["reference"] for r in rows]
                raw = score(refs, [r["hypothesis"] for r in rows])["wer"] * 100
                guarded = score(refs, [collapse_repeats(r["hypothesis"]) for r in rows])["wer"] * 100
                totals.setdefault((suite, m), []).append(raw)
                totals.setdefault((suite, m + " guarded"), []).append(guarded)
                cells += [f"{raw:.1f}", f"{guarded:.1f}" + (" *" if abs(guarded - raw) >= 0.05 else "")]
            lines.append(f"| {suite} | {name.split('_test_')[0]} | " + " | ".join(cells) + " |")
    lines += ["", "\\* the guard changed this set. Averages (mean over sets each model has):", ""]
    for suite in args.suites:
        lines.append(f"- {suite}: " + ", ".join(
            f"{m} {sum(v) / len(v):.1f} raw, {sum(g) / len(g):.1f} guarded"
            for m in args.models if (v := totals.get((suite, m))) and (g := totals[(suite, m + ' guarded')])))
    out = PROJECT_ROOT / "results" / "loopguard.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
