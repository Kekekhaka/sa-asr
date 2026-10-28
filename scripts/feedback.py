"""Corrections saved from the app (data/feedback/): show them, or push them for training.

    python scripts/feedback.py stats
    python scripts/feedback.py report                                 # how the model did on your corrections
    python scripts/feedback.py push --repo <user>/sa-asr-feedback     # private HF dataset
    python scripts/feedback.py pull --repo <user>/sa-asr-feedback     # corrections saved in the Space -> here

push writes the corrections as Parquet files (training-corpus layout): data/train-00000.parquet, which a
Kaggle fine-tuning run can read like any other corpus, and test/test-00000.parquet, a fixed ~30% that must
never be trained on (it stays a fair test of the model on your own recordings). Recordings may contain other
people's voices: keep the repo private and only save audio you have permission to use.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import sa_asr  # noqa: F401  (applies the corporate-certificate fix before any HF call)
from sa_asr.feedback import FEEDBACK_DIR, export_parquet, pull, report, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["stats", "report", "push", "pull"])
    ap.add_argument("--repo", help="for push/pull: private HF dataset repo, e.g. <user>/sa-asr-feedback")
    args = ap.parse_args()

    if args.action == "pull":
        if not args.repo:
            sys.exit("pull needs --repo <user>/sa-asr-feedback")
        print(f"Added {pull(args.repo)} new corrections from {args.repo}")
    stats = summary()
    print(json.dumps(stats, indent=2))
    if args.action in ("stats", "pull"):
        return
    if args.action == "report":
        print(json.dumps(report(), indent=2, ensure_ascii=False))
        return
    if not args.repo:
        sys.exit("push needs --repo <user>/sa-asr-feedback")
    if not stats["count"]:
        sys.exit(f"No corrections in {FEEDBACK_DIR} yet.")
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
        counts = {}
        for split, path in (("train", "data/train-00000.parquet"), ("test", "test/test-00000.parquet")):
            out = Path(tmp) / f"{split}.parquet"
            counts[split] = export_parquet(out, split=split)
            if counts[split]:
                api.upload_file(path_or_fileobj=str(out), path_in_repo=path, repo_id=args.repo, repo_type="dataset",
                                commit_message=f"{counts[split]} {split} corrections")
    print(f"Pushed {counts['train']} train and {counts['test']} test corrections to "
          f"https://huggingface.co/datasets/{args.repo} (private)")


if __name__ == "__main__":
    main()
