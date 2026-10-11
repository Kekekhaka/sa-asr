"""Corrections saved from the app (data/feedback/): show them, or push them for training.

    python scripts/feedback.py stats
    python scripts/feedback.py push --repo <user>/sa-asr-feedback   # private HF dataset

push writes all corrections as one Parquet file (training-corpus layout) to the repo, so a
Kaggle fine-tuning run can read them like any other corpus. Recordings may contain other
people's voices: keep the repo private and only save audio you have permission to use.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import sa_asr  # noqa: F401  (applies the corporate-certificate fix before any HF call)
from sa_asr.feedback import FEEDBACK_DIR, export_parquet, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["stats", "push"])
    ap.add_argument("--repo", help="for push: private HF dataset repo, e.g. <user>/sa-asr-feedback")
    args = ap.parse_args()

    stats = summary()
    print(json.dumps(stats, indent=2))
    if args.action == "stats":
        return
    if not args.repo:
        sys.exit("push needs --repo <user>/sa-asr-feedback")
    if not stats["count"]:
        sys.exit(f"No corrections in {FEEDBACK_DIR} yet.")
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "train-00000.parquet"
        n = export_parquet(out)
        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(out), path_in_repo="data/train-00000.parquet", repo_id=args.repo,
                        repo_type="dataset", commit_message=f"{n} corrections")
    print(f"Pushed {n} corrections to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
