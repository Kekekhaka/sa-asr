"""Copy a project folder to or from a private Hugging Face repo.

For environments without persistent disk (Colab without Google Drive): push results after
a run, pull them back into a fresh session so finished work is skipped.

    python scripts/hub_sync.py push results/baseline --repo <user>/sa-asr-results
    python scripts/hub_sync.py pull results/baseline --repo <user>/sa-asr-results

The folder keeps its project-relative path inside the repo. Repos are created private.
"""

from __future__ import annotations

import argparse
import logging

from sa_asr.config import PROJECT_ROOT
from sa_asr.hub import HubSync

log = logging.getLogger("hub_sync")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("folder", help="Project-relative folder, e.g. results/baseline")
    ap.add_argument("--repo", required=True, help="<user>/<name>")
    ap.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    args = ap.parse_args()

    sync = HubSync(args.repo, repo_type=args.repo_type)
    if args.action == "push":
        if not (PROJECT_ROOT / args.folder).exists():
            raise SystemExit(f"{args.folder} does not exist")
        sync.push_folder(args.folder)
    else:
        sync.pull(args.folder)


if __name__ == "__main__":
    main()
