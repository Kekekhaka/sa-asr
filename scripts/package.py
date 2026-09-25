"""Package the project code (no data, outputs or results) for a cloud notebook.

    python scripts/package.py zip                             # -> sa-asr.zip (upload to Kaggle / Colab)
    python scripts/package.py push --repo <user>/sa-asr-code  # -> private HF repo (the Kaggle notebook can pull it)

Run again after every code change so the notebook gets the new version.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import sa_asr  # noqa: F401  (applies the corporate-certificate fix before any HF call)
from sa_asr.config import PROJECT_ROOT

TOP_FILES = ["pyproject.toml", "README.md", ".gitignore"]
DIRS = ["app", "configs", "notebooks", "scripts", "src", "tests"]
SKIP_PARTS = {"__pycache__"}


def project_files() -> list[Path]:
    files = [PROJECT_ROOT / f for f in TOP_FILES]
    for d in DIRS:
        for p in sorted((PROJECT_ROOT / d).rglob("*")):
            if p.is_file() and not SKIP_PARTS & set(p.parts) and not any(x.endswith(".egg-info") for x in p.parts):
                files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["zip", "push"])
    ap.add_argument("--repo", help="for push: private HF dataset repo, e.g. <user>/sa-asr-code")
    args = ap.parse_args()
    files = project_files()

    if args.action == "zip":
        out = PROJECT_ROOT / "sa-asr.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, "sa-asr/" + p.relative_to(PROJECT_ROOT).as_posix())
        print(f"Wrote {out} ({len(files)} files, {out.stat().st_size // 1024} KB)")
    else:
        if not args.repo:
            sys.exit("push needs --repo <user>/sa-asr-code")
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
        local = {p.relative_to(PROJECT_ROOT).as_posix(): p for p in files}
        remote = set(api.list_repo_files(args.repo, repo_type="dataset")) - {".gitattributes"}
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)) for r, p in local.items()]
        ops += [CommitOperationDelete(path_in_repo=r) for r in remote - set(local)]
        api.create_commit(args.repo, repo_type="dataset", operations=ops, commit_message="update code")
        print(f"Pushed {len(local)} files to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
