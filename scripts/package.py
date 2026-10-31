"""Package the project code (no data, outputs or results) for a cloud notebook.

    python scripts/package.py zip                             # -> sa-asr.zip (upload to Kaggle / Colab)
    python scripts/package.py push --repo <user>/sa-asr-code  # -> private HF repo (the Kaggle notebook can pull it)

Run again after every code change so the notebook gets the new version.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
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


def build_id() -> str:
    """Build time + git commit, written into the package as BUILD.txt; run_stage.py prints it,
    so a notebook log shows which code version actually ran."""
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True).stdout.strip()
        rev += "+uncommitted" if dirty else ""
    except Exception:
        rev = "no-git"
    return f"{time.strftime('%Y-%m-%d %H:%M')} {rev}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["zip", "push"])
    ap.add_argument("--repo", help="for push: private HF dataset repo, e.g. <user>/sa-asr-code")
    args = ap.parse_args()
    files = project_files()
    build = build_id()

    if args.action == "zip":
        out = PROJECT_ROOT / "sa-asr.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, "sa-asr/" + p.relative_to(PROJECT_ROOT).as_posix())
            z.writestr("sa-asr/BUILD.txt", build + "\n")
        print(f"Wrote {out} ({len(files)} files, {out.stat().st_size // 1024} KB), build {build}")
    else:
        if not args.repo:
            sys.exit("push needs --repo <user>/sa-asr-code")
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
        local = {p.relative_to(PROJECT_ROOT).as_posix(): p for p in files}
        remote = set(api.list_repo_files(args.repo, repo_type="dataset")) - {".gitattributes"}
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)) for r, p in local.items() if r != ".gitignore"]
        # The Hub skips files the committed .gitignore ignores, and ours ignores BUILD.txt (for git): upload a copy
        # without that line, or the notebook never sees the build.
        gitignore = "".join(line for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
                            .splitlines(keepends=True) if line.strip() != "BUILD.txt")
        ops.append(CommitOperationAdd(path_in_repo=".gitignore", path_or_fileobj=gitignore.encode()))
        ops.append(CommitOperationAdd(path_in_repo="BUILD.txt", path_or_fileobj=(build + "\n").encode()))
        local["BUILD.txt"] = None
        ops += [CommitOperationDelete(path_in_repo=r) for r in remote - set(local)]
        api.create_commit(args.repo, repo_type="dataset", operations=ops, commit_message="update code")
        print(f"Pushed {len(local)} files (build {build}) to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
