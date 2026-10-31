"""Keep project folders in a private Hugging Face repo (for machines without persistent disk).

Paths inside the repo mirror paths inside the project, e.g. results/baseline/... or
data/eval_cache/..., so pulling into the project root restores them in place.
Push failures are logged and skipped: losing one upload must not kill a long run.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import PROJECT_ROOT

log = logging.getLogger(__name__)


class HubSync:
    def __init__(self, repo: str, repo_type: str = "dataset"):
        from huggingface_hub import HfApi

        self.repo = repo
        self.repo_type = repo_type
        self.api = HfApi()
        self.api.create_repo(repo, repo_type=repo_type, private=True, exist_ok=True)

    @staticmethod
    def rel(path: str | Path) -> str:
        p = Path(path)
        p = p if p.is_absolute() else PROJECT_ROOT / p
        return p.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()

    def pull(self, *folders: str | Path, attempts: int = 5) -> None:
        """Retried with growing waits: a brief Hub outage (503) must not end a run. It still fails in the end,
        because carrying on without the saved results would redo finished work and overwrite the summaries."""
        import time

        from huggingface_hub import snapshot_download

        patterns = [f"{self.rel(f)}/**" for f in folders]
        for attempt in range(1, attempts + 1):
            try:
                snapshot_download(self.repo, repo_type=self.repo_type, local_dir=str(PROJECT_ROOT),
                                  allow_patterns=patterns)
                break
            except Exception as e:
                if attempt == attempts:
                    raise
                wait = 30 * attempt
                log.warning("Pull from %s failed (%s); retrying in %d s (%d/%d)", self.repo, e, wait, attempt,
                            attempts)
                time.sleep(wait)
        log.info("Pulled %s from %s", patterns, self.repo)

    def push_folder(self, folder: str | Path) -> None:
        rel = self.rel(folder)
        try:
            self.api.upload_folder(
                repo_id=self.repo, repo_type=self.repo_type, folder_path=str(PROJECT_ROOT / rel),
                path_in_repo=rel, commit_message=f"sync {rel}",
            )
            log.info("Pushed %s -> %s", rel, self.repo)
        except Exception as e:
            log.warning("Could not push %s (%s); will retry on the next push", rel, e)

    def push_files(self, *paths: str | Path) -> None:
        from huggingface_hub import CommitOperationAdd

        rels = [self.rel(p) for p in paths]
        try:
            self.api.create_commit(
                repo_id=self.repo, repo_type=self.repo_type,
                operations=[CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(PROJECT_ROOT / r)) for r in rels],
                commit_message=f"sync {', '.join(Path(r).name for r in rels)}",
            )
            log.info("Pushed %s -> %s", ", ".join(rels), self.repo)
        except Exception as e:
            log.warning("Could not push %s (%s); will retry on the next push", rels, e)
