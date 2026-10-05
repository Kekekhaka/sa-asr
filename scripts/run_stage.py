"""Run one pipeline stage with settings chosen for the current GPU (used by the notebooks).

    python scripts/run_stage.py baseline --hf-user <you>

Stages live here rather than in the notebook, so a new stage or a fix only needs a new
code upload, not a new notebook.

    smoke     tests + tiny end-to-end run
    diagnose  RAM and speed of the full 11-language data pipeline (no model), ~8 min
    baseline  score existing models on the test sets
    trial     500 training steps on isiZulu + isiXhosa
    train     full training run (resumes from the last pushed checkpoint)
    compare   score the trained model against the baseline
    report    fetch saved results and rebuild the table
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

import yaml

from sa_asr.config import PROJECT_ROOT

STAGES = ["smoke", "diagnose", "baseline", "trial", "train", "compare", "report"]


def gpu_settings() -> dict:
    import torch

    gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
    if gb > 30:  # A100 / H100
        return {"gpu_gb": gb, "batch": "--batch-size 32 --grad-accum 1", "eval_samples": 200,
                "baseline_models": "whisper-large-v3-turbo whisper-large-v3 mms-1b-all dsfsi-anv-multilingual-turbo"}
    # T4 / L4 / P100: fewer test utterances; skip whisper-large-v3 (slowest, not the model we fine-tune)
    return {"gpu_gb": gb, "batch": "--batch-size 8 --grad-accum 4", "eval_samples": 100,
            "baseline_models": "whisper-large-v3-turbo dsfsi-anv-multilingual-turbo mms-1b-all"}


def commands(stage: str, user: str, s: dict, max_hours: float | None = None) -> list[tuple[str, dict]]:
    results, model_repo = f"{user}/sa-asr-results", f"{user}/sa-asr-whisper-lora"
    py = sys.executable
    if stage == "smoke":
        return [(f"{py} -m pytest -q", {}),
                (f"{py} scripts/evaluate.py --config configs/eval_smoke.yaml", {}),
                (f"{py} scripts/train.py --config configs/train_smoke.yaml", {})]
    if stage == "diagnose":
        return [(f"{py} scripts/diagnose_memory.py --seconds 300 --langs all", {})]  # the full-training mix
    if stage == "baseline":
        return [(f"{py} scripts/evaluate.py --config configs/eval_baseline.yaml --models {s['baseline_models']} "
                 f"--max-samples {s['eval_samples']} --sync-repo {results}", {})]
    if stage == "trial":
        return [(f"{py} scripts/train.py --config configs/train_lora.yaml {s['batch']} --max-steps 500 --langs zul xho "
                 f"--output-dir outputs/trial --hub-repo {user}/sa-asr-trial --resume", {})]
    if stage == "train":
        limit = f" --max-hours {max_hours}" if max_hours else ""  # less GPU quota left than a full session
        return [(f"{py} scripts/train.py --config configs/train_lora.yaml {s['batch']} --hub-repo {model_repo} --resume"
                 f"{limit}", {})]
    if stage == "compare":
        cfg = yaml.safe_load(open(PROJECT_ROOT / "configs/eval_baseline.yaml", encoding="utf-8"))
        cfg["models"].append({"name": "sa-turbo-lora", "type": "whisper", "model": model_repo, "language_mode": "proxy"})
        with open(PROJECT_ROOT / "configs/eval_compare.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return [(f"{py} scripts/evaluate.py --config configs/eval_compare.yaml --models sa-turbo-lora "
                 f"--max-samples {s['eval_samples']} --sync-repo {results}", {})]
    if stage == "report":
        return [(f"{py} scripts/hub_sync.py pull results/baseline --repo {results}", {})]
    raise ValueError(stage)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--hf-user", required=True)
    ap.add_argument("--max-hours", type=float, help="train: save and stop after this many hours (default 11)")
    args = ap.parse_args()

    build = PROJECT_ROOT / "BUILD.txt"
    print(f"Code build: {build.read_text().strip() if build.exists() else 'unknown (no BUILD.txt: an old package)'}",
          flush=True)
    s = gpu_settings()
    print(f"Stage {args.stage}: GPU {s['gpu_gb']:.0f} GB -> batch '{s['batch']}', {s['eval_samples']} test utterances/set",
          flush=True)
    for cmd, env in commands(args.stage, args.hf_user, s, args.max_hours):
        print(f">>> {' '.join(f'{k}={v}' for k, v in env.items())} {cmd}".strip(), flush=True)
        code = subprocess.run(shlex.split(cmd), cwd=PROJECT_ROOT, env={**os.environ, **env}).returncode
        if code != 0:
            if args.stage == "diagnose":  # each measurement is informative on its own; run them all
                print(f"(step ended with exit code {code}; continuing)", flush=True)
                continue
            sys.exit(f"Step failed (exit code {code}): {cmd}")
    print(f"Stage '{args.stage}' finished.", flush=True)


if __name__ == "__main__":
    main()
