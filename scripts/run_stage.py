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

STAGES = ["smoke", "diagnose", "baseline", "trial", "train", "compare", "lid",
          "train_v2", "compare_v2", "lid_v2", "lm_rescore", "natural", "train_v3", "compare_v3", "lid_v3", "natural_v3", "text_lid", "lid_combined", "route_sim", "lid_probe", "report"]


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
    if stage == "train_v2":  # round 2: continue from the first model, with LID loss + more data
        limit = f" --max-hours {max_hours}" if max_hours else ""
        return [(f"{py} scripts/train.py --config configs/train_lora_v2.yaml {s['batch']} "
                 f"--hub-repo {model_repo}-v2 --resume{limit}", {})]
    if stage == "train_v3":  # round 3: more natural speech; continues from round 2
        limit = f" --max-hours {max_hours}" if max_hours else ""
        return [(f"{py} scripts/train.py --config configs/train_lora_v3.yaml {s['batch']} "
                 f"--hub-repo {model_repo}-v3 --resume{limit}", {})]
    if stage == "compare_v2":
        cfg = yaml.safe_load(open(PROJECT_ROOT / "configs/eval_baseline.yaml", encoding="utf-8"))
        cfg["models"].append({"name": "sa-turbo-lora", "type": "whisper", "model": model_repo, "language_mode": "proxy"})
        cfg["models"].append({"name": "sa-turbo-lora-v2", "type": "whisper", "model": f"{model_repo}-v2",
                              "language_mode": "proxy"})
        with open(PROJECT_ROOT / "configs/eval_compare.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return [(f"{py} scripts/evaluate.py --config configs/eval_compare.yaml --models sa-turbo-lora-v2 "
                 f"--max-samples {s['eval_samples']} --sync-repo {results}", {})]
    if stage in ("lid", "lid_v2"):  # language-detection accuracy of the trained model, on the same test clips
        repo, out = (model_repo, "results/lid") if stage == "lid" else (f"{model_repo}-v2", "results/lid_v2")
        return [(f"{py} scripts/evaluate_lid.py --config configs/eval_baseline.yaml --model {repo} "
                 f"--output-dir {out} --max-samples {s['eval_samples']} --sync-repo {results}", {})]
    if stage == "lm_rescore":  # experiment: does MzansiLM (text-only) pick better transcripts from the n-best list?
        return [(f"{py} scripts/lm_rescore.py --config configs/eval_baseline.yaml --model {model_repo}-v2 "
                 f"--max-samples {s['eval_samples']} --sync-repo {results}", {})]
    if stage == "natural":  # natural speech: only the unscripted Swivuriso clips, scored with the models that cover them
        cfg = yaml.safe_load(open(PROJECT_ROOT / "configs/eval_baseline.yaml", encoding="utf-8"))
        cfg["output_dir"] = "results/natural"
        cfg["eval_sets"] = {"swivuriso_unscripted": "all"}
        cfg["models"] = [m for m in cfg["models"] if m["name"] in ("dsfsi-anv-multilingual-turbo", "mms-1b-all")]
        cfg["models"] += [{"name": "sa-turbo-lora", "type": "whisper", "model": model_repo, "language_mode": "proxy"},
                          {"name": "sa-turbo-lora-v2", "type": "whisper", "model": f"{model_repo}-v2",
                           "language_mode": "proxy"}]
        with open(PROJECT_ROOT / "configs/eval_natural.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return [(f"{py} scripts/evaluate.py --config configs/eval_natural.yaml --max-samples {s['eval_samples']} "
                 f"--sync-repo {results}", {})]
    if stage in ("compare_v3", "natural_v3"):  # same tables as before, with the round-3 model added (others are cached)
        cfg = yaml.safe_load(open(PROJECT_ROOT / "configs/eval_baseline.yaml", encoding="utf-8"))
        cfg["models"] += [{"name": "sa-turbo-lora", "type": "whisper", "model": model_repo, "language_mode": "proxy"},
                          {"name": "sa-turbo-lora-v2", "type": "whisper", "model": f"{model_repo}-v2",
                           "language_mode": "proxy"},
                          {"name": "sa-turbo-lora-v3", "type": "whisper", "model": f"{model_repo}-v3",
                           "language_mode": "proxy"}]
        if stage == "natural_v3":
            cfg["output_dir"], cfg["eval_sets"] = "results/natural", {"swivuriso_unscripted": "all"}
            cfg["models"] = [m for m in cfg["models"] if m["name"] in
                             ("dsfsi-anv-multilingual-turbo", "mms-1b-all", "sa-turbo-lora", "sa-turbo-lora-v2", "sa-turbo-lora-v3")]
        with open(PROJECT_ROOT / "configs/eval_compare.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return [(f"{py} scripts/evaluate.py --config configs/eval_compare.yaml --models sa-turbo-lora-v3 "
                 f"--max-samples {s['eval_samples']} --sync-repo {results}", {})]
    if stage == "lid_v3":
        return [(f"{py} scripts/evaluate_lid.py --config configs/eval_baseline.yaml --model {model_repo}-v3 "
                 f"--output-dir results/lid_v3 --max-samples {s['eval_samples']} --sync-repo {results}", {})]
    if stage == "text_lid":  # no GPU needed: bigger spelling model from training transcripts, tested on unseen test clips
        preds = "results/baseline/predictions/sa-turbo-lora-v2"
        return [(f"{py} scripts/hub_sync.py pull results/baseline --repo {results}", {}),
                (f"{py} scripts/collect_texts.py --sync-repo {results}", {}),
                (f"{py} scripts/train_text_lid.py --texts-dir data/lid_texts --predictions {preds} --holdout", {}),
                (f"{py} scripts/train_text_lid.py --texts-dir data/lid_texts --predictions {preds} "
                 f"--output models/text_lid_v2.json.gz", {}),
                (f"{py} scripts/hub_sync.py push models --repo {results}", {})]
    if stage == "lid_combined":  # tune the audio + spelling combination (needs the text_lid stage's collected texts)
        return [(f"{py} scripts/lid_combined.py --config configs/eval_baseline.yaml --model {model_repo}-v3 "
                 f"--max-samples {s['eval_samples']} --sync-repo {results}", {})]
    if stage == "route_sim":  # CPU only: would sending clips to dsfsi-anv (by language and audio bandwidth) beat our model?
        return [(f"{py} scripts/route_sim.py --sync-repo {results}", {})]
    if stage == "lid_probe":  # a classifier on the encoder's features, trained on labelled training clips of all 11 languages
        return [(f"{py} scripts/lid_probe.py --model {model_repo}-v3 --max-samples {s['eval_samples']} "
                 f"--sync-repo {results}", {})]
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
