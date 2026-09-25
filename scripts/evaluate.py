"""Score one or more ASR models on cached evaluation sets.

    python scripts/evaluate.py --config configs/eval_baseline.yaml
    python scripts/evaluate.py --config configs/eval_baseline.yaml --models whisper-large-v3-turbo --langs zul xho

Outputs (under output_dir):
    predictions/<model>/<corpus>_<lang>.jsonl   one line per utterance
    scores.csv                                  one row per model x corpus x language
    summary.md                                  WER table, models as columns

Finished (model, corpus, language) combinations are skipped on rerun, so an interrupted
run can simply be restarted.

On machines without persistent disk (Colab without Drive), add --sync-repo <user>/<name>:
the eval-set cache and results are pulled from that private HF dataset repo at start, and
each new eval set and each finished result is pushed as soon as it exists.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import pandas as pd
from transformers.utils import logging as transformers_logging

from sa_asr.backends import build_transcriber
from sa_asr.config import PROJECT_ROOT, load_languages, load_yaml, select_languages
from sa_asr.data import CORPORA, build_eval_set, check_access, corpus_languages, eval_row_audio, eval_set_path
from sa_asr.hub import HubSync
from sa_asr.metrics import score

log = logging.getLogger("evaluate")


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", help="Only these model names from the config")
    ap.add_argument("--langs", nargs="*", help="Only these language codes")
    ap.add_argument("--corpora", nargs="*", help="Only these corpora")
    ap.add_argument("--max-samples", type=int, help="Override max_samples per eval set")
    ap.add_argument("--device", help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--sync-repo", help="Private HF dataset repo to pull/push the eval cache and results")
    return ap.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    transformers_logging.set_verbosity_error()  # per-batch generation warnings flood the output
    args = parse_args()
    cfg = load_yaml(args.config)
    languages = load_languages()
    out_dir = resolve(cfg["output_dir"])
    cache_dir = resolve(cfg.get("cache_dir", "data/eval_cache"))
    split = cfg.get("split", "test")
    max_samples = args.max_samples if args.max_samples is not None else cfg.get("max_samples", 200)
    batch_size = cfg.get("batch_size", 16)
    sync = HubSync(args.sync_repo) if args.sync_repo else None
    if sync:
        sync.pull(out_dir, cache_dir)

    # 1. Build (or load) the evaluation sets. Each set has an id that encodes every
    #    sampling setting; prediction files carry the same id, so results from a quick
    #    --max-samples run are never mistaken for a full run.
    seed = cfg.get("seed", 42)
    max_dur = cfg.get("max_duration_s", 30.0)
    buffer = cfg.get("shuffle_buffer", 500)
    eval_sets = {}  # (corpus, lang) -> (set_id, Dataset)
    for corpus, codes in cfg["eval_sets"].items():
        if args.corpora and corpus not in args.corpora:
            continue
        if corpus not in CORPORA:
            raise ValueError(f"Unknown corpus {corpus!r}; known: {sorted(CORPORA)}")
        if not check_access(corpus):
            log.warning("Skipping %s: no access", corpus)
            continue
        for lang in corpus_languages(corpus, select_languages(languages, codes)):
            if args.langs and lang.code not in args.langs:
                continue
            set_path = eval_set_path(cache_dir, corpus, lang.code, split, max_samples, seed, max_dur, buffer)
            is_new = not (set_path / "dataset_info.json").exists()
            ds = build_eval_set(
                corpus, lang, split, cache_dir,
                max_samples=max_samples, seed=seed, max_duration_s=max_dur, shuffle_buffer=buffer,
            )
            if sync and is_new:
                sync.push_folder(set_path)
            eval_sets[(corpus, lang.code)] = (set_path.name, ds)
    log.info("Eval sets: %s", {f"{c}/{l}": len(d) for (c, l), (_, d) in eval_sets.items()})

    # 2. Run each model over every set it supports.
    for spec in cfg["models"]:
        name = spec["name"]
        if args.models and name not in args.models:
            continue
        pred_dir = out_dir / "predictions" / name
        todo = [
            key
            for key, (set_id, _) in eval_sets.items()
            if not (pred_dir / f"{set_id}.jsonl").exists() and not (pred_dir / f"{set_id}.unsupported").exists()
        ]
        if not todo:
            log.info("[%s] all sets already done", name)
            continue
        log.info("[%s] loading", name)
        spec = dict(spec)
        if (PROJECT_ROOT / spec["model"]).exists():  # local model dirs are relative to the project
            spec["model"] = str(PROJECT_ROOT / spec["model"])
        model = build_transcriber(spec, device=args.device)
        pred_dir.mkdir(parents=True, exist_ok=True)
        for corpus, code in todo:
            lang = languages[code]
            set_id, rows = eval_sets[(corpus, code)]
            if not model.supports(lang):
                log.info("[%s] does not support %s, skipping", name, code)
                (pred_dir / f"{set_id}.unsupported").touch()
                if sync:
                    sync.push_files(pred_dir / f"{set_id}.unsupported")
                continue
            t0 = time.time()
            records = []
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                wavs = [eval_row_audio({"flac": b}) for b in batch["flac"]]
                hyps = model.transcribe(wavs, lang)
                for utt_id, ref, hyp, dur in zip(batch["utt_id"], batch["text"], hyps, batch["duration"]):
                    records.append({"utt_id": utt_id, "reference": ref, "hypothesis": hyp, "duration": dur})
            elapsed = time.time() - t0
            s = score([r["reference"] for r in records], [r["hypothesis"] for r in records])
            rtf = elapsed / max(sum(r["duration"] for r in records), 1e-6)
            log.info(
                "[%s] %s/%s  WER %.1f%%  CER %.1f%%  (%d utts, RTF %.3f)",
                name, corpus, code, 100 * s["wer"], 100 * s["cer"], s["n_utts"], rtf,
            )
            tmp = pred_dir / f"{set_id}.jsonl.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.write(json.dumps({"_score": s, "rtf": rtf, "corpus": corpus, "lang": code}) + "\n")
            tmp.replace(pred_dir / f"{set_id}.jsonl")
            if sync:
                sync.push_files(pred_dir / f"{set_id}.jsonl")
        model.close()
        del model
        gc.collect()

    write_reports(out_dir, cfg, eval_sets)
    if sync and (out_dir / "summary.md").exists():
        sync.push_files(out_dir / "summary.md", out_dir / "scores.csv")


def write_reports(out_dir: Path, cfg: dict, eval_sets: dict):
    """Rebuild scores.csv and summary.md for this run's eval sets, across all models on disk."""
    languages = load_languages()
    rows = []
    for (corpus, code), (set_id, _) in eval_sets.items():
        for f in sorted((out_dir / "predictions").glob(f"*/{set_id}.jsonl")):
            summary = json.loads(f.read_text(encoding="utf-8").strip().splitlines()[-1])
            rows.append(
                {
                    "model": f.parent.name,
                    "corpus": corpus,
                    "lang": code,
                    "language": languages[code].name,
                    "eval_set": set_id,
                    **summary["_score"],
                    "rtf": summary.get("rtf"),
                }
            )
    if not rows:
        log.warning("No predictions yet")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "scores.csv", index=False)

    order = [m["name"] for m in cfg["models"] if m["name"] in set(df["model"])]
    order += sorted(set(df["model"]) - set(order))
    wer = df.pivot_table(index=["corpus", "language"], columns="model", values="wer")[order] * 100
    cer = df.pivot_table(index=["corpus", "language"], columns="model", values="cer")[order] * 100
    n = df.groupby(["corpus", "language"])["n_utts"].max()
    wer.insert(0, "n", n)
    cer.insert(0, "n", n)
    md = [
        f"# ASR evaluation: {out_dir.name}",
        "",
        "Scores in %, lower is better. Text is lowercased and punctuation-stripped before scoring; "
        "diacritics are kept. Empty cells: model does not support that language.",
        "",
        "## WER",
        "",
        wer.round(1).reset_index().to_markdown(index=False),
        "",
        "## CER",
        "",
        cer.round(1).reset_index().to_markdown(index=False),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    log.info("Wrote %s", out_dir / "summary.md")


if __name__ == "__main__":
    main()
