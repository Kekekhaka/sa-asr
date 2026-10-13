"""Measure how well a trained model identifies the spoken language.

    python scripts/evaluate_lid.py --config configs/eval_baseline.yaml --model KekeCode/sa-asr-whisper-lora
    python scripts/evaluate_lid.py ... --crops 3 5 --sync-repo <user>/sa-asr-results

The WER evaluation always tells the model the right language, so it says nothing about
detection. This runs the model's language detection on the same cached test sets, on each
whole clip and on its first N seconds (--crops), and writes:

    results/lid/predictions.jsonl   one line per clip and condition (top-3 probabilities)
    results/lid/summary.md          accuracy, family accuracy, confusion matrix, top mistakes
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from transformers.utils import logging as transformers_logging

from sa_asr.audio import SAMPLE_RATE
from sa_asr.config import PROJECT_ROOT, load_languages, load_yaml, quiet_progress_bars, select_languages
from sa_asr.data import build_eval_set, corpus_languages, eval_row_audio
from sa_asr.hub import HubSync
from sa_asr.inference import load_transcriber, resolve_model
from sa_asr.lid import accuracy, report_markdown

log = logging.getLogger("evaluate_lid")


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Eval config (eval sets, cache dir), e.g. configs/eval_baseline.yaml")
    ap.add_argument("--model", default="KekeCode/sa-asr-whisper-lora", help="Trained model: local folder or HF repo")
    ap.add_argument("--max-samples", type=int, help="Override max_samples per eval set")
    ap.add_argument("--crops", type=float, nargs="*", default=[3.0, 5.0], help="Also test the first N seconds")
    ap.add_argument("--output-dir", default="results/lid")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--sync-repo", help="Private HF dataset repo: pull the eval cache, push the report")
    return ap.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    transformers_logging.set_verbosity_error()
    quiet_progress_bars()
    args = parse_args()
    cfg = load_yaml(args.config)
    languages = load_languages()
    out_dir = resolve(args.output_dir)
    cache_dir = resolve(cfg.get("cache_dir", "data/eval_cache"))
    max_samples = args.max_samples if args.max_samples is not None else cfg.get("max_samples", 200)
    seed, max_dur, buffer = cfg.get("seed", 42), cfg.get("max_duration_s", 30.0), cfg.get("shuffle_buffer", 500)
    sync = HubSync(args.sync_repo) if args.sync_repo else None
    if sync:
        sync.pull(cache_dir)

    model_ref, trained = resolve_model(args.model)
    if trained is None:
        raise SystemExit(f"{args.model} is not a model trained by this project (no language mapping)")
    model = load_transcriber(model_ref, finetuned=True)
    candidates = [languages[c] for c in model.proxy_tokens if c in languages]
    log.info("Model %s, candidate languages %s", args.model, [c.code for c in candidates])

    records: list[dict] = []
    conds = [("full", None)] + [(f"{c:g}s", c) for c in args.crops]
    for corpus, codes in cfg["eval_sets"].items():
        for lang in corpus_languages(corpus, select_languages(languages, codes)):
            if lang.code not in trained:
                continue
            rows = build_eval_set(  # the cached set (same clips as the WER evaluation), else builtcorpus, lang, cfg.get("split", "test"), cache_dir, max_samples=max_samples,
                                  seed=seed, max_duration_s=max_dur, shuffle_buffer=buffer)
            t0 = time.time()
            set_recs = []
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                wavs = [eval_row_audio({"flac": b}) for b in batch["flac"]]
                for cond, crop in conds:
                    clips = wavs if crop is None else [w[: int(crop * SAMPLE_RATE)] for w in wavs]
                    for utt_id, dur, probs in zip(batch["utt_id"], batch["duration"], model.detect_languages(clips, candidates)):
                        top = list(probs.items())[:3]
                        set_recs.append({"corpus": corpus, "lang": lang.code, "cond": cond, "utt_id": utt_id,
                                         "duration": dur, "pred": top[0][0], "top3": top})
            records += set_recs
            acc = {c: accuracy([r for r in set_recs if r["cond"] == c])["accuracy"] for c, _ in conds}
            log.info("%s/%s (%d clips, %.0f s): %s", corpus, lang.code, len(rows), time.time() - t0,
                     "  ".join(f"{c} {a:.0%}" for c, a in acc.items()))

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    md = report_markdown(records, {c: l.name for c, l in languages.items()},
                         f"Language identification: {Path(args.model).name}")
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    if sync:
        sync.push_files(out_dir / "summary.md", out_dir / "predictions.jsonl")


if __name__ == "__main__":
    main()
