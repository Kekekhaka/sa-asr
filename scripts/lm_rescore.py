"""Experiment: can MzansiLM (a text-only language model) pick better transcripts?

    python scripts/lm_rescore.py --config configs/eval_baseline.yaml --model KekeCode/sa-asr-whisper-lora-v2 \
        --max-samples 100 --sync-repo <user>/sa-asr-results
    python scripts/lm_rescore.py --analyze-only --sync-repo <user>/sa-asr-results     # re-tune, no GPU

Step 1 (GPU): for every cached test clip, Whisper produces its N best candidate transcripts (beam
search, told the correct language) and MzansiLM scores each candidate's text (log-probability).
Step 2 (CPU): the candidate with the best  asr_score + lambda * lm_score  is chosen. Lambda is
tuned on half of each set's utterances (even positions) and measured on the other half (odd), so
the reported gain is not tuned on the clips it is measured on. Also reported: the oracle (best
candidate per utterance), which is the most any rescoring could gain.

Writes results/lm_rescore/nbest/<corpus>_<lang>.jsonl (one line per utterance, all candidates)
and results/lm_rescore/summary.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import jiwer
import torch
from transformers.utils import logging as transformers_logging

from sa_asr.audio import SAMPLE_RATE  # noqa: F401  (kept for parity with the other scripts)
from sa_asr.config import PROJECT_ROOT, load_languages, load_yaml, quiet_progress_bars, select_languages
from sa_asr.data import build_eval_set, corpus_languages, eval_row_audio
from sa_asr.hub import HubSync
from sa_asr.inference import load_transcriber, resolve_model
from sa_asr.rescore import lm_logprob, load_lm, sentence_case  # noqa: F401  (also imported from here by lm_tiebreak)
from sa_asr.text import normalize_for_scoring

log = logging.getLogger("lm_rescore")
LAMBDAS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def word_errors(ref: str, hyp: str) -> int:
    r, h = normalize_for_scoring(ref), normalize_for_scoring(hyp)
    if not r:
        return 0
    if not h:
        return len(r.split())
    o = jiwer.process_words(r, h)
    return o.substitutions + o.deletions + o.insertions


def generate(args, languages, cache_dir: Path, out_dir: Path, sync: HubSync | None) -> None:
    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ref, trained = resolve_model(args.model)
    whisper = load_transcriber(model_ref, finetuned=True)
    tok, lm = load_lm(args.lm, device)
    max_samples = args.max_samples if args.max_samples is not None else cfg.get("max_samples", 200)
    seed, max_dur, buffer = cfg.get("seed", 42), cfg.get("max_duration_s", 30.0), cfg.get("shuffle_buffer", 500)
    nbest_dir = out_dir / "nbest"
    nbest_dir.mkdir(parents=True, exist_ok=True)
    for corpus, codes in cfg["eval_sets"].items():
        for lang in corpus_languages(corpus, select_languages(languages, codes)):
            target = nbest_dir / f"{corpus}_{lang.code}.jsonl"
            if lang.code not in trained or target.exists():
                continue
            rows = build_eval_set(corpus, lang, cfg.get("split", "test"), cache_dir, max_samples=max_samples,
                                  seed=seed, max_duration_s=max_dur, shuffle_buffer=buffer)
            t0 = time.time()
            records = []
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                wavs = [eval_row_audio({"flac": b}) for b in batch["flac"]]
                nbest = whisper.transcribe_nbest(wavs, lang, args.beams)
                flat = [text for cands in nbest for text, _ in cands]
                raw, cased = lm_logprob(tok, lm, flat, device), lm_logprob(tok, lm, [sentence_case(t) for t in flat], device)
                k = 0
                for utt_id, ref, dur, cands in zip(batch["utt_id"], batch["text"], batch["duration"], nbest):
                    entries = []
                    for text, asr in cands:
                        entries.append({"text": text, "asr": asr, "lm_raw": raw[k][0], "lm_cased": cased[k][0],
                                        "ntok_raw": raw[k][1], "ntok_cased": cased[k][1],
                                        "err": word_errors(ref, text)})
                        k += 1
                    records.append({"utt_id": utt_id, "reference": ref, "duration": dur,
                                    "ref_words": len(normalize_for_scoring(ref).split()), "cands": entries})
            tmp = target.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
            tmp.replace(target)
            log.info("%s/%s: %d clips x %d candidates in %.0f s", corpus, lang.code, len(records), args.beams,
                     time.time() - t0)
            if sync:
                sync.push_files(target)


def choose(cands: list[dict], lam: float, variant: str, per_token: bool) -> dict:
    def score(c):
        lm = c[f"lm_{variant}"] / (max(c[f"ntok_{variant}"], 1) if per_token else 1.0)
        return c["asr"] + lam * lm
    return max(cands, key=score)


def analyze(out_dir: Path, languages: dict) -> str:
    sets: dict[str, list[dict]] = {}
    for f in sorted((out_dir / "nbest").glob("*.jsonl")):
        sets[f.stem] = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not sets:
        return "No n-best files found."

    def total(rows, pick):
        return sum(pick(r)["err"] for r in rows), sum(r["ref_words"] for r in rows)

    def wer(rows, pick):
        e, n = total(rows, pick)
        return 100 * e / max(n, 1)

    dev = {k: v[0::2] for k, v in sets.items()}
    test = {k: v[1::2] for k, v in sets.items()}
    best_candidates, grid = None, []
    for variant in ("raw", "cased"):
        for per_token in (True, False):
            for lam in LAMBDAS:
                e = sum(total(rows, lambda r: choose(r["cands"], lam, variant, per_token))[0] for rows in dev.values())
                grid.append((e, lam, variant, per_token))
    e0 = next(e for e, lam, v, p in grid if lam == 0)
    _, lam, variant, per_token = min(grid)
    pick = lambda r: choose(r["cands"], lam, variant, per_token)  # noqa: E731
    top1 = lambda r: r["cands"][0]  # noqa: E731
    oracle = lambda r: min(r["cands"], key=lambda c: c["err"])  # noqa: E731
    all_test = [r for rows in test.values() for r in rows]
    lines = [
        "# MzansiLM rescoring experiment", "",
        f"{len(sets)} test sets, {sum(len(v) for v in sets.values())} clips, {len(all_test[0]['cands'])} candidates each. "
        "Lambda is tuned on even-numbered utterances and measured on the odd-numbered ones.", "",
        f"Chosen setting (best on the tuning half): lambda={lam}, text={variant}, "
        f"LM score {'per token' if per_token else 'summed'}. Tuning half: {100 * e0 / sum(r['ref_words'] for rows in dev.values() for r in rows):.2f}% WER "
        f"with the model's own top candidate.", "",
        "## Measured on the held-out half", "",
        "| | WER % |", "|---|---|",
        f"| Whisper's top candidate (beam search) | {wer(all_test, top1):.2f} |",
        f"| Rescored with MzansiLM | {wer(all_test, pick):.2f} |",
        f"| Oracle: best candidate per clip (the most rescoring could ever gain) | {wer(all_test, oracle):.2f} |", "",
        "## Per corpus and language (held-out half)", "",
        "| set | clips | top candidate | rescored | oracle |", "|---|---|---|---|---|",
    ]
    for k, rows in test.items():
        corpus, code = k.rsplit("_", 1)
        lines.append(f"| {corpus} / {languages[code].name if code in languages else code} | {len(rows)} | "
                     f"{wer(rows, top1):.1f} | {wer(rows, pick):.1f} | {wer(rows, oracle):.1f} |")
    lines += ["", "## Tuning grid (tuning half, total word errors; lower is better)", "",
              "| text | LM score | lambda | errors |", "|---|---|---|---|"]
    lines += [f"| {v} | {'per token' if p else 'summed'} | {la} | {e} |" for e, la, v, p in sorted(grid)[:12]]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    transformers_logging.set_verbosity_error()
    quiet_progress_bars()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eval_baseline.yaml")
    ap.add_argument("--model", default="KekeCode/sa-asr-whisper-lora-v2")
    ap.add_argument("--lm", default="uctnlp/mzansilm-125m")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-samples", type=int)
    ap.add_argument("--output-dir", default="results/lm_rescore")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--sync-repo")
    args = ap.parse_args()

    languages = load_languages()
    out_dir = resolve(args.output_dir)
    cache_dir = resolve(load_yaml(args.config).get("cache_dir", "data/eval_cache"))
    sync = HubSync(args.sync_repo) if args.sync_repo else None
    if sync:
        sync.pull(out_dir) if args.analyze_only else sync.pull(cache_dir, out_dir)
    if not args.analyze_only:
        generate(args, languages, cache_dir, out_dir, sync)
    md = analyze(out_dir, languages)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    if sync:
        sync.push_files(out_dir / "summary.md")


if __name__ == "__main__":
    main()
