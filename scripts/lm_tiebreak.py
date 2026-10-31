"""Experiment: when language detection hesitates between two close languages, let the transcripts decide.

    python scripts/lm_tiebreak.py --config configs/eval_baseline.yaml --model KekeCode/sa-asr-whisper-lora-v4 \
        --lid results/lid_v4/predictions.jsonl --output-dir results/lm_tiebreak_v4 --max-samples 100 \
        --sync-repo <user>/sa-asr-results
    python scripts/lm_tiebreak.py --analyze-only --output-dir results/lm_tiebreak_v4 --sync-repo <user>/sa-asr-results

Most language-ID mistakes are between siblings (isiNdebele heard as isiZulu, Setswana as Sepedi). For every test
clip whose two most likely languages (from evaluate_lid.py, whole clip) are siblings and the top one is below
--max-top, Whisper transcribes it in both languages (n-best) and MzansiLM scores every candidate.

Step 2 (CPU) picks the language with the best  candidate score + alpha * log P(language), where a candidate's score
is  asr + lambda * LM  (step 1's rescoring) and the second language is only considered when its probability is at
least margin * the first's. margin, lambda and alpha are tuned on even-numbered clips and measured on odd ones, as in
lm_rescore.py. Reported: language-ID accuracy over all clips, and WER on the clips where the choice was made.

Writes results/lm_tiebreak_v4/clips/<corpus>_<lang>.jsonl (both languages' candidates per clip) and summary.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers.utils import logging as transformers_logging

from lm_rescore import lm_logprob, load_lm, resolve, word_errors
from sa_asr.config import load_languages, load_yaml, quiet_progress_bars, select_languages
from sa_asr.data import build_eval_set, corpus_languages, eval_row_audio
from sa_asr.hub import HubSync
from sa_asr.inference import load_transcriber, resolve_model
from sa_asr.text import normalize_for_scoring

log = logging.getLogger("lm_tiebreak")
FAMILIES = [{"zul", "xho", "ssw", "nbl"}, {"nso", "sot", "tsn"}]
MARGINS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
LAMBDAS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
ALPHAS = [0, 0.05, 0.1, 0.2, 0.5, 1.0]


def siblings(a: str, b: str) -> bool:
    return any(a in f and b in f for f in FAMILIES)


def load_lid(path: Path) -> dict[str, list[dict]]:
    """Whole-clip predictions per test set, in file order (the order the clips were scored)."""
    sets: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line) if line.strip() else None
        if r and r["cond"] == "full":
            sets[f"{r['corpus']}_{r['lang']}"].append(r)
    return sets


def eligible(r: dict, max_top: float) -> bool:
    top = r["top3"]
    return len(top) > 1 and top[0][1] < max_top and siblings(top[0][0], top[1][0])


def generate(args, languages, cache_dir: Path, out_dir: Path, lid: dict, sync: HubSync | None) -> None:
    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ref, trained = resolve_model(args.model)
    whisper = load_transcriber(model_ref, finetuned=True)
    tok, lm = load_lm(args.lm, device)
    max_samples = args.max_samples if args.max_samples is not None else cfg.get("max_samples", 200)
    seed, max_dur, buffer = cfg.get("seed", 42), cfg.get("max_duration_s", 30.0), cfg.get("shuffle_buffer", 500)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for corpus, codes in cfg["eval_sets"].items():
        for lang in corpus_languages(corpus, select_languages(languages, codes)):
            key = f"{corpus}_{lang.code}"
            target = clips_dir / f"{key}.jsonl"
            todo = {r["utt_id"]: r for r in lid.get(key, []) if eligible(r, args.max_top)}
            if lang.code not in trained or target.exists():
                continue
            rows = build_eval_set(corpus, lang, cfg.get("split", "test"), cache_dir, max_samples=max_samples,
                                  seed=seed, max_duration_s=max_dur, shuffle_buffer=buffer)
            t0 = time.time()
            picked = [(u, ref, flac) for u, ref, flac in zip(rows["utt_id"], rows["text"], rows["flac"]) if u in todo]
            records = {u: {"utt_id": u, "true": lang.code, "ref_words": len(normalize_for_scoring(ref).split()),
                           "probs": dict(todo[u]["top3"]), "top": [c for c, _ in todo[u]["top3"][:2]], "langs": {}}
                       for u, ref, _ in picked}
            refs = {u: ref for u, ref, _ in picked}
            # one batch per candidate language, so every clip in it is decoded with the same language token
            by_lang: dict[str, list] = defaultdict(list)
            for u, _, flac in picked:
                for code in records[u]["top"]:
                    by_lang[code].append((u, flac))
            for code, items in by_lang.items():
                for start in range(0, len(items), args.batch_size):
                    batch = items[start : start + args.batch_size]
                    wavs = [eval_row_audio({"flac": f}) for _, f in batch]
                    nbest = whisper.transcribe_nbest(wavs, languages[code], args.beams)
                    flat = [t for cands in nbest for t, _ in cands]
                    scored = iter(lm_logprob(tok, lm, flat, device))
                    for (u, _), cands in zip(batch, nbest):
                        records[u]["langs"][code] = []
                        for text, asr in cands:
                            lp, ntok = next(scored)
                            records[u]["langs"][code].append({"text": text, "asr": asr, "lm": lp, "ntok": ntok,
                                                              "err": word_errors(refs[u], text)})
            tmp = target.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(records[u], ensure_ascii=False) + "\n" for u, _, _ in picked),
                           encoding="utf-8")
            tmp.replace(target)
            log.info("%s: %d of %d clips hesitate between siblings, decoded in both in %.0f s", key, len(picked),
                     len(rows), time.time() - t0)
            if sync:
                sync.push_files(target)


def cand_score(c: dict, lam: float, per_token: bool) -> float:
    return c["asr"] + lam * (c["lm"] / max(c["ntok"], 1) if per_token else c["lm"])


def decide(r: dict, margin: float | None, lam: float, alpha: float, per_token: bool) -> tuple[str, dict]:
    """(chosen language, chosen candidate). margin None = always the detector's top language."""
    first, second = r["top"]
    langs = [first]
    if margin is not None and r["probs"][second] >= margin * r["probs"][first]:
        langs.append(second)
    best = {code: max(r["langs"][code], key=lambda c: cand_score(c, lam, per_token)) for code in langs}
    code = max(langs, key=lambda k: cand_score(best[k], lam, per_token) + alpha * math.log(max(r["probs"][k], 1e-9)))
    return code, best[code]


def analyze(out_dir: Path, lid: dict, languages: dict, max_top: float) -> str:
    clips = {}
    for f in sorted((out_dir / "clips").glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                clips[r["utt_id"]] = r
    if not clips:
        return "No clip files found."
    # halves by position in each set's LID list: the same split for clips with and without a choice
    dev, test, test_all = [], [], []
    for recs in lid.values():
        for i, r in enumerate(recs):
            c = clips.get(r["utt_id"]) if eligible(r, max_top) else None
            if i % 2:
                test_all.append((r, c))
            if c and len(c["langs"]) == len(c["top"]):
                (test if i % 2 else dev).append(c)

    def errors(rows, *setting):
        return sum(decide(r, *setting)[1]["err"] for r in rows)

    grid = [(m, lam, a, pt) for m, lam, a, pt in itertools.product(MARGINS, LAMBDAS, ALPHAS, (True, False))]
    best = min(grid, key=lambda s: (errors(dev, *s), -sum(decide(r, *s)[0] == r["true"] for r in dev)))
    only_rescore = min(((None, lam, 0, pt) for lam in LAMBDAS for pt in (True, False)), key=lambda s: errors(dev, *s))
    top1 = (None, 0, 0, True)
    oracle_lang = lambda r: r["true"] if r["true"] in r["top"] else r["top"][0]  # noqa: E731

    def wer(rows, setting):
        n = max(sum(r["ref_words"] for r in rows), 1)
        if setting == "oracle":
            return 100 * sum(r["langs"][oracle_lang(r)][0]["err"] for r in rows) / n
        return 100 * errors(rows, *setting) / n

    def lid_acc(setting):  # over every test clip, decided or not
        ok = 0
        for r, c in test_all:
            if c is None or len(c["langs"]) < len(c["top"]):
                ok += r["pred"] == r["lang"]
            else:
                ok += (oracle_lang(c) if setting == "oracle" else decide(c, *setting)[0]) == c["true"]
        return 100 * ok / max(len(test_all), 1)

    m, lam, a, pt = best
    rows = [("Detector's top language, Whisper's top candidate (today)", top1),
            (f"Detector's top language, rescored candidate (lambda {only_rescore[1]}, "
             f"{'per token' if only_rescore[3] else 'summed'})", only_rescore),
            (f"Tie-break between the two siblings + rescoring (margin {m}, lambda {lam}, alpha {a}, "
             f"{'per token' if pt else 'summed'})", best),
            ("Oracle: the right language whenever it is one of the two", "oracle")]
    lines = ["# Tie-breaking close languages with the transcripts (MzansiLM)", "",
             f"Clips whose two most likely languages are siblings (Nguni: isiZulu, isiXhosa, siSwati, isiNdebele; "
             f"Sotho-Tswana: Sepedi, Sesotho, Setswana) and whose top language is below {max_top:.0%}: "
             f"{len(dev) + len(test)} of {sum(len(v) for v in lid.values())}. Each was transcribed in both languages "
             f"({len(next(iter(clips.values()))['langs'][next(iter(clips.values()))['top'][0]])} candidates each). "
             "Settings tuned on even-numbered clips, measured on the odd ones.", "",
             "Close languages often get the same words whichever of the two Whisper is told, so a better language "
             "choice shows mostly in the language-ID column (the label the app shows, and which second model and "
             "translation a part gets), less in WER.", "",
             f"| | language ID, all {len(test_all)} test clips | WER on the {len(test)} hesitant test clips |",
             "|---|---|---|"]
    lines += [f"| {name} | {lid_acc(s):.1f}% | {wer(test, s):.2f} |" for name, s in rows]
    lines += ["", "## Hesitant test clips per true language", "",
              "| language | clips | LID today | LID tie-break | WER today | WER tie-break |", "|---|---|---|---|---|---|"]
    by_lang = defaultdict(list)
    for r in test:
        by_lang[r["true"]].append(r)
    for code, rs in sorted(by_lang.items()):
        lid_before = 100 * sum(r["top"][0] == code for r in rs) / len(rs)
        lid_after = 100 * sum(decide(r, *best)[0] == code for r in rs) / len(rs)
        lines.append(f"| {languages[code].name} | {len(rs)} | {lid_before:.0f}% | {lid_after:.0f}% | "
                     f"{wer(rs, top1):.1f} | {wer(rs, best):.1f} |")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    transformers_logging.set_verbosity_error()
    quiet_progress_bars()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eval_baseline.yaml")
    ap.add_argument("--model", default="KekeCode/sa-asr-whisper-lora-v4")
    ap.add_argument("--lm", default="uctnlp/mzansilm-125m")
    ap.add_argument("--lid", default="results/lid_v4/predictions.jsonl", help="evaluate_lid.py's predictions")
    ap.add_argument("--max-top", type=float, default=0.98, help="only clips whose top language is below this")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-samples", type=int)
    ap.add_argument("--output-dir", default="results/lm_tiebreak_v4")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--sync-repo")
    args = ap.parse_args()

    languages = load_languages()
    out_dir, lid_path = resolve(args.output_dir), resolve(args.lid)
    cache_dir = resolve(load_yaml(args.config).get("cache_dir", "data/eval_cache"))
    sync = HubSync(args.sync_repo) if args.sync_repo else None
    if sync:
        folders = [out_dir, lid_path.parent] + ([] if args.analyze_only else [cache_dir])
        sync.pull(*folders)
    lid = load_lid(lid_path)
    if not args.analyze_only:
        generate(args, languages, cache_dir, out_dir, lid, sync)
    md = analyze(out_dir, lid, languages, args.max_top)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    if sync:
        sync.push_files(out_dir / "summary.md")


if __name__ == "__main__":
    main()
