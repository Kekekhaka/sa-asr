"""Tune how the audio language detector and the spelling-based one are combined.

    python scripts/lid_combined.py --config configs/eval_baseline.yaml --model KekeCode/sa-asr-whisper-lora-v3 \
        --max-samples 100 --sync-repo <user>/sa-asr-results
    python scripts/lid_combined.py --analyze-only --sync-repo <user>/sa-asr-results     # re-tune, no GPU

Step 1 (GPU): for every cached test clip, on the whole clip and on its first 5 s (how the app's mixed mode
sees speech), record the audio detector's probabilities and the transcript the app would produce (clip
transcribed under the audio's top language).
Step 2 (CPU): combine audio and text exactly as inference.combine does, for a grid of text weights and
text-model temperatures, and report exact-language accuracy. The text model is trained here from the
collected *training* transcripts only (data/lid_texts), so no test sentence is in it. The setting is chosen
on one random half of the clips and reported on the other half.

Writes results/lid_combined/records.jsonl and summary.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import zlib
from collections import defaultdict
from pathlib import Path

from sa_asr.config import PROJECT_ROOT, load_languages, load_yaml, quiet_progress_bars, select_languages
from sa_asr.hub import HubSync
from sa_asr.lid import FAMILIES, family_of
from sa_asr.textlid import TextLID, combine

log = logging.getLogger("lid_combined")
WEIGHTS = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
TEMPERATURES = [0.03, 0.05, 0.1, 0.2]
CROPS = {"full": None, "5s": 5.0}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def generate(args, languages, cache_dir: Path, out_dir: Path) -> None:
    from transformers.utils import logging as transformers_logging

    from sa_asr.audio import SAMPLE_RATE
    from sa_asr.data import build_eval_set, corpus_languages, eval_row_audio
    from sa_asr.inference import load_transcriber, resolve_model

    transformers_logging.set_verbosity_error()
    cfg = load_yaml(args.config)
    max_samples = args.max_samples if args.max_samples is not None else cfg.get("max_samples", 200)
    seed, max_dur, buffer = cfg.get("seed", 42), cfg.get("max_duration_s", 30.0), cfg.get("shuffle_buffer", 500)
    model_ref, trained = resolve_model(args.model)
    model = load_transcriber(model_ref, finetuned=True)
    candidates = [languages[c] for c in model.proxy_tokens if c in languages]
    records = []
    for corpus, codes in cfg["eval_sets"].items():
        for lang in corpus_languages(corpus, select_languages(languages, codes)):
            if lang.code not in trained:
                continue
            rows = build_eval_set(corpus, lang, cfg.get("split", "test"), cache_dir, max_samples=max_samples,
                                  seed=seed, max_duration_s=max_dur, shuffle_buffer=buffer)
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                wavs = [eval_row_audio({"flac": b}) for b in batch["flac"]]
                for cond, crop in CROPS.items():
                    clips = wavs if crop is None else [w[: int(crop * SAMPLE_RATE)] for w in wavs]
                    probs = model.detect_languages(clips, candidates)
                    texts = [""] * len(clips)
                    by_lang = defaultdict(list)
                    for i, p in enumerate(probs):
                        by_lang[next(iter(p))].append(i)
                    for code, idx in by_lang.items():
                        for i, t in zip(idx, model.transcribe([clips[i] for i in idx], languages[code])):
                            texts[i] = t
                    for utt_id, p, t in zip(batch["utt_id"], probs, texts):
                        records.append({"corpus": corpus, "lang": lang.code, "cond": cond, "utt_id": utt_id,
                                        "audio": {c: round(v, 5) for c, v in p.items()}, "text": t})
            log.info("%s/%s: %d clips", corpus, lang.code, len(rows))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def train_text_model(texts_dir: Path) -> TextLID:
    by_lang = {p.stem: [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
               for p in sorted(texts_dir.glob("*.txt"))}
    if not by_lang:
        raise SystemExit(f"No training texts in {texts_dir}; run the text_lid stage first")
    return TextLID.train(by_lang)


def analyze(out_dir: Path, texts_dir: Path, languages: dict) -> str:
    recs = [json.loads(l) for l in (out_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    text_model = train_text_model(texts_dir)
    post_cache: dict[tuple[int, float], dict] = {}
    # The weight a clip gets can depend on the family of the audio's top language (textlid.family_weight).
    audio_family = {i: family_of(max(r["audio"], key=r["audio"].get)) for i, r in enumerate(recs)}

    def posterior(i: int, r: dict, temp: float) -> dict:
        key = (i, temp)
        if key not in post_cache:
            post_cache[key] = text_model.posterior(r["text"], temp)
        return post_cache[key]

    def predict(i: int, r: dict, weight, temp: float) -> str:
        if weight == "text":  # the spelling alone
            p = posterior(i, r, temp)
            return max(p, key=p.get)
        w = weight.get(audio_family[i], weight.get("default", 1.0)) if isinstance(weight, dict) else weight
        return next(iter(combine(r["audio"], posterior(i, r, temp), w, len(r["text"]))))

    def acc(rows, weight, temp, family=False):
        hits = sum((family_of(predict(i, r, weight, temp)) == family_of(r["lang"])) if family
                   else predict(i, r, weight, temp) == r["lang"] for i, r in rows)
        return 100 * hits / max(len(rows), 1)

    indexed = list(enumerate(recs))
    # Split by clip (both conditions of a clip go to the same half): tune on one half, report on the other.
    half = lambda r: zlib.crc32(f"{r['corpus']}/{r['lang']}/{r['utt_id']}".encode()) % 2  # noqa: E731
    dev = [(i, r) for i, r in indexed if half(r) == 0]
    test = [(i, r) for i, r in indexed if half(r) == 1]
    grid = [(sum(acc([x for x in dev if x[1]["cond"] == c], w, t) for c in CROPS), w, t)
            for w in WEIGHTS for t in TEMPERATURES]
    # Ties go to the setting nearest the app's current one.
    _, best_w, best_t = max(grid, key=lambda g: (g[0], -abs(g[1] - 1.0), -abs(g[2] - 0.05)))

    # Per family: a clip's weight depends only on its audio family, so each family's weight is tuned on its own
    # clips (both conditions counted), for each temperature; the temperature with the most hits overall wins.
    families = sorted(set(audio_family.values()))
    fam_best = None
    for t in TEMPERATURES:
        weights, total = {}, 0
        for f in families:
            rows = [x for x in dev if audio_family[x[0]] == f]
            hits = {w: sum(predict(i, r, w, t) == r["lang"] for i, r in rows) for w in WEIGHTS}
            weights[f] = max(WEIGHTS, key=lambda w: (hits[w], -abs(w - best_w)))
            total += hits[weights[f]]
        if fam_best is None or total > fam_best[0]:
            fam_best = (total, t, weights)
    _, fam_t, fam_w = fam_best
    fam_desc = ", ".join(f"{f} {w}" for f, w in fam_w.items())

    settings = [("audio only", 0, 0.05), ("spelling only", "text", best_t), ("current app (w 1.0, T 0.05)", 1.0, 0.05),
                (f"one weight (w {best_w}, T {best_t})", best_w, best_t), (f"per family (T {fam_t})", fam_w, fam_t)]
    lines = ["# Audio + spelling language identification: tuning", "",
             f"{len(recs)} clip-conditions. Text model trained only on collected training transcripts. "
             "Chosen on a random half of the clips and reported on the other half.", "",
             f"**One weight for all: text weight {best_w}, temperature {best_t}** (the app uses weight 1.0, temperature 0.05).",
             "",
             f"**Per family of the audio's top language (temperature {fam_t}):** {fam_desc}.", "",
             "## Held-out accuracy (exact language / family)", "",
             "| setting | " + " | ".join(f"{c} exact | {c} family" for c in CROPS) + " |",
             "|---|" + "---|---|" * len(CROPS)]
    for name, w, t in settings:
        cells = []
        for c in CROPS:
            rows = [x for x in test if x[1]["cond"] == c]
            cells += [f"{acc(rows, w, t):.1f}%", f"{acc(rows, w, t, True):.1f}%"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Weight sweep (held-out, exact; temperature = one-weight choice)", "",
              "| weight | " + " | ".join(CROPS) + " |", "|---|" + "---|" * len(CROPS)]
    for w in WEIGHTS:
        lines.append(f"| {w} | " + " | ".join(f"{acc([x for x in test if x[1]['cond'] == c], w, best_t):.1f}%" for c in CROPS) + " |")
    head = "| " + " | ".join(n for n, _, _ in settings) + " |"
    lines += ["", "## Per language (held-out, whole clip, exact)", "", "| language " + head,
              "|---|" + "---|" * len(settings)]
    for code in sorted({r["lang"] for _, r in test}):
        rows = [x for x in test if x[1]["cond"] == "full" and x[1]["lang"] == code]
        name = languages[code].name if code in languages else code
        lines.append(f"| {name} | " + " | ".join(f"{acc(rows, w, t):.0f}%" for _, w, t in settings) + " |")
    lines += ["", "## By corpus (held-out, whole clip, exact)", "", "| corpus " + head, "|---|" + "---|" * len(settings)]
    for corpus in sorted({r["corpus"] for _, r in test}):
        rows = [x for x in test if x[1]["cond"] == "full" and x[1]["corpus"] == corpus]
        lines.append(f"| {corpus} | " + " | ".join(f"{acc(rows, w, t):.1f}%" for _, w, t in settings) + " |")
    st = FAMILIES["Sotho-Tswana"]
    lines += ["", "## Sotho-Tswana by corpus (held-out, whole clip, exact)", "", "| corpus / language " + head,
              "|---|" + "---|" * len(settings)]
    for corpus in sorted({r["corpus"] for _, r in test}):
        for code in st:
            rows = [x for x in test if x[1]["cond"] == "full" and x[1]["corpus"] == corpus and x[1]["lang"] == code]
            if rows:
                lines.append(f"| {corpus} / {code} ({len(rows)}) | "
                             + " | ".join(f"{acc(rows, w, t):.0f}%" for _, w, t in settings) + " |")
    for name, w, t in (settings[0], settings[-1]):
        lines += ["", f"## Sotho-Tswana confusions, {name} (held-out, whole clip): rows = true, columns = detected", "",
                  "| true \\ detected | " + " | ".join(st) + " | other |", "|---|" + "---|" * (len(st) + 1)]
        for code in st:
            rows = [x for x in test if x[1]["cond"] == "full" and x[1]["lang"] == code]
            got = [predict(i, r, w, t) for i, r in rows]
            n = max(len(got), 1)
            lines.append(f"| {code} | " + " | ".join(f"{100 * got.count(c) / n:.0f}%" for c in st)
                         + f" | {100 * sum(g not in st for g in got) / n:.0f}% |")
    lines += ["", "## For the app", "",
              "inference.TEXT_WEIGHT = {" + ", ".join(f"{f!r}: {w}" for f, w in fam_w.items()) + "}, "
              f"TEXT_TEMPERATURE = {fam_t}, if the per-family row beats the current app above."]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    quiet_progress_bars()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eval_baseline.yaml")
    ap.add_argument("--model", default="KekeCode/sa-asr-whisper-lora-v6")
    ap.add_argument("--max-samples", type=int)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output-dir", default="results/lid_combined")
    ap.add_argument("--texts-dir", default="data/lid_texts")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--sync-repo")
    args = ap.parse_args()

    languages = load_languages()
    out_dir, texts_dir = resolve(args.output_dir), resolve(args.texts_dir)
    cache_dir = resolve(load_yaml(args.config).get("cache_dir", "data/eval_cache"))
    sync = HubSync(args.sync_repo) if args.sync_repo else None
    if sync:
        sync.pull(texts_dir, out_dir) if args.analyze_only else sync.pull(texts_dir, cache_dir)
    if not args.analyze_only:
        generate(args, languages, cache_dir, out_dir)
        if sync:
            sync.push_files(out_dir / "records.jsonl")
    md = analyze(out_dir, texts_dir, languages)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    if sync:
        sync.push_files(out_dir / "summary.md")


if __name__ == "__main__":
    main()
