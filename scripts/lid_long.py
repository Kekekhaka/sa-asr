"""How good is language detection on whole recordings, not single clips? (CPU only, no new transcription)

    python scripts/lid_long.py                      # needs results/lid_combined/records.jsonl and data/lid_texts

The test clips are 2-30 s, but the app judges a whole recording: it labels every 4-10 s window and smooths the
labels (inference.smooth_languages). This builds recordings out of the saved clips and applies the app's own
combination (audio + spelling, weight 1) and smoothing to them:

  A. one-language recordings of k windows (k = 1, 2, 4, 8): how often is every window labelled right, and how
     often is the recording's main language right?
  B. two-language recordings (k windows of language X, then k of language Y): are both stretches labelled
     correctly with the switch in the right place?

Caveat: the windows of a simulated recording come from different speakers, so their errors are more independent
than in a real recording by one speaker, which makes the numbers optimistic. Treat them as an upper estimate.
The pairs in B are whole clips joined at a boundary; they test switching between languages, not words of two
languages inside one sentence.

Writes results/lid_long/summary.md.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from sa_asr.config import PROJECT_ROOT, load_languages
from sa_asr.inference import SWITCH_PENALTY, TEXT_WEIGHT, smooth_languages
from sa_asr.textlid import TextLID, combine

KS = (1, 2, 4, 8)
PAIRS = [("zul", "eng"), ("sot", "eng"), ("zul", "xho"), ("zul", "nbl"), ("sot", "tsn"), ("nso", "sot"), ("nso", "tsn"),
         ("ssw", "zul"), ("xho", "ven"), ("afr", "eng")]
WINDOW_WEIGHT = 2.0  # a 10 s window; the weight only sets the evidence per window against the switch penalty


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def windows(records_path: Path, texts_dir: Path, cond: str) -> dict[tuple[str, str], list[dict]]:
    """{(corpus, lang): [combined posterior of each clip]}"""
    tm = TextLID.train({p.stem: [x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
                        for p in sorted(texts_dir.glob("*.txt"))})
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for line in records_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["cond"] != cond:
            continue
        out[(r["corpus"], r["lang"])].append(
            combine(r["audio"], tm.posterior(r["text"]), TEXT_WEIGHT, len(r["text"])))
    return out


PENALTY = SWITCH_PENALTY


def label(probs: list[dict]) -> list[str]:
    return smooth_languages(probs, [WINDOW_WEIGHT] * len(probs), PENALTY)


def single_language(pool: dict, k: int, n: int, rng: random.Random) -> tuple[float, float, float]:
    """(window accuracy, share of recordings with every window right, share with the main language right)"""
    keys = [key for key, v in pool.items() if len(v) >= k]
    win = all_ok = main_ok = 0
    for _ in range(n):
        corpus, lang = rng.choice(keys)
        labels = label(rng.sample(pool[(corpus, lang)], k))
        win += sum(x == lang for x in labels) / k
        all_ok += all(x == lang for x in labels)
        main_ok += Counter(labels).most_common(1)[0][0] == lang
    return 100 * win / n, 100 * all_ok / n, 100 * main_ok / n


def two_language(pool: dict, a: str, b: str, k: int, n: int, rng: random.Random):
    """(window accuracy, share of recordings labelled exactly A...A B...B) for k windows of each language"""
    by_lang = defaultdict(list)
    for (corpus, lang), v in pool.items():
        by_lang[lang] += v
    if len(by_lang[a]) < k or len(by_lang[b]) < k:
        return None
    win = exact = 0
    for _ in range(n):
        labels = label(rng.sample(by_lang[a], k) + rng.sample(by_lang[b], k))
        win += (sum(x == a for x in labels[:k]) + sum(x == b for x in labels[k:])) / (2 * k)
        exact += labels == [a] * k + [b] * k
    return 100 * win / n, 100 * exact / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default="results/lid_combined/records.jsonl")
    ap.add_argument("--texts-dir", default="data/lid_texts")
    ap.add_argument("--output-dir", default="results/lid_long")
    ap.add_argument("--n", type=int, default=1500, help="simulated recordings per setting")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    languages = load_languages()
    name = lambda c: languages[c].name if c in languages else c  # noqa: E731
    rng = random.Random(args.seed)
    lines = ["# Language detection on whole recordings (simulated from saved clips)", "",
             "Each window is a saved test clip (2-30 s) with its audio probabilities combined with the spelling "
             "check, labelled with the app's smoothing. Windows of one recording come from different speakers, so "
             "these numbers are optimistic for real recordings by one speaker.", ""]
    for cond, title in (("full", "whole clips as windows"), ("5s", "first 5 s of each clip as windows")):
        pool = windows(resolve(args.records), resolve(args.texts_dir), cond)
        lines += [f"## One-language recordings ({title})", "",
                  "| windows per recording | each window right | every window right | main language right |", "|---|---|---|---|"]
        for k in KS:
            w, a, m = single_language(pool, k, args.n, rng)
            lines.append(f"| {k} | {w:.1f}% | {a:.1f}% | {m:.1f}% |")
        lines += [""]
        by_lang = defaultdict(list)
        for (corpus, lang), v in pool.items():
            by_lang[lang] += v
        lines += [f"Main language right, by language ({title}), 4 windows then 8 windows:", "",
                  "| language | 1 window | 4 windows | 8 windows |", "|---|---|---|---|"]
        for lang in sorted(by_lang):
            row = []
            for k in (1, 4, 8):
                ok = sum(Counter(label(rng.sample(by_lang[lang], k))).most_common(1)[0][0] == lang for _ in range(400))
                row.append(f"{100 * ok / 400:.0f}%")
            lines.append(f"| {name(lang)} | " + " | ".join(row) + " |")
        lines += [""]
        if cond == "full":
            lines += ["## Two-language recordings: k windows of the first language, then k of the second", "",
                      "Share of recordings labelled exactly right (every window, one switch at the boundary), and the "
                      "share of windows right.", "",
                      "| pair | k=1 exact / windows | k=2 | k=4 | k=8 |", "|---|---|---|---|---|"]
            for a, b in PAIRS:
                cells = []
                for k in KS:
                    r = two_language(pool, a, b, k, args.n // 3, rng)
                    cells.append("n/a" if r is None else f"{r[1]:.0f}% / {r[0]:.0f}%")
                lines.append(f"| {name(a)} then {name(b)} | " + " | ".join(cells) + " |")
            lines += [""]
    # How strong should the smoothing be? Stronger keeps one-language recordings steady but may miss short switches.
    global PENALTY
    pool = windows(resolve(args.records), resolve(args.texts_dir), "full")
    lines += ["## Smoothing strength (switch penalty; the app uses " + f"{SWITCH_PENALTY:g})", "",
              "One-language recordings of 8 windows, and two-language recordings (average over the pairs above), "
              "by how many windows each language gets.", "",
              "| penalty | 1-lang: each window right | 1-lang: every window right | 2-lang exact k=1 | k=2 | k=4 | k=8 |",
              "|---|---|---|---|---|---|---|"]
    for pen in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0):
        PENALTY = pen
        w, a, _ = single_language(pool, 8, args.n // 2, random.Random(1))
        exact = []
        for k_ in KS:
            vals = [two_language(pool, x, y, k_, args.n // 6, random.Random(2)) for x, y in PAIRS]
            exact.append(sum(v[1] for v in vals if v) / max(sum(1 for v in vals if v), 1))
        lines.append(f"| {pen:g} | {w:.1f}% | {a:.1f}% | " + " | ".join(f"{e:.0f}%" for e in exact) + " |")
    PENALTY = SWITCH_PENALTY
    md = "\n".join(lines)
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
