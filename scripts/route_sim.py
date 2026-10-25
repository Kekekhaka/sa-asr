"""Simulate routing between our model and dsfsi-anv (no GPU, no new transcription).

    python scripts/route_sim.py --sync-repo <user>/sa-asr-results

dsfsi-anv beats our model on clean close-microphone speech (Swivuriso, a little on FLEURS) but is far
worse on telephone speech (Lwazi) and on NCHLT read prompts. A router that sends a clip to dsfsi only when
its language is one dsfsi covers AND the audio is wideband could keep the gain and avoid the losses.

This joins the cached predictions of both models (results/baseline and results/natural) with one number per
clip from the cached audio: how much of its energy lies above 4 kHz (telephone audio has almost none), and
scores each routing rule on the clips of dsfsi's languages. The rule is chosen on one random half of the
clips and reported on the other half. The language is taken from the test set's label, i.e. an ideal
language detector (ours is ~99% on Swivuriso and ~80-95% elsewhere; the app would route on its detected
language).

Writes results/route_sim/summary.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import zlib
from pathlib import Path

import numpy as np

from sa_asr.config import PROJECT_ROOT
from sa_asr.hub import HubSync

log = logging.getLogger("route_sim")
DSFSI = "dsfsi-anv-multilingual-turbo"
OURS = "sa-turbo-lora-v3"
DSFSI_LANGS = {"zul", "xho", "sot", "tsn", "tso", "ven"}
THRESHOLDS_DB = [-80, -70, -60, -55, -50, -45, -40, -35, -30, -25, -20, 0]
MIN_SECONDS = [0, 3, 4, 5, 6, 7, 8, 10, 12]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def words(text: str) -> list[str]:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", "", text.lower())).strip().split()


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(word errors, reference words) after lowercasing and stripping punctuation."""
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[:], i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1], len(r)


def high_band_db(wav: np.ndarray, sr: int = 16000) -> float:
    """Energy above 4 kHz relative to 0.3-4 kHz, in dB (about -60 or lower for telephone audio)."""
    n = 512
    if len(wav) < n:
        return -100.0
    frames = np.lib.stride_tricks.sliding_window_view(wav.astype(np.float64), n)[:: n // 2]
    spec = (np.abs(np.fft.rfft(frames * np.hanning(n), axis=1)) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    low, high = spec[(freqs >= 300) & (freqs < 4000)].sum(), spec[freqs >= 4000].sum()
    return float(10 * np.log10((high + 1e-12) / (low + 1e-12)))


def load_preds(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if "reference" in d and "hypothesis" in d:
            out[str(d["utt_id"])] = d
    return out


def collect(root: Path, cache_dir: Path) -> list[dict]:
    """One record per clip of dsfsi's languages that both models transcribed."""
    from datasets import load_from_disk

    from sa_asr.audio import decode_audio

    recs = []
    for f in sorted((root / "predictions" / DSFSI).glob("*.jsonl")):
        stem = f.stem
        lang = stem.split("_test_")[0].rsplit("_", 1)[1]
        ours_file = root / "predictions" / OURS / f.name
        if lang not in DSFSI_LANGS or not ours_file.exists() or not (cache_dir / stem).exists():
            continue
        a, b = load_preds(f), load_preds(ours_file)
        ds = load_from_disk(str(cache_dir / stem))
        feats = {str(u): ds[i] for i, u in enumerate(ds["utt_id"])}
        corpus = stem.split("_test_")[0].rsplit("_", 1)[0]
        for utt, da in a.items():
            if utt in b and utt in feats:
                ea, n = word_errors(da["reference"], da["hypothesis"])
                eb, _ = word_errors(da["reference"], b[utt]["hypothesis"])
                wav = decode_audio({"bytes": feats[utt]["flac"]})
                recs.append({"set": stem, "corpus": corpus, "lang": lang, "utt": utt, "n": n, "dsfsi": ea,
                             "ours": eb, "hb": high_band_db(wav), "sec": len(wav) / 16000})
    return recs


def analyze(recs: list[dict]) -> str:
    half = lambda r: zlib.crc32(f"{r['set']}/{r['utt']}".encode()) % 2  # noqa: E731
    dev = [r for r in recs if half(r) == 0]
    test = [r for r in recs if half(r) == 1]

    def wer(rows, pick):
        return 100 * sum(pick(r) for r in rows) / max(sum(r["n"] for r in rows), 1)

    def route(t, d=0):
        return lambda r: r["dsfsi"] if r["hb"] >= t and r["sec"] >= d else r["ours"]

    best = min(THRESHOLDS_DB, key=lambda t: wer(dev, route(t)))
    best_t2, best_d = min(((t, d) for t in THRESHOLDS_DB for d in MIN_SECONDS), key=lambda td: wer(dev, route(*td)))
    corpora = sorted({r["corpus"] for r in recs})
    lines = ["# Routing simulation: our model vs dsfsi-anv on dsfsi's six languages", "",
             f"{len(recs)} clips (zul, xho, sot, tsn, tso, ven) from {len(corpora)} corpora; ideal language detector. "
             "Threshold = energy above 4 kHz relative to 0.3-4 kHz (dB); clips at or above it go to dsfsi. "
             "Chosen on one random half of the clips, reported on the other half.", "",
             f"**Chosen threshold: {best} dB**", "",
             "## Held-out WER (%)", "", "| rule | " + " | ".join(corpora) + " | all |", "|---|" + "---|" * (len(corpora) + 1)]
    rules = [("our model only", lambda r: r["ours"]), ("dsfsi only", lambda r: r["dsfsi"]),
             (f"route (>= {best} dB to dsfsi)", route(best)),
             (f"route (>= {best_t2} dB and >= {best_d} s to dsfsi)", route(best_t2, best_d)),
             ("oracle: best of the two per clip", lambda r: min(r["ours"], r["dsfsi"]))]
    for name, pick in rules:
        cells = [f"{wer([r for r in test if r['corpus'] == c], pick):.1f}" for c in corpora]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {wer(test, pick):.1f} |")
    lines += ["", "## Threshold sweep (held-out WER %, all sets / share of clips sent to dsfsi)", "",
              "| threshold dB | WER | to dsfsi |", "|---|---|---|"]
    for t in THRESHOLDS_DB:
        share = 100 * sum(r["hb"] >= t for r in test) / max(len(test), 1)
        lines.append(f"| {t} | {wer(test, route(t)):.1f} | {share:.0f}% |")
    lines += ["", "## Minimum clip length (held-out WER %, bandwidth threshold = chosen; share of clips to dsfsi)", "",
              "| min seconds | WER | to dsfsi |", "|---|---|---|"]
    for d in MIN_SECONDS:
        share = 100 * sum(r["hb"] >= best and r["sec"] >= d for r in test) / max(len(test), 1)
        lines.append(f"| {d} | {wer(test, route(best, d)):.1f} | {share:.0f}% |")
    lines += ["", "## Clip length per corpus (median / 10th / 90th percentile, seconds)", "", "| corpus | median | 10th | 90th |", "|---|---|---|---|"]
    for c in corpora:
        v = np.array([r["sec"] for r in recs if r["corpus"] == c])
        lines.append(f"| {c} | {np.median(v):.1f} | {np.percentile(v, 10):.1f} | {np.percentile(v, 90):.1f} |")
    lines += ["", "## Typical bandwidth per corpus (median dB)", "", "| corpus | median | 10th pct | 90th pct |", "|---|---|---|---|"]
    for c in corpora:
        v = np.array([r["hb"] for r in recs if r["corpus"] == c])
        lines.append(f"| {c} | {np.median(v):.0f} | {np.percentile(v, 10):.0f} | {np.percentile(v, 90):.0f} |")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="data/eval_cache")
    ap.add_argument("--output-dir", default="results/route_sim")
    ap.add_argument("--sync-repo")
    args = ap.parse_args()

    cache_dir, out_dir = resolve(args.cache_dir), resolve(args.output_dir)
    if args.sync_repo:
        HubSync(args.sync_repo).pull(Path("results/baseline"), Path("results/natural"), cache_dir)
    recs = []
    for root in ("results/baseline", "results/natural"):
        recs += collect(resolve(root), cache_dir)
    if not recs:
        raise SystemExit("No clips found: pull results/baseline, results/natural and data/eval_cache first")
    md = analyze(recs)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    if args.sync_repo:
        HubSync(args.sync_repo).push_files(out_dir / "summary.md")


if __name__ == "__main__":
    main()
