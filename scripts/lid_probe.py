"""A dedicated language classifier on Whisper's encoder features.

    python scripts/lid_probe.py --model KekeCode/sa-asr-whisper-lora-v3 --sync-repo <user>/sa-asr-results

Our current detector reads the language off Whisper's first decoding step, which was trained with one extra
loss term. This trains a classifier on the encoder's own representation of the audio instead: mean-pooled
encoder activations (three layers) of random 2.5-10 s crops of labelled *training* clips of all 11 languages,
fed to a softmax regression. It is then scored on the same cached test clips as evaluate_lid (whole clip, first
5 s, first 3 s) and combined with the spelling model exactly as the app does.

Step 1 (GPU, slow part): embed training crops -> results/lid_probe/train_embeddings.npz (reused on re-runs).
Step 2 (GPU): embed the test clips -> results/lid_probe/test_embeddings.npz (reused).
Step 3 (CPU): fit, evaluate, write results/lid_probe/summary.md and models/lid_probe.npz.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from sa_asr.config import PROJECT_ROOT, load_languages, load_yaml, quiet_progress_bars, select_languages
from sa_asr.hub import HubSync
from sa_asr.lid import family_of
from sa_asr.textlid import TextLID, combine

log = logging.getLogger("lid_probe")
TRAIN_CORPORA = ["nchlt", "swivuriso", "lwazi", "fleurs", "nchlt_eng", "openslr32"]
LAYERS = (16, 24, 32)
FRAMES_PER_SECOND = 50
CONDS = {"full": None, "5s": 5.0, "3s": 3.0}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


@torch.inference_mode()
def embed(whisper, wavs: list[np.ndarray]) -> np.ndarray:
    """(len(wavs), 3 * 1280): per clip, the mean over its real (unpadded) frames of three encoder layers."""
    from sa_asr.audio import SAMPLE_RATE

    feats = whisper.processor.feature_extractor([w[: 30 * SAMPLE_RATE] for w in wavs], sampling_rate=SAMPLE_RATE,
                                                return_tensors="pt")
    out = whisper.model.model.encoder(feats.input_features.to(whisper.device, whisper.dtype), output_hidden_states=True)
    rows = []
    for i, w in enumerate(wavs):
        n = int(min(1500, max(1, math.ceil(min(len(w), 30 * SAMPLE_RATE) / SAMPLE_RATE * FRAMES_PER_SECOND))))
        rows.append(torch.cat([out.hidden_states[l][i, :n].float().mean(0) for l in LAYERS]).cpu().numpy())
    return np.stack(rows)


def training_embeddings(whisper, languages, per_language: int, seed: int, path: Path) -> dict:
    from sa_asr.audio import SAMPLE_RATE, decode_audio
    from sa_asr.data import stream_corpus

    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    rng = random.Random(seed)
    X, y, src = [], [], []
    for lang in languages.values():
        corpora = [c for c in TRAIN_CORPORA if c in lang.corpora]
        if not corpora:
            continue
        quota = per_language // len(corpora)
        for corpus in corpora:
            t0, wavs, kept = time.time(), [], 0
            try:
                ds = stream_corpus(corpus, lang, "train", max_duration_s=30.0, seed=seed).shuffle(seed=seed, buffer_size=32)
                for row in ds:
                    wav = decode_audio(row["audio"])
                    if len(wav) < 2 * SAMPLE_RATE:
                        continue
                    n = int(rng.uniform(2.5, 10.0) * SAMPLE_RATE)
                    start = rng.randint(0, max(len(wav) - n, 0))
                    wavs.append(wav[start : start + n])
                    if len(wavs) == 16 or len(wavs) + kept >= quota:
                        X.append(embed(whisper, wavs)); kept += len(wavs)
                        y += [lang.code] * len(wavs); src += [corpus] * len(wavs); wavs = []
                    if kept >= quota:
                        break
                if wavs:
                    X.append(embed(whisper, wavs)); kept += len(wavs)
                    y += [lang.code] * len(wavs); src += [corpus] * len(wavs)
            except Exception as e:  # one unreachable corpus must not lose the rest
                log.warning("%s/%s stopped after %d clips: %s", corpus, lang.code, kept, e)
            log.info("%s/%s: %d training crops in %.0f s", corpus, lang.code, kept, time.time() - t0)
    data = {"X": np.concatenate(X), "y": np.array(y), "src": np.array(src)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return data


def test_embeddings(whisper, cfg, languages, cache_dir: Path, max_samples: int, path: Path) -> dict:
    from sa_asr.audio import SAMPLE_RATE
    from sa_asr.data import build_eval_set, corpus_languages, eval_row_audio

    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    seed, max_dur, buffer = cfg.get("seed", 42), cfg.get("max_duration_s", 30.0), cfg.get("shuffle_buffer", 500)
    X, meta = [], []
    for corpus, codes in cfg["eval_sets"].items():
        for lang in corpus_languages(corpus, select_languages(languages, codes)):
            rows = build_eval_set(corpus, lang, cfg.get("split", "test"), cache_dir, max_samples=max_samples, seed=seed,
                                  max_duration_s=max_dur, shuffle_buffer=buffer)
            for start in range(0, len(rows), 16):
                batch = rows[start : start + 16]
                wavs = [eval_row_audio({"flac": b}) for b in batch["flac"]]
                for cond, crop in CONDS.items():
                    clips = wavs if crop is None else [w[: int(crop * SAMPLE_RATE)] for w in wavs]
                    X.append(embed(whisper, clips))
                    meta += [(corpus, lang.code, cond, str(u)) for u in batch["utt_id"]]
            log.info("test %s/%s done", corpus, lang.code)
    data = {"X": np.concatenate(X), "meta": np.array(meta)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return data


def fit_softmax(X: np.ndarray, y: np.ndarray, n_classes: int, wd: float):
    W = torch.zeros(X.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    opt = torch.optim.LBFGS([W, b], max_iter=300)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt) + wd * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def analyze(train: dict, test: dict, out_dir: Path, texts_dir: Path, records_path: Path, languages: dict) -> str:
    codes = sorted(set(train["y"]))
    cidx = {c: i for i, c in enumerate(codes)}
    mu, sd = train["X"].mean(0), train["X"].std(0) + 1e-6
    ytr = np.array([cidx[c] for c in train["y"]])
    Xtr = ((train["X"] - mu) / sd).astype(np.float32)
    # weight decay chosen on a held-out tenth of the training crops (never on test clips)
    order = np.random.RandomState(0).permutation(len(Xtr))
    val, fit = order[: len(order) // 10], order[len(order) // 10 :]
    best_wd, best_acc = None, -1.0
    for wd in (1e-4, 1e-3, 1e-2, 1e-1):
        W, b = fit_softmax(Xtr[fit], ytr[fit], len(codes), wd)
        a = float(((torch.tensor(Xtr[val]) @ W + b).argmax(1).numpy() == ytr[val]).mean())
        log.info("weight decay %g: validation accuracy %.3f", wd, a)
        if a > best_acc:
            best_wd, best_acc = wd, a
    W, b = fit_softmax(Xtr, ytr, len(codes), best_wd)
    (PROJECT_ROOT / "models").mkdir(exist_ok=True)
    np.savez(PROJECT_ROOT / "models" / "lid_probe.npz", W=W.numpy(), b=b.numpy(), mu=mu, sd=sd, codes=np.array(codes),
             layers=np.array(LAYERS))

    meta = test["meta"]
    Xte = ((test["X"] - mu) / sd).astype(np.float32)
    probs = torch.softmax(torch.tensor(Xte) @ W + b, dim=1).numpy()
    texts = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        texts[(r["corpus"], r["lang"], r["cond"], str(r["utt_id"]))] = (r["text"], r["audio"])
    tm = TextLID.train({p.stem: [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
                        for p in sorted(texts_dir.glob("*.txt"))})

    rows = []
    for i, (corpus, lang, cond, utt) in enumerate(meta):
        p_probe = {c: float(probs[i, cidx[c]]) for c in codes}
        text, p_audio = texts.get((corpus, lang, cond, utt), ("", None))
        rows.append({"corpus": corpus, "lang": lang, "cond": cond, "probe": p_probe, "audio": p_audio, "text": text})
    def top(p):
        return max(p, key=p.get)

    def pred(r, kind, w):
        if kind == "probe":
            return top(r["probe"])
        if kind == "audio":
            return top(r["audio"])
        if "post" not in r:
            r["post"] = tm.posterior(r["text"])
        base = r["probe"] if kind == "probe+text" else r["audio"]
        return next(iter(combine(base, r["post"], w, len(r["text"]))))
    lines = ["# Encoder-feature language classifier", "",
             f"{len(train['y'])} training crops ({len(codes)} languages), weight decay {best_wd:g} "
             f"(validation accuracy {best_acc:.3f}); {len(rows)} test clip-conditions.", "",
             "## Exact-language accuracy on the test clips (%)", "",
             "| detector | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    for name, kind, w in (("current: decoder-token detector (audio only)", "audio", 0), ("encoder probe (audio only)", "probe", 0),
                          ("current audio + spelling (weight 1)", "audio+text", 1.0),
                          ("probe + spelling (weight 1)", "probe+text", 1.0),
                          ("probe + spelling (weight 2)", "probe+text", 2.0)):
        cells = []
        for cond in CONDS:
            # transcripts (and the current detector's probabilities) exist only for the whole clip and the first 5 s
            sub = [r for r in rows if r["cond"] == cond and (kind == "probe" or r["audio"] is not None)]
            cells.append(f"{100 * np.mean([pred(r, kind, w) == r['lang'] for r in sub]):.1f}" if sub else "n/a")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Per language, whole clip: current audio+spelling vs probe+spelling", "", "| language | current | probe+spelling | probe only |", "|---|---|---|---|"]
    for c in codes:
        sub = [r for r in rows if r["cond"] == "full" and r["lang"] == c]
        lines.append(f"| {languages[c].name if c in languages else c} | "
                     f"{100 * np.mean([pred(r, 'audio+text', 1.0) == c for r in sub]):.0f}% | "
                     f"{100 * np.mean([pred(r, 'probe+text', 1.0) == c for r in sub]):.0f}% | "
                     f"{100 * np.mean([pred(r, 'probe', 0) == c for r in sub]):.0f}% |")
    lines += ["", "## Per corpus, whole clip (probe + spelling)", "", "| corpus | current | probe+spelling |", "|---|---|---|"]
    for corpus in sorted({r["corpus"] for r in rows}):
        sub = [r for r in rows if r["cond"] == "full" and r["corpus"] == corpus]
        lines.append(f"| {corpus} | {100 * np.mean([pred(r, 'audio+text', 1.0) == r['lang'] for r in sub]):.1f}% | "
                     f"{100 * np.mean([pred(r, 'probe+text', 1.0) == r['lang'] for r in sub]):.1f}% |")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    quiet_progress_bars()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eval_baseline.yaml")
    ap.add_argument("--model", default="KekeCode/sa-asr-whisper-lora-v3")
    ap.add_argument("--per-language", type=int, default=600)
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output-dir", default="results/lid_probe")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--sync-repo")
    args = ap.parse_args()

    languages = load_languages()
    out_dir = resolve(args.output_dir)
    cfg = load_yaml(args.config)
    cache_dir = resolve(cfg.get("cache_dir", "data/eval_cache"))
    texts_dir, records = resolve("data/lid_texts"), resolve("results/lid_combined/records.jsonl")
    sync = HubSync(args.sync_repo) if args.sync_repo else None
    if sync:
        sync.pull(Path("results/lid_probe"), Path("results/lid_combined"), Path("data/lid_texts"))
        if not args.analyze_only:
            sync.pull(cache_dir)
    train_path, test_path = out_dir / "train_embeddings.npz", out_dir / "test_embeddings.npz"
    if not (train_path.exists() and test_path.exists()):
        from sa_asr.inference import load_transcriber, resolve_model

        model_ref, trained = resolve_model(args.model)
        whisper = load_transcriber(model_ref, finetuned=True)
        in_scope = {c: l for c, l in languages.items() if c in trained}
        train = training_embeddings(whisper, in_scope, args.per_language, args.seed, train_path)
        if sync:
            sync.push_files(train_path)
        test = test_embeddings(whisper, cfg, languages, cache_dir, args.max_samples, test_path)
        if sync:
            sync.push_files(test_path)
    else:
        train, test = (dict(np.load(p, allow_pickle=True)) for p in (train_path, test_path))
    md = analyze(train, test, out_dir, texts_dir, records, languages)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    if sync:
        sync.push_files(out_dir / "summary.md", PROJECT_ROOT / "models" / "lid_probe.npz")


if __name__ == "__main__":
    main()
