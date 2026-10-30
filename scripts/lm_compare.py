"""Which text model picks Whisper's best candidate better: MzansiLM or MORENA? (CPU; no transcription)

    python scripts/lm_compare.py                                   # vamboai/morena-0.5b-mini
    python scripts/lm_compare.py --lm vamboai/morena-1.5b-base

Uses the 5-candidate lists saved by lm_rescore.py (results/lm_rescore/nbest/, which already hold MzansiLM's
scores) for the languages MORENA covers: isiZulu, isiXhosa, Setswana, isiNdebele, Afrikaans and English. Each
candidate is scored by the new model; then, as in lm_rescore.py, the weight of each text model is tuned on the
even-numbered clips and measured on the odd-numbered ones: MzansiLM alone, the new model alone, and both together.
MORENA (Vambo AI) is Apache 2.0. Writes results/lm_compare/<model>/summary.md; scores are cached next to it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from sa_asr.config import PROJECT_ROOT, load_languages

COVERED = {"zul", "xho", "tsn", "nbl", "afr", "eng"}
LAMBDAS = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def load_morena(model_id: str):
    """MORENA is not a transformers architecture: its repo ships modeling_morena.py (a training script whose
    Transformer class is a Llama-style decoder) and a tokenizer.json. Loaded as in the repo's load_example.py;
    importing the module only defines classes (its training code runs under __main__ only)."""
    import importlib.util

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    path = Path(model_id) if Path(model_id).is_dir() else Path(
        snapshot_download(model_id, allow_patterns=["*.json", "*.py", "*.safetensors"]))
    spec = importlib.util.spec_from_file_location("modeling_morena", path / "modeling_morena.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modeling_morena"] = mod  # its dataclasses look their module up while being defined
    spec.loader.exec_module(mod)
    cfg = json.loads((path / "config.json").read_text())
    model = mod.Transformer(mod.ModelConfig(**cfg["model"]), "sdpa")
    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    eos = tok.token_to_id("<eos>")

    def logits(x):  # (B, T) right-padded token ids -> (B, T, vocab); causal, so padding never leaks back
        pos = torch.arange(x.shape[1]).repeat(x.shape[0], 1)
        return model(x, pos, None, x.shape[1], None)

    return (lambda t: tok.encode(t, add_special_tokens=False).ids), logits, eos, eos, eos  # docs start with <eos>


def load_hf(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return (lambda t: tok(t, add_special_tokens=False).input_ids), (lambda x: lm(input_ids=x).logits), start, \
        tok.eos_token_id, pad


@torch.inference_mode()
def score_texts(model_id: str, texts: list[str], batch: int) -> list[float]:
    """Total log-probability of each text, from a start token through an end-of-text token."""
    torch.set_num_threads(max(torch.get_num_threads(), 4))
    encode, forward, start, end, pad = load_morena(model_id) if "morena" in model_id else load_hf(model_id)
    out: list[float] = []
    t0 = time.time()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths per batch: less padding
    results = [0.0] * len(texts)
    for i in range(0, len(order), batch):
        idx = order[i : i + batch]
        ids = [[start] + encode(texts[j]) + [end] for j in idx]
        n = max(map(len, ids))
        x = torch.tensor([s + [pad] * (n - len(s)) for s in ids])
        mask = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in ids])
        logits = forward(x)[:, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(2, x[:, 1:].unsqueeze(-1)).squeeze(-1) * mask[:, 1:]
        for j, v in zip(idx, lp.sum(1).tolist()):
            results[j] = v
        if (i // batch) % 50 == 0:
            print(f"  {i + len(ids)}/{len(texts)} texts, {time.time() - t0:.0f} s", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lm", default="vamboai/morena-0.5b-mini", help="a Hub id, or a local folder with its files")
    ap.add_argument("--nbest-dir", default="results/lm_rescore/nbest")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    names = {c: lang.name for c, lang in load_languages().items()}
    short = Path(args.lm).name  # a Hub id or a local folder (Windows paths included) -> "morena-0.5b-mini"
    out_dir = PROJECT_ROOT / "results" / "lm_compare" / short
    out_dir.mkdir(parents=True, exist_ok=True)

    sets: dict[str, list[dict]] = {}
    for f in sorted((PROJECT_ROOT / args.nbest_dir).glob("*.jsonl")):
        if f.stem.rsplit("_", 1)[1] in COVERED:
            sets[f.stem] = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    cache = out_dir / "scores.json"
    scores: dict[str, list[list[float]]] = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [k for k in sets if k not in scores]
    if todo:
        flat = [c["text"] for k in todo for r in sets[k] for c in r["cands"]]
        print(f"Scoring {len(flat)} candidates from {len(todo)} sets with {args.lm}", flush=True)
        vals = iter(score_texts(args.lm, flat, args.batch))
        for k in todo:
            scores[k] = [[next(vals) for _ in r["cands"]] for r in sets[k]]
        cache.write_text(json.dumps(scores))
    for k, rows in sets.items():
        for r, s in zip(rows, scores[k]):
            for c, v in zip(r["cands"], s):
                c["lm_new"] = v

    def pick(r, a, b):  # asr + a * MzansiLM + b * new model
        return max(r["cands"], key=lambda c: c["asr"] + a * c["lm_raw"] + b * c["lm_new"])

    def wer(rows, a, b):
        return 100 * sum(pick(r, a, b)["err"] for r in rows) / max(sum(r["ref_words"] for r in rows), 1)

    dev = [r for rows in sets.values() for r in rows[0::2]]
    test = {k: rows[1::2] for k, rows in sets.items()}
    all_test = [r for rows in test.values() for r in rows]
    best_mz = min(LAMBDAS, key=lambda a: wer(dev, a, 0))
    best_new = min(LAMBDAS, key=lambda b: wer(dev, 0, b))
    best_both = min(((a, b) for a in LAMBDAS for b in LAMBDAS), key=lambda ab: wer(dev, *ab))
    oracle = lambda rows: 100 * sum(min(c["err"] for c in r["cands"]) for r in rows) / sum(r["ref_words"] for r in rows)  # noqa: E731
    configs = [("Whisper's top candidate", 0, 0), (f"MzansiLM (weight {best_mz})", best_mz, 0),
               (f"{short} (weight {best_new})", 0, best_new),
               (f"both (weights {best_both[0]} / {best_both[1]})", *best_both)]
    lines = [f"# Rescoring Whisper's 5 candidates: MzansiLM against {short}", "",
             f"{len(sets)} test sets ({', '.join(sorted({names[k.rsplit('_', 1)[1]] for k in sets}))}), "
             f"{sum(len(v) for v in sets.values())} clips. Weights tuned on even-numbered clips, measured on the "
             "odd-numbered ones. WER %, lower is better.", "",
             "| | held-out WER |", "|---|---|"]
    lines += [f"| {n} | {wer(all_test, a, b):.2f} |" for n, a, b in configs]
    lines += [f"| oracle (best of 5) | {oracle(all_test):.2f} |", "",
              "## Per set (held-out half)", "",
              "| set | clips | top | MzansiLM | " + short + " | both | oracle |",
              "|---|---|---|---|---|---|---|"]
    for k, rows in test.items():
        corpus, code = k.rsplit("_", 1)
        lines.append(f"| {corpus} / {names.get(code, code)} | {len(rows)} | "
                     + " | ".join(f"{wer(rows, a, b):.1f}" for _, a, b in configs) + f" | {oracle(rows):.1f} |")
    md = "\n".join(lines)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
