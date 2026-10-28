"""Is dsfsi/za-lid-bert a better text language check than our spelling model? (CPU, no new transcription)

    python scripts/lid_bert.py                      # needs results/lid_combined/records.jsonl and data/lid_texts

Uses the saved clips of lid_combined (audio probabilities + the text the model wrote). For every clip it compares,
on the same text: our spelling model (textlid.py), the BERT model, and each combined with the audio probabilities
the way the app does. Writes results/lid_bert/summary.md. The BERT model was trained on Vuk'uzenzele government
text (CC BY-SA 4.0), which none of the test corpora come from.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from sa_asr.config import PROJECT_ROOT
from sa_asr.inference import TEXT_WEIGHT
from sa_asr.textlid import TextLID, combine

MODEL = "dsfsi/za-lid-bert"
CODES = {"af": "afr"}


def bert_posteriors(texts: list[str], batch: int = 32) -> list[dict[str, float]]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    names = [CODES.get(model.config.id2label[i], model.config.id2label[i]) for i in range(model.config.num_labels)]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=128, return_tensors="pt")
            p = model(**enc).logits.softmax(-1)
            out += [dict(zip(names, row.tolist())) for row in p]
    return out


def top(d: dict[str, float]) -> str:
    return max(d, key=d.get)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default="results/lid_combined/records.jsonl")
    ap.add_argument("--texts-dir", default="data/lid_texts")
    ap.add_argument("--output-dir", default="results/lid_bert")
    args = ap.parse_args()
    root = PROJECT_ROOT

    recs = [json.loads(x) for x in (root / args.records).read_text(encoding="utf-8").splitlines()]
    recs = [r for r in recs if r["cond"] == "full" and r["text"].strip()]
    spell = TextLID.train({p.stem: [x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
                           for p in sorted((root / args.texts_dir).glob("*.txt"))})
    bert = bert_posteriors([r["text"] for r in recs])

    methods = {
        "audio only": lambda r, b: r["audio"],
        "spelling model": lambda r, b: spell.posterior(r["text"]),
        "BERT": lambda r, b: b,
        "audio + spelling (app)": lambda r, b: combine(r["audio"], spell.posterior(r["text"]), TEXT_WEIGHT, len(r["text"])),
        "audio + BERT": lambda r, b: combine(r["audio"], b, TEXT_WEIGHT, len(r["text"])),
        "audio + spelling + BERT": lambda r, b: combine(
            combine(r["audio"], spell.posterior(r["text"]), TEXT_WEIGHT, len(r["text"])), b, 0.5, len(r["text"])),
    }
    langs = sorted({r["lang"] for r in recs})
    hits: dict[str, dict[str, list[int]]] = {m: defaultdict(list) for m in methods}
    for r, b in zip(recs, bert):
        for m, f in methods.items():
            hits[m][r["lang"]].append(top(f(r, b)) == r["lang"])

    def pct(v):
        return f"{100 * sum(v) / len(v):.1f}%" if v else "-"

    lines = [f"# Text language check: {MODEL} against our spelling model", "",
             f"{len(recs)} saved clips (whole clips). Accuracy of the top language.", "",
             "| method | all clips | mean over languages | " + " | ".join(langs) + " |",
             "|---|---|---|" + "---|" * len(langs)]
    for m in methods:
        allv = [x for v in hits[m].values() for x in v]
        means = [sum(hits[m][c]) / len(hits[m][c]) for c in langs if hits[m][c]]
        lines.append(f"| {m} | {pct(allv)} | {100 * sum(means) / len(means):.1f}% | "
                     + " | ".join(pct(hits[m][c]) for c in langs) + " |")
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
