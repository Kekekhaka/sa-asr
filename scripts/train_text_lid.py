"""Train the spelling-based language identifier (src/sa_asr/textlid.py).

    python scripts/train_text_lid.py                      # from the saved evaluation transcripts
    python scripts/train_text_lid.py --texts-dir corpus/  # plus <code>.txt files, one sentence per line

Uses the *reference* sentences in results/baseline/predictions/<model>/*.jsonl (pulled from the
results repo by the evaluation stages), labelled by the language in each file name. Writes
models/text_lid.json.gz, which the app loads when present.

    --holdout   train on half of the distinct sentences and report accuracy on the other half
                (using the model's own transcripts as the text to classify), instead of writing a file.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

from sa_asr.config import PROJECT_ROOT
from sa_asr.lid import accuracy
from sa_asr.textlid import DEFAULT_PATH, TextLID, normalize


def read_predictions(pred_dir: Path) -> list[tuple[str, str, str]]:
    """(language code, reference, hypothesis) from every predictions file of one model."""
    rows = []
    for f in sorted(glob.glob(str(pred_dir / "*.jsonl"))):
        name = Path(f).name.split("_test_")[0]
        code = "eng" if name == "nchlt_eng_eng" else name.rsplit("_", 1)[1]
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            if "reference" in d and "hypothesis" in d:
                rows.append((code, d["reference"], d["hypothesis"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default="results/baseline/predictions/sa-turbo-lora-v2")
    ap.add_argument("--texts-dir", help="Folder of <language code>.txt files to train on as well")
    ap.add_argument("--output", default=str(DEFAULT_PATH))
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()

    pred_dir = Path(args.predictions)
    pred_dir = pred_dir if pred_dir.is_absolute() else PROJECT_ROOT / pred_dir
    rows = read_predictions(pred_dir)
    if not rows:
        sys.exit(f"No predictions found in {pred_dir}; run the compare stage or pull the results repo first")
    extra: dict[str, list[str]] = {}
    if args.texts_dir:
        for p in Path(args.texts_dir).glob("*.txt"):
            extra[p.stem] = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    if args.holdout:
        sentences = sorted({normalize(r[1]) for r in rows})
        random.Random(0).shuffle(sentences)
        test_s = set(sentences[: len(sentences) // 2])
        by_lang: dict[str, list[str]] = {}
        for code, ref, _ in rows:
            if normalize(ref) not in test_s:
                by_lang.setdefault(code, []).append(ref)
        model = TextLID.train({c: by_lang.get(c, []) + extra.get(c, []) for c in set(by_lang) | set(extra)})
        recs = [{"lang": code, "pred": max((s := model.scores(hyp)), key=s.get)}
                for code, ref, hyp in rows if normalize(ref) in test_s]
        print(accuracy(recs))
        return

    by_lang = {}
    for code, ref, _ in rows:
        by_lang.setdefault(code, []).append(ref)
    for code, texts in extra.items():
        by_lang.setdefault(code, []).extend(texts)
    model = TextLID.train(by_lang)
    out = Path(args.output)
    model.save(out)
    print(f"Wrote {out} ({out.stat().st_size / 1e3:.0f} kB): " + ", ".join(f"{c} {len(t)}" for c, t in sorted(by_lang.items())))


if __name__ == "__main__":
    main()
