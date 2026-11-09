"""Does Autshumato government text make the spelling-based language check better? (CPU, no transcription)

    python scripts/get_autshumato.py            # first: data/autshumato/<lang>/...
    python scripts/lid_autshumato.py            # needs results/lid_combined/records.jsonl and data/lid_texts

Trains the spelling check (textlid.py) on: our current texts (data/lid_texts, from the speech corpora); current +
Autshumato lines (20,000 or 100,000 per language); and Autshumato alone. Each is scored on the saved clips of
lid_combined (the text our model wrote for each clip), on its own and combined with the audio the way the app
does. English has no Autshumato data here (it is already right on 97-100% of clips), so it keeps its current text.
Writes results/lid_autshumato/summary.md.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from sa_asr.config import PROJECT_ROOT, load_languages
from sa_asr.inference import TEXT_WEIGHT
from sa_asr.textlid import TextLID, combine

MIN_CHARS = 20


def autshumato_lines(folder: Path) -> list[str]:
    """Every line of every .txt in the folder, inside zips too."""
    lines: list[str] = []

    def add(raw: bytes) -> None:
        text = raw.decode("utf-8", errors="replace").lstrip("﻿")
        lines.extend(x.strip() for x in text.splitlines() if len(x.strip()) >= MIN_CHARS)

    for p in sorted(folder.glob("*")):
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                for name in z.namelist():
                    if name.lower().endswith(".txt") and "readme" not in name.lower():
                        add(z.read(name))
        elif p.suffix.lower() == ".txt" and "readme" not in p.name.lower():
            add(p.read_bytes())
    return lines


def top(d: dict[str, float]) -> str:
    return max(d, key=d.get)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default="results/lid_combined/records.jsonl")
    ap.add_argument("--texts-dir", default="data/lid_texts")
    ap.add_argument("--autshumato-dir", default="data/autshumato")
    ap.add_argument("--output-dir", default="results/lid_autshumato")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    names = {c: lang.name for c, lang in load_languages().items()}
    rng = random.Random(args.seed)

    current = {p.stem: [x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
               for p in sorted((PROJECT_ROOT / args.texts_dir).glob("*.txt"))}
    auts = {}
    for d in sorted((PROJECT_ROOT / args.autshumato_dir).glob("*")):
        if d.is_dir():
            lines = autshumato_lines(d)
            rng.shuffle(lines)
            auts[d.name] = lines
            print(f"{d.name}: {len(lines):,} Autshumato lines (current texts: {len(current.get(d.name, [])):,})")

    def mix(n: int | None, with_current: bool) -> dict[str, list[str]]:
        out = {}
        for c in current:
            extra = auts.get(c, [])[:n] if n else []
            out[c] = (current[c] if with_current or c not in auts else []) + extra
        return out

    variants = {
        "current texts (shipped)": mix(None, True),
        "current + 20k Autshumato": mix(20_000, True),
        "current + 100k Autshumato": mix(100_000, True),
        "Autshumato only (20k)": mix(20_000, False),
    }
    recs = [json.loads(x) for x in (PROJECT_ROOT / args.records).read_text(encoding="utf-8").splitlines()]
    langs = sorted({r["lang"] for r in recs})
    rows = []
    per_lang: dict[str, dict[str, dict[str, list[int]]]] = {}
    for vname, texts in variants.items():
        model = TextLID.train(texts)
        hits = {k: defaultdict(list) for k in ("text full", "audio+text full", "audio+text 5s")}
        for r in recs:
            if not r["text"].strip():
                continue
            post = model.posterior(r["text"])
            if r["cond"] == "full":
                hits["text full"][r["lang"]].append(top(post) == r["lang"])
            hits[f"audio+text {r['cond']}"][r["lang"]].append(
                top(combine(r["audio"], post, TEXT_WEIGHT, len(r["text"]))) == r["lang"])
        per_lang[vname] = hits
        row = [vname]
        for k in hits:
            allv = [x for v in hits[k].values() for x in v]
            row.append(f"{100 * sum(allv) / len(allv):.1f}%")
        rows.append(row)
        print(" | ".join(row), flush=True)

    lines = ["# Spelling-based language check with Autshumato text", "",
             f"Saved clips from lid_combined (whole clips, and their first 5 s). Autshumato lines per language: "
             + ", ".join(f"{c} {len(v):,}" for c, v in sorted(auts.items())) + ".", "",
             "| training text | text alone | audio + text (whole clips) | audio + text (first 5 s) |",
             "|---|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in rows]
    lines += ["", "## Audio + text on whole clips, by language", "",
              "| language | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
    for c in langs:
        cells = []
        for v in variants:
            h = per_lang[v]["audio+text full"][c]
            cells.append(f"{100 * sum(h) / len(h):.1f}%" if h else "-")
        lines.append(f"| {names.get(c, c)} | " + " | ".join(cells) + " |")
    out = PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
