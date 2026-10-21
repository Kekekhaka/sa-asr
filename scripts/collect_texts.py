"""Collect training transcripts per language for the spelling-based language identifier.

    python scripts/collect_texts.py --sync-repo <user>/sa-asr-results

Reads only the text columns of the training split of every corpus in the round-3 mix (no audio),
de-duplicates, caps each language, and writes data/lid_texts/<code>.txt (one sentence per line).
Test sets are never read. Then:

    python scripts/train_text_lid.py --texts-dir data/lid_texts --predictions results/baseline/predictions/sa-turbo-lora-v3 --holdout
    python scripts/train_text_lid.py --texts-dir data/lid_texts --predictions results/baseline/predictions/sa-turbo-lora-v3
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from pathlib import Path

from sa_asr.config import PROJECT_ROOT, load_languages, load_yaml
from sa_asr.data import CORPORA, read_texts
from sa_asr.hub import HubSync
from sa_asr.textlid import normalize

log = logging.getLogger("collect_texts")
TAGS = re.compile(r"\[[^\]]*\]|<[^>]*>")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train_lora_v3.yaml")
    ap.add_argument("--output-dir", default="data/lid_texts")
    ap.add_argument("--max-per-language", type=int, default=20000)
    ap.add_argument("--sync-repo")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    corpora = [c for c in load_yaml(PROJECT_ROOT / args.config)["corpora"] if c in CORPORA]
    rng = random.Random(0)
    written = []
    for code, lang in load_languages().items():
        seen: dict[str, str] = {}
        for corpus in corpora:
            if corpus not in lang.corpora:
                continue
            try:
                texts = read_texts(corpus, lang, "train")
            except Exception as e:  # one unreachable corpus must not lose the rest
                log.warning("%s/%s skipped: %s", corpus, code, e)
                continue
            for t in texts:
                t = " ".join(TAGS.sub(" ", t).split())
                if len(t) >= 8:
                    seen.setdefault(normalize(t), t)
            log.info("%s/%s: %d texts (%d distinct so far)", corpus, code, len(texts), len(seen))
        sentences = list(seen.values())
        rng.shuffle(sentences)
        sentences = sentences[: args.max_per_language]
        if sentences:
            path = out / f"{code}.txt"
            path.write_text("\n".join(sentences) + "\n", encoding="utf-8")
            written.append(path)
    log.info("Wrote %d files to %s", len(written), out)
    if args.sync_repo and written:
        HubSync(args.sync_repo).push_files(*written)


if __name__ == "__main__":
    main()
