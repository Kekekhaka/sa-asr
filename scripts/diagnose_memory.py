"""Measure what the training data pipeline costs in RAM, without loading a model.

    python scripts/diagnose_memory.py --seconds 150
    python scripts/diagnose_memory.py --seconds 90 --corpora swivuriso --trace

1. Prints each corpus's Parquet layout (rows per row group, MB per row group, bytes per
   row) for the first training file of each language.
2. Streams the same mixed training stream as train.py (same shuffle buffer, duration
   filter, augmentation, feature extraction) and prints RAM every 15 s. Stops early,
   cleanly, if free RAM drops below --min-free-gb, instead of being killed by the OS.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

import psutil
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from transformers import WhisperProcessor

from sa_asr.config import PROJECT_ROOT, load_languages, load_yaml, select_languages
from sa_asr.data import CORPORA, check_access
from sa_asr.training import LabelBuilder, build_train_stream, make_example_fn, memory_stats


def parquet_layout(corpus: str, lang_value: str) -> str:
    spec = CORPORA[corpus]
    fs = HfFileSystem()
    root = f"datasets/{spec.repo}"
    split = spec.splits["train"]
    pattern = f"{root}/**/{split}-*.parquet"
    files = sorted(f for f in fs.glob(pattern) if spec.language_column or f"/{lang_value}/" in f)
    if not files:
        return "no train files found"
    with fs.open(files[0], block_size=2**16) as fh:
        md = pq.ParquetFile(fh).metadata
    groups = [md.row_group(i) for i in range(md.num_row_groups)]
    mb = max(g.total_byte_size for g in groups) / 2**20
    size = fs.info(files[0])["size"]
    return (
        f"{files[0].split('/')[-1]}: {md.num_rows} rows, {md.num_row_groups} row groups, "
        f"{groups[0].num_rows} rows/group, max {mb:.0f} MB/group uncompressed, "
        f"file {size / 2**20:.0f} MB, {size / max(md.num_rows, 1) / 2**20:.2f} MB/row"
    )


def main():
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train_lora.yaml")
    ap.add_argument("--langs", nargs="*", default=["zul", "xho"])
    ap.add_argument("--seconds", type=int, default=150)
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--corpora", nargs="*", help="Only these corpora (default: all with weight > 0)")
    ap.add_argument("--trace", action="store_true", help="Track Python allocations; print the biggest holders at the end")
    args = ap.parse_args()
    if args.trace:
        import tracemalloc

        tracemalloc.start(8)

    cfg = load_yaml(PROJECT_ROOT / args.config)
    registry = load_languages()
    languages = select_languages(registry, args.langs)
    corpora = [c for c, w in cfg["corpora"].items() if w and w > 0 and check_access(c)]
    if args.corpora:
        corpora = [c for c in corpora if c in args.corpora]

    print("== Parquet layout (first training file per corpus/language)", flush=True)
    for corpus in corpora:
        for lang in languages:
            if corpus in lang.corpora:
                try:
                    print(f"  {corpus}/{lang.code}: {parquet_layout(corpus, lang.corpora[corpus])}", flush=True)
                except Exception as e:  # diagnostics must not stop the measurement below
                    print(f"  {corpus}/{lang.code}: could not read layout ({e})", flush=True)

    streams = [(c, lang, 1.0) for lang in languages for c in corpora if c in lang.corpora]
    processor = WhisperProcessor.from_pretrained(cfg["base_model"])
    labels = LabelBuilder(processor, languages)
    aug = cfg.get("augment") or {}
    max_dur = cfg.get("max_duration_s", 30.0)
    fn = make_example_fn(processor, labels, max_dur, cfg.get("max_label_tokens", 440),
                         aug.get("telephone_prob", 0.0), aug.get("noise_prob", 0.0), seed=42)
    ds = build_train_stream(streams, fn, seed=42, shuffle_buffer=cfg.get("shuffle_buffer", 64), max_duration_s=max_dur)

    import os

    import datasets

    print(f"== Streaming {[(c, l.code) for c, l, _ in streams]}, shuffle_buffer={cfg.get('shuffle_buffer', 64)}, "
          f"datasets {datasets.__version__}", flush=True)
    t0, count, stop = time.time(), [0], threading.Event()

    def holders():
        """Where the memory is: PyArrow's pool, Python objects in total, and the top Python allocation sites."""
        import pyarrow as pa

        print(f"  arrow_pool_gb={pa.total_allocated_bytes() / 2**30:.2f}", flush=True)
        if not args.trace:
            return
        import tracemalloc

        snap = tracemalloc.take_snapshot()
        stats = snap.statistics("traceback")
        total = sum(st.size for st in stats)
        print(f"  python_objects_gb={total / 2**30:.2f}; biggest allocation sites:", flush=True)
        for st in stats[:6]:
            print(f"    {st.size / 2**30:.2f} GB in {st.count} blocks", flush=True)
            for line in st.traceback.format()[-8:]:
                print(f"      {line}", flush=True)

    def report():
        while not stop.is_set():
            m = memory_stats()
            print(f"  t={time.time() - t0:4.0f}s examples={count[0]:5d} {m}", flush=True)
            if m["ram_free_gb"] < args.min_free_gb:
                print(f"  ABORT: free RAM below {args.min_free_gb} GB", flush=True)
                holders()
                os._exit(3)
            if time.time() - t0 > args.seconds + 120:  # the main loop only checks time per example
                print(f"  STOP: time limit reached with {count[0]} examples (stream is slow to start)", flush=True)
                holders()
                os._exit(0)
            stop.wait(15)

    threading.Thread(target=report, daemon=True).start()
    for _ in ds:
        count[0] += 1
        if time.time() - t0 > args.seconds:
            break
    stop.set()
    holders()
    print(f"== Done: {count[0]} examples in {time.time() - t0:.0f}s, final {memory_stats()}", flush=True)


if __name__ == "__main__":
    main()
