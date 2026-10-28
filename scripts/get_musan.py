"""Download MUSAN's noise recordings (OpenSLR 17, CC BY 4.0) for training augmentation.

    python scripts/get_musan.py                 # -> data/musan/musan/noise (about 930 files, ~1 GB)

The archive is 11 GB with music/, noise/ and speech/; it is streamed and only noise/ is written to disk, so this
takes about 10-20 minutes on Kaggle and needs ~1 GB of space. Does nothing if the folder is already filled.
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request

from sa_asr.config import PROJECT_ROOT

URL = "https://www.openslr.org/resources/17/musan.tar.gz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="data/musan")
    ap.add_argument("--part", default="noise", choices=["noise", "music", "speech"])
    args = ap.parse_args()
    out = PROJECT_ROOT / args.output_dir
    target = out / "musan" / args.part
    if target.exists() and any(target.rglob("*.wav")):
        print(f"Already there: {target}")
        return
    prefix = f"musan/{args.part}/"
    n = 0
    with urllib.request.urlopen(URL) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if member.name.startswith(prefix) and (member.isfile() or member.isdir()):
                tar.extract(member, out, filter="data")
                n += member.isfile() and member.name.endswith(".wav")
                if n and n % 200 == 0 and member.isfile():
                    print(f"{n} files", flush=True)
    if not n:
        raise SystemExit(f"No {prefix} files found in {URL}")
    print(f"Saved {n} .wav files to {target}")


if __name__ == "__main__":
    main()
