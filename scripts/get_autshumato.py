"""Download the Autshumato monolingual text corpora (CC BY 4.0, SADiLaR) for the spelling-based language check.

    python scripts/get_autshumato.py                    # -> data/autshumato/<lang>/...  (all 11 languages)
    python scripts/get_autshumato.py --list             # only show the files

Source: "Datasets for South African Languages: Bilingual Aligned and Monolingual Data for Machine Translation",
Journal of Open Humanities Data (2025), doi:10.5334/johd.372. The corpora are government text (one segment per
line), held in the SADiLaR repository (DSpace 7); files are fetched through its REST API.
"""

from __future__ import annotations

import argparse
import re

import requests

import sa_asr  # noqa: F401  (applies the corporate-certificate fix before any HF call)
from sa_asr.config import PROJECT_ROOT

REPO = "https://repo.sadilar.org"
HANDLES = {
    "afr": "20.500.12185/580", "eng": "20.500.12185/686", "nbl": "20.500.12185/573", "xho": "20.500.12185/692",
    "zul": "20.500.12185/581", "nso": "20.500.12185/582", "sot": "20.500.12185/583", "tsn": "20.500.12185/584",
    "ssw": "20.500.12185/559.2", "ven": "20.500.12185/681", "tso": "20.500.12185/570",
}


def item_uuid(handle: str) -> str:
    """The landing page of a handle redirects to /items/<uuid>."""
    r = requests.get(f"{REPO}/handle/{handle}", timeout=120)
    r.raise_for_status()
    m = re.search(r"/items/([0-9a-f-]{36})", r.url) or re.search(r"/items/([0-9a-f-]{36})", r.text)
    if not m:
        raise RuntimeError(f"no item for handle {handle}")
    return m.group(1)


def files(uuid: str) -> list[tuple[str, str, int]]:
    """(name, download url, bytes) of the item's ORIGINAL bundle."""
    out = []
    bundles = requests.get(f"{REPO}/server/api/core/items/{uuid}/bundles", timeout=120).json()
    for b in bundles["_embedded"]["bundles"]:
        if b["name"] != "ORIGINAL":
            continue
        bits = requests.get(b["_links"]["bitstreams"]["href"], params={"size": 100}, timeout=120).json()
        for f in bits["_embedded"]["bitstreams"]:
            out.append((f["name"], f"{REPO}/server/api/core/bitstreams/{f['uuid']}/content", f["sizeBytes"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--langs", nargs="*", default=list(HANDLES))
    ap.add_argument("--output-dir", default="data/autshumato")
    args = ap.parse_args()
    for lang in args.langs:
        found = files(item_uuid(HANDLES[lang]))
        for name, url, size in found:
            print(f"{lang}: {name} ({size / 1e6:.1f} MB)")
            if args.list:
                continue
            out = PROJECT_ROOT / args.output_dir / lang / name
            if out.exists() and out.stat().st_size == size:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(out, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)


if __name__ == "__main__":
    main()
