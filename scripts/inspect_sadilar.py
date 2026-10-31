"""Look inside SADiLaR speech corpora before writing a converter for them (CPU only; Kaggle stage sadilar_inspect).

    python scripts/inspect_sadilar.py --sync-repo <user>/sa-asr-results
    python scripts/inspect_sadilar.py --only sama_nso soap ast_be --match soap=5lang.xml --sync-repo <user>/...

Downloads each corpus's archive from SADiLaR's repository (its DSpace REST API lists the files), then reports the
layout: file counts per extension and folder, sample paths, readme files, the first lines of every kind of text file
(transcriptions, segment XML, metadata; only the start of large ones), the format of a few audio files and the hours
of audio (estimated from file sizes). Archives inside an archive are opened too. Writes
results/sadilar_inspect/report.md (or report_<keys>.md with --only). The archives stay in --download-dir.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from sa_asr.config import PROJECT_ROOT
from sa_asr.hub import HubSync

API = "https://repo.sadilar.org/server/api/core"
# key -> (title, item UUID, substring a file name must contain, or None for every file; files under 1 MB always come)
ITEMS = {
    "spcs": ("SPCS Sepedi-English code-switched speech (CC BY 2.5 SA)", "9b255944-5050-47b6-a662-1d43ba0e7441", None),
    "sama_nso": ("CSIR SAMA broadcast speech, Sepedi part (CC BY 4.0)", "e44a1b0c-34c5-4957-9560-4266ca17c7d2",
                 "nso_"),
    "multipron": ("Multipron proper names (CC BY 3.0)", "396b4c02-4ee9-4996-b081-0e1a1aa88d1f", None),
    # spontaneous code-switched speech (zul, xho, tsn, sot + eng); licence "research only": test set, private models
    # (its balanced_* files are subsets of 5lang)
    "soap": ("Soap-opera code-switched speech (research only)", "4a5a4b8b-0a00-4056-aea9-7cc50217afbc", "5lang"),
    # African Speech Technology telephone corpora: one of each kind (the others share the format)
    "ast_be": ("AST Black-English telephone speech, SA English (CC BY 2.5 SA)", "96de900e-028e-4fde-bc84-2f7f16c707e9",
               None),
    "ast_aa": ("AST Afrikaans-Afrikaans telephone speech (CC BY 2.5 SA)", "34fa9041-12b4-438a-990e-73c4975c76bf", None),
}
TEXT = {".txt", ".xml", ".csv", ".tsv", ".lab", ".trn", ".json", ".md", ".textgrid", ".stm", ".ctm", ".eaf", ".lst",
        ".dict", ".ini", ".html", ".htm", ".prompts", ".log", ".t02", ".wdl", ""}
AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".alaw", ".ulaw", ".raw", ".pcm"}
HEAD_BYTES = 200_000  # large text files: only their start is read


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def files_of(item: str, must: str | None) -> list[dict]:
    bundles = get_json(f"{API}/items/{item}/bundles")["_embedded"]["bundles"]
    original = next(b for b in bundles if b["name"] == "ORIGINAL")
    bits = get_json(original["_links"]["bitstreams"]["href"])["_embedded"]["bitstreams"]
    return [b for b in bits if must is None or must in b["name"] or b["sizeBytes"] < 1_000_000]


def download(bit: dict, folder: Path) -> Path:
    target = folder / bit["name"]
    if target.exists() and target.stat().st_size == bit["sizeBytes"]:
        return target
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(f"{API}/bitstreams/{bit['uuid']}/content", timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    tmp.replace(target)
    print(f"  downloaded {bit['name']} ({bit['sizeBytes'] / 1e9:.2f} GB)", flush=True)
    return target


def members(path: Path):
    """(name, size, read(limit=None) -> bytes) for every file in a zip or tar archive, in archive order."""
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        for info in z.infolist():
            if not info.is_dir():
                def read(limit=None, i=info):
                    with z.open(i) as f:
                        return f.read() if limit is None else f.read(limit)
                yield info.filename, info.file_size, read
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as t:
            for m in t:
                if m.isfile():
                    data = t.extractfile(m).read(min(m.size, 50_000_000))
                    yield m.name, m.size, (lambda limit=None, d=data: d if limit is None else d[:limit])
    else:
        def read(limit=None):
            with open(path, "rb") as f:
                return f.read() if limit is None else f.read(limit)
        yield path.name, path.stat().st_size, read


def audio_info(name: str, data: bytes) -> tuple[str, dict | None]:
    try:
        import soundfile as sf

        info = sf.info(io.BytesIO(data))
        return (f"{name}: {info.samplerate} Hz, {info.channels} ch, {info.duration:.1f} s, {info.subtype}",
                {"sr": info.samplerate, "ch": info.channels, "subtype": info.subtype})
    except Exception as e:  # e.g. headerless A-law, or an MP3 this libsndfile cannot read
        return f"{name}: {len(data)} bytes, not readable by soundfile ({type(e).__name__})", None


def hours(ext: str, total_bytes: int, n: int, fmt: dict | None) -> str:
    if ext == ".wav" and fmt and fmt["subtype"] == "PCM_16":
        return f"{(total_bytes - 44 * n) / (fmt['sr'] * fmt['ch'] * 2) / 3600:.1f} h (from sizes, {fmt['sr']} Hz 16-bit)"
    if ext in (".alaw", ".ulaw"):
        return f"{total_bytes / 8000 / 3600:.1f} h (from sizes, if 8 kHz 8-bit telephone audio, headerless)"
    return "unknown (format not known from the samples)"


def inspect(path: Path, lines: list[str], depth: int = 0) -> None:
    exts, folders, samples, heads, audio, readmes = Counter(), Counter(), [], {}, [], []
    audio_bytes, audio_fmt, inner = Counter(), {}, []
    total = 0
    for name, size, read in members(path):
        p = PurePosixPath(name)
        if p.name.startswith("._") or "__MACOSX" in p.parts or p.name == ".DS_Store":
            continue  # macOS metadata
        ext = p.suffix.lower()
        exts[ext or "(none)"] += 1
        folders[str(PurePosixPath(*p.parts[:-1][:3])) or "."] += 1
        total += size
        if len(samples) < 25:
            samples.append(f"{name} ({size:,} B)")
        if "readme" in p.name.lower() and len(readmes) < 4:
            readmes.append((name, read(HEAD_BYTES).decode("utf-8", errors="replace")))
        elif ext in TEXT and len(heads.setdefault(ext, [])) < 2:
            text = read(HEAD_BYTES).decode("utf-8", errors="replace")
            heads[ext].append((name, size, "\n".join(text.splitlines()[:60])[:4000]))
        elif ext in AUDIO:
            audio_bytes[ext] += size
            if len(audio) < 4:
                desc, fmt = audio_info(name, read())
                audio.append(desc)
                audio_fmt.setdefault(ext, fmt)
        elif ext in (".zip", ".tar", ".gz", ".tgz") and depth == 0:
            inner.append((name, read))
    lines += [f"Files: {sum(exts.values()):,}, {total / 1e9:.2f} GB unpacked", "",
              "By extension: " + ", ".join(f"{e} {n:,}" for e, n in exts.most_common()), "",
              "By folder (first 3 levels):", "```"] + [f"{n:6,}  {d}" for d, n in folders.most_common(30)] + ["```", "",
              "First files:", "```"] + samples + ["```", ""]
    for ext, b in audio_bytes.items():
        lines += [f"Audio {ext}: {exts[ext]:,} files, about {hours(ext, b, exts[ext], audio_fmt.get(ext))}", ""]
    if audio:
        lines += ["Audio samples:", "```"] + audio + ["```", ""]
    for name, text in readmes:
        lines += [f"Readme `{name}`:", "```", "\n".join(text.splitlines()[:150]), "```", ""]
    for ext, items in heads.items():
        for name, size, text in items:
            part = " (start)" if size > HEAD_BYTES else ""
            lines += [f"Start of `{name}`{part}:", "```", text, "```", ""]
    for name, read in inner:  # an archive inside the archive: unpack it to a temporary file and look inside
        with tempfile.NamedTemporaryFile(suffix=PurePosixPath(name).suffix, delete=False) as f:
            f.write(read())
        lines += [f"#### Inside `{name}`", ""]
        inspect(Path(f.name), lines, depth + 1)
        Path(f.name).unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(ITEMS), help="inspect only these corpora")
    ap.add_argument("--match", nargs="+", default=[], metavar="KEY=SUBSTRING",
                    help="download only files whose name contains SUBSTRING for that corpus")
    ap.add_argument("--download-dir", default="/tmp/sadilar")
    ap.add_argument("--output-dir", default="results/sadilar_inspect")
    ap.add_argument("--sync-repo")
    args = ap.parse_args()
    match = dict(m.split("=", 1) for m in args.match)
    folder = Path(args.download_dir)
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["# SADiLaR speech corpora: what is inside", ""]
    for key in args.only or ITEMS:
        title, item, must = ITEMS[key]
        print(title, flush=True)
        lines += [f"## {title}", "", f"Item https://repo.sadilar.org/items/{item}", ""]
        try:
            for bit in files_of(item, match.get(key, must)):
                lines += [f"### `{bit['name']}` ({bit['sizeBytes'] / 1e9:.2f} GB)", ""]
                inspect(download(bit, folder), lines)
        except Exception as e:
            lines += [f"Failed: {type(e).__name__}: {e}", ""]
    out = PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report = out / ("report.md" if not args.only else f"report_{'_'.join(args.only)}.md")
    md = "\n".join(lines)
    report.write_text(md, encoding="utf-8")
    print(md, flush=True)
    if args.sync_repo:
        HubSync(args.sync_repo).push_files(report)


if __name__ == "__main__":
    main()
