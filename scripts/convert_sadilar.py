"""Convert SADiLaR speech corpora into this project's Parquet layout, in a private Hugging Face dataset (CPU only;
Kaggle stage sadilar_data, Accelerator: None).

    python scripts/convert_sadilar.py --repo <user>/sa-asr-sadilar
    python scripts/convert_sadilar.py --repo <user>/sa-asr-sadilar --only spcs soap
    python scripts/convert_sadilar.py --repo <user>/sa-asr-sadilar --only sama ast --redo   # replace those corpora

One config per corpus and language (spcs_nso, sama_afr, soap_zul, ast_eng, ...), each with train/validation/test
Parquet files: audio (16 kHz FLAC bytes), utt_id, speaker, text, lang, duration, and info (genre, accent or the
languages of a code-switched clip). Splits hold out whole speakers or episodes (sa_asr.sadilar.split_of); the soap
opera keeps its published dev/test lists. Each config is uploaded as soon as it is written and skipped when it is
already in the repo, so a cut-off run continues where it stopped. Downloads are checked against SADiLaR's sizes
(a stream can end early without an error) and resumed.

Licences: SPCS CC BY 2.5 ZA, SAMA CC BY 4.0 (its readme still says TBC), Multipron CC BY 3.0, AST CC BY 2.5 ZA,
soap opera research only. Keep the dataset and models trained on it private.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf

import sa_asr  # noqa: F401  (corporate-certificate fix before any download)
from sa_asr.sadilar import (clean_ast, clean_sama, decode_alaw, lab_utterances, soap_label, split_of,
                            t02_utterance, textgrid_utterances)

log = logging.getLogger("convert_sadilar")
API = "https://repo.sadilar.org/server/api/core"
SR = 16_000
ITEMS = {  # SADiLaR item UUIDs (the AST ones: one per accent group, read from the repository's search API)
    "spcs": ["9b255944-5050-47b6-a662-1d43ba0e7441"],
    "sama": ["e44a1b0c-34c5-4957-9560-4266ca17c7d2"],
    "multipron": ["396b4c02-4ee9-4996-b081-0e1a1aa88d1f"],
    "soap": ["4a5a4b8b-0a00-4056-aea9-7cc50217afbc"],
    "ast": ["96de900e-028e-4fde-bc84-2f7f16c707e9",  # Black-English
            "cc1f907c-544a-40ae-b504-31367e61f279",  # Afrikaans-English
            "b6abd874-1a08-4cd8-a81e-7bcad7978961",  # Coloured-English
            "aaef2e4e-b4ce-4582-8774-46f397ef29bd",  # English-English
            "aed711e5-6e1a-4ab6-903d-2c64f23f4ef2",  # Indian-English
            "34fa9041-12b4-438a-990e-73c4975c76bf",  # Afrikaans-Afrikaans
            "24635f91-41db-4e73-92f9-c1855f5e3369",  # Black-Afrikaans
            "067347ba-08bb-44f6-964a-191f123fafe2",  # Coloured-Afrikaans
            "1129ad5c-847d-411f-a131-c058cbc87bd7",  # isiXhosa
            "c3e6cced-ce34-41c2-8920-9abcacd80b31",  # isiZulu
            "224be57b-38fb-4b7c-8b76-aa4d0771d4cc"],  # Sesotho
}
MULTIPRON_LANGS = {"afrikaans": "afr", "english": "eng", "isizulu": "zul", "sesotho": "sot"}
SAMA_LANGS = {"afr", "nso", "sot", "ven", "xho", "zul"}


# ---------------------------------------------------------------------------------------------------------------
# Download and write
# ---------------------------------------------------------------------------------------------------------------


def get_json(url: str) -> dict:
    import json

    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def item_files(uuid: str) -> list[dict]:
    bundles = get_json(f"{API}/items/{uuid}/bundles")["_embedded"]["bundles"]
    original = next(b for b in bundles if b["name"] == "ORIGINAL")
    return get_json(original["_links"]["bitstreams"]["href"])["_embedded"]["bitstreams"]


def download(bit: dict, folder: Path) -> Path:
    """The whole file, resuming with HTTP Range when the server ends a stream early."""
    target = folder / bit["name"]
    if target.exists() and target.stat().st_size == bit["sizeBytes"]:
        return target
    url, got = f"{API}/bitstreams/{bit['uuid']}/content", 0
    with open(target, "wb") as f:
        for _ in range(30):
            req = urllib.request.Request(url, headers={"Range": f"bytes={got}-"} if got else {})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    if got and r.status != 206:  # the server ignored Range: start again
                        f.seek(0)
                        f.truncate()
                        got = 0
                    while chunk := r.read(1 << 22):
                        f.write(chunk)
                        got += len(chunk)
            except Exception as e:
                log.warning("%s: %s after %d bytes; resuming", bit["name"], type(e).__name__, got)
            if got >= bit["sizeBytes"]:
                break
    if target.stat().st_size != bit["sizeBytes"]:
        raise IOError(f"{bit['name']}: got {target.stat().st_size} of {bit['sizeBytes']} bytes")
    log.info("downloaded %s (%.2f GB)", bit["name"], bit["sizeBytes"] / 1e9)
    return target


def flac(wav: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(wav, -1, 1), SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")


def text_of(data: bytes) -> str:
    """Text file bytes -> str, for the UTF-16 files Praat writes as well as UTF-8."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    if len(data) > 1 and data[1:2] == b"\x00":  # UTF-16 LE without a byte-order mark
        return data.decode("utf-16-le", "replace")
    return data.decode("utf-8-sig", "replace")


def key_of(name: str) -> str:
    """Archive member or CSV filename -> matching key: lowercase base name without the audio extension."""
    base = PurePosixPath(name.replace("\\", "/").strip()).name.lower()
    return base.rsplit(".", 1)[0] if base.endswith(AUDIO_EXT) else base


BAD_AUDIO = [0]  # files neither soundfile nor ffmpeg could decode (skipped, counted)


def read_wav(data: bytes) -> np.ndarray:
    """Decode to 16 kHz mono. Falls back to ffmpeg for files libsndfile rejects (mislabelled
    mp3/odd WAV codecs); returns an empty array for undecodable files, which Writer.add skips."""
    import soxr

    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        try:
            out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1",
                                  "-ar", str(SR), "pipe:1"], input=data, capture_output=True, check=True).stdout
            return np.frombuffer(out, dtype=np.float32).copy()
        except Exception:
            BAD_AUDIO[0] += 1
            if BAD_AUDIO[0] <= 5:
                log.warning("undecodable audio skipped (%d bytes, starts %r)", len(data), data[:12])
            return np.zeros(0, dtype=np.float32)
    x = x.mean(axis=1)
    return soxr.resample(x, sr, SR).astype(np.float32) if sr != SR else x


class Writer:
    """Rows -> <out>/<config>/<split>-NNNNN.parquet, a new file every `per_file` rows."""

    def __init__(self, out: Path, per_file: int = 2000):
        self.out, self.per_file = out, per_file
        self.rows: dict[tuple[str, str], list] = defaultdict(list)
        self.files: dict[tuple[str, str], int] = defaultdict(int)
        self.stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def add(self, config: str, split: str, utt_id: str, speaker: str, text: str, lang: str, wav: np.ndarray,
            info: str = "", max_s: float = 30.0) -> None:
        dur = len(wav) / SR
        if not text or dur < 0.3 or dur > max_s:
            return
        self.rows[(config, split)].append({"audio": {"bytes": flac(wav), "path": f"{utt_id}.flac"}, "utt_id": utt_id,
                                           "speaker": speaker, "text": text, "lang": lang,
                                           "duration": round(dur, 3), "info": info})
        self.stats[config][split] += dur / 3600
        if len(self.rows[(config, split)]) >= self.per_file:
            self.flush(config, split)

    def flush(self, config: str, split: str) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = self.rows.pop((config, split), [])
        if rows:
            path = self.out / config / f"{split}-{self.files[(config, split)]:05d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=100)
            self.files[(config, split)] += 1

    def finish(self) -> list[str]:
        for key in list(self.rows):
            self.flush(*key)
        return sorted({c for c, _ in self.files})


# ---------------------------------------------------------------------------------------------------------------
# One function per corpus: (archives folder, writer) -> rows added
# ---------------------------------------------------------------------------------------------------------------


def convert_spcs(work: Path, w: Writer) -> None:
    arc = download(item_files(ITEMS["spcs"][0])[0], work)
    with tarfile.open(arc, "r:gz") as t:
        files = {m.name.split("SPCS/", 1)[-1]: t.extractfile(m).read() for m in t if m.isfile()}
    clean = files["lists/CLEAN.trans.lst"].decode("utf-8", "replace").split()
    for rel in clean:  # 001/001_Female_23_100.txt
        txt, wav = files.get(f"transcriptions/{rel}"), files.get(f"audio/{rel[:-4]}.wav")
        if txt is None or wav is None:
            continue
        speaker = rel.split("/")[0]
        text = " ".join(txt.decode("utf-8", "replace").split())
        w.add("spcs_nso", split_of(f"spcs:{speaker}"), PurePosixPath(rel).stem, speaker, text, "nso", read_wav(wav),
              info="nso+eng")


def convert_sama(work: Path, w: Writer) -> None:
    for bit in item_files(ITEMS["sama"][0]):
        lang = bit["name"][:3]
        if lang not in SAMA_LANGS or not bit["name"].endswith(".zip"):
            continue
        outer = zipfile.ZipFile(download(bit, work))
        for inner_name in [n for n in outer.namelist() if n.endswith(".zip") and "__MACOSX" not in n]:
            genre = PurePosixPath(inner_name).stem  # news / drama
            with tempfile.NamedTemporaryFile(suffix=".zip", dir=work, delete=False) as tmp:
                with outer.open(inner_name) as src:
                    shutil.copyfileobj(src, tmp, 1 << 22)
            inner = zipfile.ZipFile(tmp.name)
            names = inner.namelist()
            audio = {key_of(n): n for n in names if n.lower().endswith(AUDIO_EXT)}
            kept, unclear, missing = 0, 0, []
            csvs = [n for n in names if n.lower().endswith(".csv") and "/info/" in n.lower()]
            for csv_name in csvs:
                rows = csv.DictReader(io.StringIO(text_of(inner.read(csv_name))))
                for row in rows:
                    row = {k.strip().lower(): (v or "") for k, v in row.items() if k}
                    text, path = clean_sama(row.get("transcription", "")), audio.get(key_of(row.get("filename", "")))
                    if path is None:
                        missing.append(row.get("filename", ""))
                        continue
                    if text is None:
                        unclear += 1
                        continue
                    episode = PurePosixPath(path).stem.rsplit("_", 1)[0]  # sama_nso_p00118
                    w.add(f"sama_{lang}", split_of(f"{genre}:{episode}"), PurePosixPath(path).stem,
                          f"{genre}_{episode}", text, lang, read_wav(inner.read(path)), info=genre)
                    kept += 1
            log.info("sama %s %s: %d clips kept, %d unclear speech, %d with no audio (%d csv, %d audio files)",
                     lang, genre, kept, unclear, len(missing), len(csvs), len(audio))
            if not kept:  # show what did not match, to fix the reader
                log.warning("sama %s %s: csv names %s; audio %s; members %s", lang, genre, missing[:3],
                            list(audio.values())[:3], names[:8])
            inner.close()
            Path(tmp.name).unlink()
        outer.close()
        (work / bit["name"]).unlink()  # 1-3 GB each


def convert_multipron(work: Path, w: Writer) -> None:
    z = zipfile.ZipFile(download(item_files(ITEMS["multipron"][0])[0], work))
    names = set(z.namelist())
    for n in sorted(names):
        parts = PurePosixPath(n).parts  # data/list_A/afrikaans/<speaker>/audio/<id>.wav
        if len(parts) < 6 or parts[0] != "data" or parts[4] != "audio" or not n.endswith(".wav"):
            continue
        lang, speaker = MULTIPRON_LANGS.get(parts[2]), parts[3]
        txt = f"data/{parts[1]}/{parts[2]}/{speaker}/transcription/{PurePosixPath(n).stem}.txt"
        if lang is None or txt not in names:
            continue
        text = " ".join(z.read(txt).decode("utf-8", "replace").split())
        w.add(f"multipron_{lang}", split_of(f"multipron:{speaker}"), PurePosixPath(n).stem, speaker, text, lang,
              read_wav(z.read(n)), info="names")


def convert_soap(work: Path, w: Writer) -> None:
    bits = {b["name"]: b for b in item_files(ITEMS["soap"][0])}
    root = ET.parse(download(bits["5lang.xml"], work)).getroot()
    held = {}
    with zipfile.ZipFile(download(bits["soapies_dev_and_test_set_utterance_ids.zip"], work)) as ids:
        for n in ids.namelist():
            split = "test" if "_tst_" in n else "validation" if "_dev_" in n else None
            if split:
                held.update({u.strip(): split for u in ids.read(n).decode().split() if u.strip()})
    z = zipfile.ZipFile(download(bits["5lang.zip"], work))
    audio = {PurePosixPath(n).name: n for n in z.namelist() if n.endswith(".wav")}
    skipped = 0
    for u in root.iter("utterance"):
        segs = u.findall("utterance_segment")
        langs = [s.findtext("lang_id") or "" for s in segs]
        lang, name = soap_label(langs), u.findtext("audio") or ""
        text = " ".join(" ".join((s.findtext("transcription") or "").split()) for s in segs).strip()
        if lang is None or name not in audio or not text:
            skipped += 1
            continue
        uid = name.rsplit(".", 1)[0]
        mix = "+".join(dict.fromkeys(langs))
        w.add(f"soap_{lang}", held.get(uid, "train"), uid, u.findtext("speaker_id") or "", text, lang,
              read_wav(z.read(audio[name])), info=mix)
    log.info("soap: %d utterances skipped (two Bantu languages mixed, no audio or no text)", skipped)


def by_stem(names: list[str], ext: str) -> dict[str, list[str]]:
    """Lower-case stem -> every member with that extension (case ignored). Some AST corpora reuse file names in
    different speaker folders, so a stem can have several."""
    out: dict[str, list[str]] = {}
    for n in names:
        if n.lower().endswith(ext):
            out.setdefault(PurePosixPath(n).stem.lower(), []).append(n)
    return out


def nearest(audio: str, candidates: list[str]) -> str | None:
    """The candidate sharing the most folder names with the audio file; None if there is none or it is a tie."""
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    dirs = {d.lower() for d in PurePosixPath(audio).parent.parts}
    scored = sorted(((len(dirs & {d.lower() for d in PurePosixPath(c).parent.parts}), c) for c in candidates),
                    reverse=True)
    return scored[0][1] if scored[0][0] > scored[1][0] else None


def convert_ast(work: Path, w: Writer) -> None:
    for uuid in ITEMS["ast"]:
        try:
            bits = [b for b in item_files(uuid) if b["name"].endswith(".zip")]
        except Exception as e:
            log.warning("AST item %s: %s; skipped", uuid, e)
            continue
        for bit in bits:
            z = zipfile.ZipFile(download(bit, work))
            names = z.namelist()
            # the corpus folder, e.g. "AST Black-English", "Afrikaans-Afrikaans"
            accent = Counter(n.split("/")[0] for n in names if "/" in n).most_common(1)[0][0]
            low = accent.lower()
            lang = ("eng" if low.endswith("english") else "afr" if low.endswith("afrikaans") else
                    "xho" if "xhosa" in low else "zul" if "zulu" in low else "sot" if "sotho" in low else None)
            if lang is None:
                log.warning("AST %s: language not recognised; skipped", accent)
                continue
            grids, t02s = by_stem(names, ".textgrid"), by_stem(names, ".t02")
            # htk/*.lab are phone labels (isiZulu: "0 25320001 sta"), not words
            labs = by_stem([n for n in names if "/htk/" not in n.lower()], ".lab")
            for stem, cands in t02s.items():  # one .t02 per phone set (astbet, xsampa, ...): the Word column is shared
                astbet = [c for c in cands if "/astbet/" in c.lower()]
                t02s[stem] = astbet or cands
            alaws = [n for n in names if n.lower().endswith(".alaw")]
            kept = ambiguous = from_t02 = from_lab = 0
            examples = []
            reused = {s for s, k in Counter(PurePosixPath(n).stem.lower() for n in alaws).items() if k > 1}
            for n in alaws:
                stem = PurePosixPath(n).stem  # BE000CF001: database, speaker, call, utterance
                # a reused name (001.alaw in each speaker folder): the folder is the speaker
                folder = PurePosixPath(n).parent.name if stem.lower() in reused else None
                uid, speaker = (f"{folder}_{stem}", folder) if folder else (stem, stem[:7])
                key = stem.lower()
                if (src := nearest(n, grids.get(key, []))) is not None:
                    utts = textgrid_utterances(text_of(z.read(src)))
                elif (src := nearest(n, t02s.get(key, []))) is not None:
                    utts = t02_utterance(text_of(z.read(src)))
                    from_t02 += 1
                elif (src := nearest(n, labs.get(key, []))) is not None:
                    utts = lab_utterances(text_of(z.read(src)))
                    from_lab += 1
                else:
                    ambiguous += any(len(d.get(key, [])) > 1 for d in (grids, t02s, labs))
                    continue
                wav = None
                for i, (a, b, raw) in enumerate(utts):
                    text = clean_ast(raw)
                    if text is None:
                        continue
                    wav = decode_alaw(z.read(n)) if wav is None else wav
                    clip = wav if b is None else wav[max(int((a - 0.15) * SR), 0) : int((b + 0.15) * SR)]
                    w.add(f"ast_{lang}", split_of(f"ast:{accent}:{speaker}"), f"{uid}_{i}", speaker, text, lang, clip,
                          info=accent.removeprefix("AST ").strip())
                    kept += 1
                    if len(examples) < 3:
                        examples.append(f"{uid} {len(clip) / SR:.1f}s: {text}")
            log.info("AST %s: %d utterances (%d audio files, %d TextGrids, %d .t02 used, %d .lab used, "
                     "%d skipped as ambiguous); e.g. %s", accent, kept, len(alaws), sum(map(len, grids.values())),
                     from_t02, from_lab, ambiguous, examples)
            if from_t02 or from_lab or not kept:  # show the layout and sample transcriptions, to check the reader
                samples = {}
                for n in names:
                    ext = PurePosixPath(n).suffix.lower()
                    if ext in (".lab", ".t02", ".textgrid") and ext not in samples:
                        samples[ext] = (n, text_of(z.read(n))[:800])
                log.warning("AST %s: file types %s; members %s; samples %s", accent,
                            Counter(PurePosixPath(n).suffix.lower() for n in names).most_common(8), names[:8], samples)
            z.close()
            (work / bit["name"]).unlink()


CONVERTERS = {"spcs": convert_spcs, "sama": convert_sama, "multipron": convert_multipron, "soap": convert_soap,
              "ast": convert_ast}


def readme(configs: list[str], repo_files: list[str]) -> str:
    lines = ["---", "configs:"]
    for c in configs:
        lines.append(f"- config_name: {c}\n  data_files:")
        for split in ("train", "validation", "test"):
            if any(f.startswith(f"{c}/{split}-") for f in repo_files):
                lines.append(f"  - split: {split}\n    path: {c}/{split}-*.parquet")
    lines += ["---", "", "# SADiLaR speech corpora in the sa-asr training layout (private)", "",
              "Converted by scripts/convert_sadilar.py from https://repo.sadilar.org. Licences: SPCS CC BY 2.5 ZA, "
              "CSIR SAMA CC BY 4.0, Multipron CC BY 3.0, African Speech Technology (AST) CC BY 2.5 ZA, soap-opera "
              "code-switching corpus research only (van der Westhuizen & Niesler, LREC 2018). Keep this dataset and "
              "models trained on it private."]
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="private HF dataset to write, e.g. <user>/sa-asr-sadilar")
    ap.add_argument("--only", nargs="+", choices=list(CONVERTERS))
    ap.add_argument("--redo", action="store_true",
                    help="convert the --only corpora again, replacing their folders in the repo (one commit, and "
                         "only when the new conversion wrote something)")
    ap.add_argument("--work", default="/tmp/sadilar")
    args = ap.parse_args()
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    for name in args.only or CONVERTERS:
        done = {f.split("/")[0] for f in api.list_repo_files(args.repo, repo_type="dataset")}
        if any(c.startswith(f"{name}_") for c in done) and not (args.redo and args.only):
            log.info("%s: already in %s; skipped (delete its folders there to convert again)", name, args.repo)
            continue
        out = work / f"out_{name}"
        w = Writer(out)
        CONVERTERS[name](work, w)
        configs = w.finish()
        if BAD_AUDIO[0]:
            log.warning("%s: %d undecodable audio files skipped", name, BAD_AUDIO[0])
            BAD_AUDIO[0] = 0
        for c in configs:
            log.info("%s: %s", c, ", ".join(f"{s} {h:.1f} h" for s, h in sorted(w.stats[c].items())))
        if not configs:
            log.warning("%s: nothing converted; repo left as it was", name)
            continue
        api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(out),
                          commit_message=f"{name}: {', '.join(configs)}",
                          delete_patterns=[f"{name}_*/*"] if args.redo else None)
        shutil.rmtree(out, ignore_errors=True)
    files = api.list_repo_files(args.repo, repo_type="dataset")
    configs = sorted({f.split("/")[0] for f in files if f.endswith(".parquet")})
    api.upload_file(path_or_fileobj=readme(configs, files).encode(), path_in_repo="README.md", repo_id=args.repo,
                    repo_type="dataset", commit_message="configs")
    log.info("configs in %s: %s", args.repo, ", ".join(configs))


if __name__ == "__main__":
    main()
