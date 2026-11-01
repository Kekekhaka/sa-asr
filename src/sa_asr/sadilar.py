"""SADiLaR speech corpora: transcript cleaning, telephone audio decoding and stable splits (used by
scripts/convert_sadilar.py, which turns them into this project's Parquet training layout).

Layouts and markup were read from the corpora themselves (scripts/inspect_sadilar.py, 2026-10-31/11-01):
  SPCS       Sepedi-English code-switched prompts: audio/<spk>/<id>.wav + transcriptions/<spk>/<id>.txt, CLEAN list.
  SAMA       SABC radio news/drama, manually verified: info/*.csv (filename,duration,transcription); English and
             other words inside a Bantu sentence are marked [foreign_eng + word]; [music] [laughter] [?] ...
  Multipron  read proper names: data/list_X/<language>/<speaker>/{audio/*.wav, transcription/*.txt}.
  Soap opera 5lang.xml: <utterance> with <audio> and <utterance_segment>s, each with <lang_id> and <transcription>.
  AST        telephone corpora: headerless 8 kHz A-law + Praat TextGrids whose Orthographic tier marks
             utterances (u)...(/u), noise [int] [sta] [sil], other-language speech ($)...(/$), truncated words '+'.
"""

from __future__ import annotations

import re
import zlib

import numpy as np


def split_of(key: str, test: float = 0.05, validation: float = 0.05) -> str:
    """Stable split by a key (speaker or episode), so one speaker's clips never straddle train and test."""
    u = zlib.crc32(key.encode()) / 2**32
    return "test" if u < test else "validation" if u < test + validation else "train"


def _alaw_table() -> np.ndarray:
    """G.711 A-law byte -> 16-bit sample (the ITU decoding, as in Sun's g711.c)."""
    out = np.zeros(256, dtype=np.int16)
    for i in range(256):
        a = i ^ 0x55
        t = (a & 0x0F) << 4
        seg = (a & 0x70) >> 4
        if seg == 0:
            t += 8
        else:
            t = (t + 0x108) << (seg - 1)
        out[i] = t if a & 0x80 else -t
    return out


ALAW = _alaw_table()


def decode_alaw(data: bytes, sr_out: int = 16_000) -> np.ndarray:
    """Headerless 8 kHz A-law bytes -> float32 audio at sr_out."""
    import soxr

    x = ALAW[np.frombuffer(data, dtype=np.uint8)].astype(np.float32) / 32768.0
    return soxr.resample(x, 8_000, sr_out).astype(np.float32) if sr_out != 8_000 else x


_SPACES = re.compile(r"\s+")


def clean_sama(text: str) -> str | None:
    """[foreign_eng + today] -> today; sound tags dropped; None when part of the speech is unintelligible ([?])."""
    if "[?]" in text:
        return None
    t = re.sub(r"\[foreign_\w+\s*\+\s*([^\]]+)\]", r"\1", text)
    t = re.sub(r"\[[^\]]*\]", " ", t)
    t = re.sub(r"\b((?:\w_)+\w)\b", lambda m: m.group(1).replace("_", ""), t)  # f_m -> fm (spelled letters)
    t = _SPACES.sub(" ", t).strip()
    return t or None


def clean_ast(text: str) -> str | None:
    """One utterance of an AST Orthographic tier -> plain words; None if a word is cut off (the audio then holds
    sounds the text does not) or nothing is left."""
    t = re.sub(r"\((/?u|/?\$)\)", " ", text)
    t = re.sub(r"\[/?[^\]]*\]", " ", t)
    words = []
    for w in t.split():
        if w in ("0", "~"):
            continue
        if w.startswith("+") or w.endswith("+"):
            return None
        words.append(w.lstrip("*").replace("_", " "))
    t = _SPACES.sub(" ", " ".join(words)).strip()
    return t or None


def textgrid_utterances(textgrid: str) -> list[tuple[float, float, str]]:
    """(start, end, raw text) of each (u)...(/u) utterance in a TextGrid's first (Orthographic) tier."""
    first = re.split(r'item \[2\]:', textgrid)[0]
    intervals = [(float(a), float(b), t) for a, b, t in
                 re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "((?:[^"]|"")*)"', first)]
    return _group_utterances(intervals)


def _group_utterances(intervals: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    out, cur = [], None
    for a, b, t in intervals:
        if "(u)" in t and cur is None:
            cur = [a, b, t]
        elif cur is not None:
            cur[1], cur[2] = b, cur[2] + " " + t
        if cur is not None and "(/u)" in t:
            out.append(tuple(cur))
            cur = None
    return out


_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def lab_utterances(lab: str) -> list[tuple[float, float | None, str]]:
    """(start, end, raw text) from an AST .lab file. Handles HTK labels ("start end word", 100 ns units when the
    numbers are whole and large), xwaves labels (a header ending in "#", then "end colour word") and plain
    orthographic text (one utterance over the whole file, end None). Labelled files with (u)...(/u) markup are
    grouped like a TextGrid; without it the whole label sequence is one utterance."""
    lines = [ln.strip() for ln in lab.splitlines() if ln.strip()]
    if "#" in lines:  # xwaves
        rows, prev = [], 0.0
        for ln in lines[lines.index("#") + 1:]:
            p = ln.split(None, 2)
            if len(p) == 3 and _NUM.match(p[0]):
                rows.append((prev, float(p[0]), p[2]))
                prev = float(p[0])
    else:
        rows = []
        for ln in lines:
            p = ln.split(None, 2)
            if len(p) == 3 and _NUM.match(p[0]) and _NUM.match(p[1]):
                rows.append((float(p[0]), float(p[1]), p[2]))
        if rows and len(rows) == len(lines):
            if all("." not in ln.split()[1] for ln in lines) and max(b for _, b, _ in rows) > 10_000:
                rows = [(a / 1e7, b / 1e7, t) for a, b, t in rows]
        else:  # plain text
            return [(0.0, None, " ".join(lines))] if lines else []
    if not rows:
        return []
    if any("(u)" in t for _, _, t in rows):
        return _group_utterances(rows)
    return [(rows[0][0], rows[-1][1], " ".join(t for _, _, t in rows))]


def soap_label(langs: list[str]) -> str | None:
    """Language label of a soap-opera clip: English if only English, else its one Bantu language (code-switched
    clips are English inside a Bantu sentence or the reverse); None when two Bantu languages are mixed."""
    bantu = {lang for lang in langs if lang != "eng"}
    return "eng" if not bantu else next(iter(bantu)) if len(bantu) == 1 else None
