"""Transcript cleanup (for training targets) and normalization (for scoring).

The two are deliberately different:
- clean_transcript keeps case and punctuation, and only removes corpus annotation tags
  such as [noise] or <unk>. The model learns to produce readable text.
- normalize_for_scoring removes everything that should not count as an error (case,
  punctuation) so WER compares only the words.

Both keep diacritics. Tshivenda (ḓ ḽ ṅ ṋ ṱ) and Sesotho/Setswana/Sepedi (š ê ô) use them
to tell words apart, so stripping them would hide real errors.
"""

from __future__ import annotations

import re
import unicodedata

# [noise], [um], <unk>, <sil>, (laughs), {breath} ... as used by NCHLT, Lwazi and similar corpora.
ANNOTATION_TAG = re.compile(r"\[[^\]]*\]|<[^>]*>|\([^)]*\)|\{[^}]*\}")
WHITESPACE = re.compile(r"\s+")
APOSTROPHES = str.maketrans({"‘": "'", "’": "'", "ʼ": "'", "`": "'", "´": "'"})
DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-"})


def clean_transcript(text: str) -> str:
    """Training-target cleanup: NFC, strip annotation tags, tidy whitespace."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.translate(APOSTROPHES).translate(DASHES)
    text = ANNOTATION_TAG.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


# A 1-4 letter piece repeated 7+ times inside one word ("hahahahahahaha...").
LETTER_LOOP = re.compile(r"(\S{1,4}?)\1{6,}")


def collapse_repeats(text: str, max_repeats: int = 5, max_phrase: int = 6) -> str:
    """Loop guard for decoded text: a word or phrase (up to max_phrase words) repeated more than
    max_repeats times in a row is cut back to max_repeats copies.

    Whisper sometimes gets stuck on long or noisy clips and repeats one word until the token limit
    ("ha ha ha ha ...", "sithandathu sithandathu ..."), and a single such clip can add hundreds of
    insertions to a test set. Real speech repeats a word a few times at most ("hayi hayi hayi", a
    stammered "swa swa swa swa" in Swivuriso Xitsonga), which stays untouched. Letter loops inside
    one word are cut back the same way.
    """
    text = LETTER_LOOP.sub(lambda m: m.group(1) * max_repeats, text)
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        for n in range(1, max_phrase + 1):
            unit = words[i : i + n]
            reps = 1
            while words[i + reps * n : i + (reps + 1) * n] == unit:
                reps += 1
            if reps > max_repeats:
                out += unit * max_repeats
                i += reps * n
                break
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def normalize_for_scoring(text: str) -> str:
    """Scoring normalization: clean, lowercase, drop punctuation.

    Apostrophes and hyphens are kept when they sit inside a word, because they are part
    of the spelling ("'n" in Afrikaans, "i-Afrika" in isiZulu). Everywhere else they
    are treated as punctuation.
    """
    text = clean_transcript(text).lower()
    chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "M", "N") or ch in "'-":
            chars.append(ch)
        else:
            chars.append(" ")
    words = []
    for word in "".join(chars).split():
        word = word.strip("-")
        # A leading apostrophe is spelling in Afrikaans ('n, 's); a trailing one is a quote mark.
        word = word.rstrip("'")
        if word.strip("'-"):
            words.append(word)
    return " ".join(words)
