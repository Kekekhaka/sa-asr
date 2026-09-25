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
