"""LLM post-correction of Whisper's n-best list (generative error correction).

Whisper's mistakes on the Nguni and Sotho-Tswana languages are mostly near-miss spellings:
about 35-45% of substituted words are one character off (tša -> tsa, ajabulisayo ->
ajabulishayo). A language model that sees all beam candidates can merge them and fix such
spellings, which picking one candidate (MzansiLM rescoring) cannot.

The risk is that a general LLM, weak in these languages, rewrites the sentence or translates
it. `guard` therefore only accepts a correction whose every word is in some candidate or a
small edit away from one, and whose length stays close to the top candidate's; otherwise the
top candidate is kept.
"""

from __future__ import annotations

import re

from .text import normalize_for_scoring

# Spelling notes for the languages where round 2 makes systematic errors (see the error
# breakdown of results/baseline/predictions/sa-turbo-lora-v2).
ORTHOGRAPHY_HINTS = {
    "nso": "Sepedi spelling uses š (tša, tše, lentšu, šomiša); do not replace it with s.",
    "zul": (
        "isiZulu is written conjunctively: prefixes join the word (ngasekhaya, kwezempilo). "
        "Use isiZulu spelling, not siSwati (no 'tf' or 'tj': isithupha, nitshele)."
    ),
    "xho": "isiXhosa is written conjunctively: prefixes join the word.",
    "ssw": "siSwati spelling uses 'tf' and 'tj' where isiZulu uses 'th' and 'tsh'.",
    "sot": (
        "Sesotho (South African orthography) is written disjunctively: short words such as "
        "ho, ya, le, ba, ka are separate words and must not be dropped."
    ),
    "tsn": "Setswana is written disjunctively: short words such as go, ya, le, ba are separate words.",
}

SYSTEM_TEMPLATE = (
    "You correct speech-recognition output in {language}. You receive candidate "
    "transcriptions of the same audio clip, most likely first. Return the single most "
    "likely correct transcript in {language}.\n"
    "Rules:\n"
    "- Keep the wording of the candidates. Fix only spelling, diacritics and word boundaries, "
    "and choose between the candidates where they differ.\n"
    "- Never translate. Keep English words that the speaker said in English.\n"
    "- Never add words that no candidate supports, and never drop words.\n"
    "- Reply with the transcript only: no quotes, labels or explanation.{hint}"
)


def build_messages(language: str, code: str, candidates: list[str], hints: bool = True) -> list[dict]:
    hint = f"\n- {ORTHOGRAPHY_HINTS[code]}" if hints and code in ORTHOGRAPHY_HINTS else ""
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(candidates, start=1))
    return [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(language=language, hint=hint)},
        {"role": "user", "content": f"Candidates:\n{numbered}\n\nCorrected transcript:"},
    ]


_LABEL = re.compile(r"^(corrected transcript|transcript|correction|answer)\s*:\s*", re.IGNORECASE)


def parse_response(text: str) -> str:
    """First non-empty line, without code fences, list numbering, labels or quotes."""
    for line in (text or "").replace("```", "\n").splitlines():
        line = _LABEL.sub("", line.strip())
        line = re.sub(r"^\d+[.)]\s+", "", line)
        line = line.strip().strip('"“”').strip()
        if line:
            return line
    return ""


def edit_distance(a: str, b: str, limit: int) -> int:
    """Levenshtein distance, stopping early once it exceeds `limit` (returns limit + 1)."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def guard(correction: str, candidates: list[str]) -> tuple[bool, str]:
    """Accept `correction` only if it stays close to the candidates. Returns (ok, reason)."""
    words = normalize_for_scoring(correction).split()
    top = normalize_for_scoring(candidates[0]).split()
    if not words:
        return False, "empty"
    if abs(len(words) - len(top)) > max(2, round(0.2 * len(top))):
        return False, "length"
    vocab = {w for c in candidates for w in normalize_for_scoring(c).split()}
    for word in words:
        if word in vocab:
            continue
        limit = 1 if len(word) <= 4 else 2
        if not any(edit_distance(word, v, limit) <= limit for v in vocab):
            return False, "new_word"
    return True, "ok"
