"""Spoken language identification (LID): language families and accuracy reports.

South Africa's languages fall into a few families; confusions *within* a family
(isiZulu/isiXhosa/siSwati/isiNdebele, or Sepedi/Sesotho/Setswana) are expected to be far
more common than across families, so reports show both exact and family accuracy.
"""

from __future__ import annotations

from collections import Counter, defaultdict

FAMILIES: dict[str, list[str]] = {
    "Nguni": ["zul", "xho", "ssw", "nbl"],
    "Sotho-Tswana": ["nso", "sot", "tsn"],
    "Germanic": ["eng", "afr"],
    "Tswa-Ronga": ["tso"],
    "Venda": ["ven"],
}
FAMILY_OF = {code: fam for fam, codes in FAMILIES.items() for code in codes}


def family_of(code: str) -> str:
    return FAMILY_OF.get(code, code)


def accuracy(records: list[dict]) -> dict[str, float]:
    """records: {"lang": true code, "pred": predicted code}."""
    n = len(records)
    if not n:
        return {"n": 0, "accuracy": float("nan"), "family_accuracy": float("nan")}
    exact = sum(r["pred"] == r["lang"] for r in records)
    fam = sum(family_of(r["pred"]) == family_of(r["lang"]) for r in records)
    return {"n": n, "accuracy": exact / n, "family_accuracy": fam / n}


def confusion(records: list[dict]) -> dict[str, Counter]:
    """true language -> Counter of predicted languages."""
    out: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        out[r["lang"]][r["pred"]] += 1
    return out


def top_confusions(records: list[dict], k: int = 10) -> list[tuple[str, str, int, float]]:
    """(true, predicted, count, share of true) for the most frequent mistakes."""
    conf = confusion(records)
    pairs = [
        (true, pred, n, n / sum(row.values()))
        for true, row in conf.items()
        for pred, n in row.items()
        if pred != true
    ]
    return sorted(pairs, key=lambda p: (-p[2], p[0]))[:k]


def report_markdown(records: list[dict], names: dict[str, str], title: str) -> str:
    """Markdown report: overall, per corpus x language, confusion matrix, top mistakes.

    records carry "cond" (e.g. "full", "5s"), "corpus", "lang", "pred".
    """
    lines = [f"# {title}", "",
             "Accuracy of the detected language (exact, and within the right family). "
             "Conditions: `full` = the whole test clip; `Ns` = only its first N seconds "
             "(how the app's mixed-language mode sees speech).", ""]
    conds = list(dict.fromkeys(r["cond"] for r in records))
    lines += ["## Overall", "", "| condition | clips | accuracy | family accuracy |", "|---|---|---|---|"]
    for c in conds:
        a = accuracy([r for r in records if r["cond"] == c])
        lines.append(f"| {c} | {a['n']} | {a['accuracy']:.1%} | {a['family_accuracy']:.1%} |")

    lines += ["", "## Accuracy per corpus and language", "",
              "| corpus | language | " + " | ".join(conds) + " |", "|---|---|" + "---|" * len(conds)]
    for corpus, lang in sorted({(r["corpus"], r["lang"]) for r in records}):
        cells = [accuracy([r for r in records if r["cond"] == c and r["corpus"] == corpus and r["lang"] == lang])
                 for c in conds]
        lines.append(f"| {corpus} | {names.get(lang, lang)} | " + " | ".join(f"{a['accuracy']:.0%}" for a in cells) + " |")

    for c in conds:
        recs = [r for r in records if r["cond"] == c]
        conf = confusion(recs)
        codes = [code for fam in FAMILIES.values() for code in fam if code in conf or any(code in row for row in conf.values())]
        lines += ["", f"## Confusion matrix ({c}): rows = true language, columns = detected", "",
                  "| true \\ detected | " + " | ".join(codes) + " |", "|---|" + "---|" * len(codes)]
        for t in codes:
            if t in conf:
                total = sum(conf[t].values())
                lines.append(f"| {names.get(t, t)} | " + " | ".join(
                    f"**{conf[t][p] / total:.0%}**" if p == t else (f"{conf[t][p] / total:.0%}" if conf[t][p] else "")
                    for p in codes) + " |")
        lines += ["", f"Most frequent mistakes ({c}):", ""]
        for true, pred, n, share in top_confusions(recs):
            lines.append(f"- {names.get(true, true)} heard as {names.get(pred, pred)}: {n} clips ({share:.0%})")
    return "\n".join(lines) + "\n"
