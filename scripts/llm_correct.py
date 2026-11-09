"""Experiment: can an LLM (e.g. Llama) correct Whisper's transcripts from its n-best list?

    $env:LLM_BASE_URL="https://.../v1"; $env:LLM_API_KEY="..."; $env:LLM_MODEL="..."
    python scripts/llm_correct.py --max-per-set 10 --langs nso zul sot    # quick, cheap check
    python scripts/llm_correct.py                                         # full held-out half
    python scripts/llm_correct.py --dry-run                               # show one prompt, no calls

Reads the n-best files written by scripts/lm_rescore.py (results/lm_rescore/nbest/*.jsonl, five
beam candidates per clip, CPU only from here). For each clip the LLM sees all candidates and
returns one corrected transcript; `sa_asr.correction.guard` falls back to Whisper's top
candidate when the correction strays from the candidates. Scored on the odd-numbered clips,
the same held-out half as the MzansiLM summary, so the numbers compare directly.

Any OpenAI-compatible endpoint works (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL, or --env-file).
Responses are cached in results/llm_correct/<model>/responses.jsonl, so a rerun only calls the
model for clips it has not seen. Writes summary.md and corrections.jsonl next to it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jiwer

import sa_asr  # noqa: F401  (OS certificate store for corporate TLS proxies)
from sa_asr.config import PROJECT_ROOT, load_languages
from sa_asr.correction import build_messages, guard, parse_response
from sa_asr.text import normalize_for_scoring


def word_errors(ref: str, hyp: str) -> int:
    r, h = normalize_for_scoring(ref), normalize_for_scoring(hyp)
    if not r:
        return 0
    if not h:
        return len(r.split())
    o = jiwer.process_words(r, h)
    return o.substitutions + o.deletions + o.insertions


def load_env_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class ChatClient:
    def __init__(self, base_url: str, api_key: str, model: str, max_tokens: int, timeout: float) -> None:
        base = base_url.rstrip("/")
        self.url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        self.api_key, self.model, self.max_tokens, self.timeout = api_key, model, max_tokens, timeout

    def complete(self, messages: list[dict]) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0.0,
                           "max_tokens": self.max_tokens}).encode("utf-8")
        request = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                return content or ""
            except urllib.error.HTTPError as exc:
                if exc.code != 429 and exc.code < 500 or attempt == 3:
                    raise RuntimeError(f"LLM request failed with HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 3:
                    raise RuntimeError(f"LLM endpoint unreachable: {exc}") from exc
            time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")


def load_sets(nbest_dir: Path, langs: list[str] | None, half: str, max_per_set: int | None) -> dict[str, list[dict]]:
    sets = {}
    for path in sorted(nbest_dir.glob("*.jsonl")):
        if langs and path.stem.rsplit("_", 1)[1] not in langs:
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = rows[1::2] if half == "test" else rows[0::2] if half == "dev" else rows
        sets[path.stem] = rows[:max_per_set] if max_per_set else rows
    return sets


def mzansilm_pick(cands: list[dict], lam: float) -> dict:
    # The setting chosen in results/lm_rescore/summary.md: raw text, summed LM score.
    return max(cands, key=lambda c: c["asr"] + lam * c["lm_raw"])


def summarize(sets: dict[str, list[dict]], results: dict[str, dict], languages: dict, model: str, lam: float) -> str:
    cols = ("top", "mzansilm", "llm_raw", "llm_guarded", "oracle")
    totals = dict.fromkeys(cols, 0)
    total_words = fallbacks = 0
    reasons: dict[str, int] = {}
    rows_md = []
    for name, rows in sets.items():
        errs = dict.fromkeys(cols, 0)
        words = sum(r["ref_words"] for r in rows)
        set_fallbacks = 0
        for r in rows:
            res = results[f"{name}:{r['utt_id']}"]
            errs["top"] += r["cands"][0]["err"]
            errs["mzansilm"] += mzansilm_pick(r["cands"], lam)["err"]
            errs["oracle"] += min(c["err"] for c in r["cands"])
            errs["llm_raw"] += word_errors(r["reference"], res["raw"])
            errs["llm_guarded"] += word_errors(r["reference"], res["final"])
            if res["reason"] != "ok":
                set_fallbacks += 1
                reasons[res["reason"]] = reasons.get(res["reason"], 0) + 1
        for c in cols:
            totals[c] += errs[c]
        total_words += words
        fallbacks += set_fallbacks
        corpus, code = name.rsplit("_", 1)
        label = languages[code].name if code in languages else code
        cells = " | ".join(f"{100 * errs[c] / max(words, 1):.1f}" for c in cols)
        rows_md.append(f"| {corpus} / {label} | {len(rows)} | {cells} | {100 * set_fallbacks / max(len(rows), 1):.0f}% |")
    n = sum(len(v) for v in sets.values())
    overall = " | ".join(f"{100 * totals[c] / max(total_words, 1):.2f}" for c in cols)
    return "\n".join([
        f"# LLM n-best correction ({model})", "",
        f"{len(sets)} test sets, {n} clips. Each clip's 5 beam candidates go to the LLM; the guard keeps "
        "Whisper's top candidate when the correction adds unsupported words or changes the length.", "",
        "| | clips | top candidate | MzansiLM | LLM (unguarded) | LLM (guarded) | oracle (best of 5) | fell back |",
        "|---|---|---|---|---|---|---|---|",
        f"| **all** | {n} | {overall} | {100 * fallbacks / max(n, 1):.0f}% |",
        *rows_md, "",
        "WER %, lower is better. Fallback reasons: "
        + (", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) or "none") + ".",
        f"MzansiLM column: lambda={lam} on the same clips (from results/lm_rescore). LLM (guarded) can beat the "
        "oracle, because it can fix words that no candidate spelled right.",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nbest-dir", default="results/lm_rescore/nbest")
    ap.add_argument("--output-dir", default="results/llm_correct")
    ap.add_argument("--langs", nargs="+", help="ISO codes, e.g. nso zul sot (default: all sets)")
    ap.add_argument("--half", choices=["test", "dev", "all"], default="test",
                    help="test = odd clips, the held-out half used in the MzansiLM summary")
    ap.add_argument("--max-per-set", type=int, help="first N clips of each set: a cheap trial run")
    ap.add_argument("--no-hints", action="store_true", help="leave out the per-language spelling notes")
    ap.add_argument("--env-file", help="read LLM_BASE_URL / LLM_API_KEY / LLM_MODEL from this file")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--lm-lambda", type=float, default=0.05)
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and exit")
    args = ap.parse_args()

    languages = load_languages()
    sets = load_sets(PROJECT_ROOT / args.nbest_dir, args.langs, args.half, args.max_per_set)
    if not sets:
        raise SystemExit(f"No n-best files found in {args.nbest_dir} (run scripts/lm_rescore.py first).")

    def messages_for(name: str, row: dict) -> list[dict]:
        code = name.rsplit("_", 1)[1]
        name_ = languages[code].name if code in languages else code
        return build_messages(name_, code, [c["text"] for c in row["cands"]], hints=not args.no_hints)

    if args.dry_run:
        name, rows = next(iter(sets.items()))
        for m in messages_for(name, rows[0]):
            print(f"--- {m['role']}\n{m['content']}")
        return

    if args.env_file:
        load_env_file(Path(args.env_file))
    missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Set {', '.join(missing)} (environment or --env-file).")
    model = os.environ["LLM_MODEL"]
    client = ChatClient(os.environ["LLM_BASE_URL"], os.environ["LLM_API_KEY"], model, args.max_tokens, args.timeout)

    out_dir = PROJECT_ROOT / args.output_dir / re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "responses.jsonl"
    cache: dict[str, str] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                cache[item["key"]] = item["response"]

    todo = [(name, row) for name, rows in sets.items() for row in rows if f"{name}:{row['utt_id']}" not in cache]
    print(f"{sum(len(v) for v in sets.values())} clips in {len(sets)} sets; {len(todo)} need the model ({model}).")
    lock = threading.Lock()
    done = failed = 0
    started = time.time()

    def work(item: tuple[str, dict]) -> None:
        nonlocal done, failed
        name, row = item
        key = f"{name}:{row['utt_id']}"
        try:
            response = client.complete(messages_for(name, row))
        except RuntimeError as exc:
            with lock:
                failed += 1
                print(f"  {key}: {exc}", flush=True)
            return
        with lock:
            cache[key] = response
            with cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(todo)} in {time.time() - started:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    if failed:
        raise SystemExit(f"{failed} requests failed; rerun to retry them (finished ones are cached).")

    results: dict[str, dict] = {}
    with (out_dir / "corrections.jsonl").open("w", encoding="utf-8") as handle:
        for name, rows in sets.items():
            for row in rows:
                key = f"{name}:{row['utt_id']}"
                candidates = [c["text"] for c in row["cands"]]
                raw = parse_response(cache[key])
                ok, reason = guard(raw, candidates)
                results[key] = {"raw": raw, "final": raw if ok else candidates[0], "reason": reason}
                handle.write(json.dumps({"key": key, "reference": row["reference"], "top": candidates[0],
                                         **results[key]}, ensure_ascii=False) + "\n")

    md = summarize(sets, results, languages, model, args.lm_lambda)
    (out_dir / "summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
