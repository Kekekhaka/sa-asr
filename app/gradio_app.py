"""Speech-to-text for South African languages, Gradio version (for a Hugging Face Space on ZeroGPU).

    python app/gradio_app.py                      # locally (CPU, or GPU if there is one)
    python scripts/space.py push --repo <user>/sa-asr-app

Same engine as app/streamlit_app.py (which stays the local app): language detection part by part, the optional
second model for clear speech, translation to English, and corrections saved with the language of every part.
On ZeroGPU the models are loaded once at start-up and each transcription borrows a GPU for up to GPU_SECONDS.
Saved corrections are copied to the private dataset SA_ASR_FEEDBACK_REPO, because a Space's disk is wiped on restart.
"""

from __future__ import annotations

import html
import os
import sys
import tempfile
from pathlib import Path

# On a Space the package is not installed: use src/ next to the app.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr
import numpy as np

from sa_asr.audio import duration_seconds
from sa_asr.config import load_languages
from sa_asr.feedback import save_correction, summary, upload_correction
from sa_asr.inference import load_audio, load_transcriber, resolve_model, transcribe_audio
from sa_asr.rescore import LAMBDAS, MZANSILM, RescoringTranscriber, TextLM
from sa_asr.routing import DSFSI_MODEL, MIN_SECONDS, load_second_model
from sa_asr.textlid import DEFAULT_PATH as TEXT_LID_PATH, TextLID
from sa_asr.translate import can_translate, translate_segments

try:  # on a ZeroGPU Space; a no-op locally
    import spaces

    gpu = spaces.GPU
except ImportError:
    def gpu(duration=None):
        return lambda fn: fn

MODEL_ID = os.environ.get("SA_ASR_MODEL", "KekeCode/sa-asr-whisper-lora-v6")
FEEDBACK_REPO = os.environ.get("SA_ASR_FEEDBACK_REPO")
# The second model costs ~3 GB and start-up time; set SA_ASR_SECOND=0 to leave it out.
WITH_SECOND = os.environ.get("SA_ASR_SECOND", "1") != "0"
# "Extra accuracy": MzansiLM picks among Whisper's 5 best (sa_asr.rescore); ~0.5 GB, SA_ASR_RESCORE=0 leaves it out.
WITH_RESCORE = os.environ.get("SA_ASR_RESCORE", "1") != "0"
GPU_SECONDS = 120          # the most one transcription may hold the GPU (counts against the daily ZeroGPU quota)
MAX_MINUTES = 10           # longer recordings are refused, so a run fits in GPU_SECONDS
RESCORE_MINUTES = 4        # rescoring takes 2-3x as long, so only recordings up to this length get it
AUTO = "Detect automatically"

registry = load_languages()
model_ref, trained_langs = resolve_model(MODEL_ID)
LANGS = trained_langs or list(registry)
NAMES = {registry[c].name: c for c in LANGS}
# typed languages: the full name, its first word ("English" for "English (South African)") or the code, any case
LOOKUP = {**{n.split(" (")[0].lower(): c for n, c in NAMES.items()}, **{n.lower(): c for n, c in NAMES.items()},
          **{c: c for c in LANGS}}
# one colour per language, used for the chips, the language bar and the timeline
COLORS = {"eng": "#3B82F6", "afr": "#F97316", "zul": "#22C55E", "xho": "#A855F7", "nso": "#EAB308",
          "sot": "#06B6D4", "tsn": "#EC4899", "tso": "#14B8A6", "ven": "#DC2626", "ssw": "#84CC16", "nbl": "#6366F1"}

# Loaded once at start-up (on ZeroGPU, "cuda" is available here and the weights move to the GPU on each call).
MODEL = load_transcriber(model_ref, trained_langs is not None)
TEXT_LID = TextLID.load() if TEXT_LID_PATH.exists() else None
SECOND = load_second_model() if WITH_SECOND else None
RESCORER = RescoringTranscriber(MODEL, TextLM.load()) if WITH_RESCORE else None


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def short(code: str | None) -> str:
    return registry[code].name.split(" (")[0] if code in registry else "Unknown"


def chip(code: str | None, text: str | None = None) -> str:
    return (f'<span class="sa-chip" style="--c:{COLORS.get(code or "", "#94A3B8")}"><i></i>'
            f'{html.escape(text or short(code))}</span>')


def gpu_seconds(wav: np.ndarray, lang_code: str | None, use_second: bool, beams: int, rescore: bool = False) -> int:
    """GPU time to ask ZeroGPU for, from the recording's length. ZeroGPU only starts a call if the quota left covers
    the whole request, so asking for GPU_SECONDS every time refused short clips long before the quota was used up.
    An 18 s clip took 1.7 s; this allows 20 s plus 12 s per minute of audio, more for the slower options."""
    need = 20 + 12 * duration_seconds(wav) / 60
    need *= (2.5 if rescore else 1) * (1.5 if use_second else 1) * (1 + 0.25 * (max(beams, 1) - 1))
    return int(min(GPU_SECONDS, need))


@gpu(duration=gpu_seconds)
def run_model(wav: np.ndarray, lang_code: str | None, use_second: bool, beams: int, rescore: bool = False) -> dict:
    MODEL.num_beams = beams
    model = RESCORER if rescore and RESCORER is not None else MODEL
    return transcribe_audio(model, wav, registry, lang_code, mixed=lang_code is None, text_lid=TEXT_LID,
                            second_model=SECOND if use_second else None)


def summary_card(stats: list[tuple[str, str]], title: str, parts: list[tuple[str, float]], legend: str, note: str) -> str:
    """The summary card always has the same rows (tiles, label, bar, one line of chips, a two-line note), so it keeps
    its height whatever the result: the page must not change size when a result arrives (phones shook)."""
    bar = "".join(f'<i style="width:{100 * p:.1f}%;background:{COLORS.get(c, "#94A3B8")}"></i>' for c, p in parts)
    return ('<div class="sa-card sa-summary"><div class="sa-stats">'
            + "".join(f'<div class="sa-stat"><span>{k}</span><b>{html.escape(v)}</b></div>' for k, v in stats)
            + f'</div><div class="sa-label">{title}</div><div class="sa-bar">{bar}</div>'
            f'<div class="sa-legend">{legend or "&nbsp;"}</div><div class="sa-note">{note or "&nbsp;"}</div></div>')


def summary_html(result: dict, shares: dict[str, float], chosen: bool, note: str = "") -> str:
    """Stat tiles, then a bar of the languages: share of speaking time if mixed, else detection confidence."""
    mixed = len(shares) > 1
    if mixed:
        lang_value, title, parts = f"{len(shares)} languages", "Language mix · share of speaking time", list(
            shares.items())
        legend = "".join(chip(c, f"{short(c)} {p:.0%}") for c, p in parts)
    elif chosen:
        lang_value, title = short(result["language"]), "You chose the language"
        parts, legend = [(result["language"], 1.0)], chip(result["language"])
    else:
        lang_value = short(result["language"]) if result["language"] else "Chosen by Whisper"
        title, parts = "Detection confidence", list((result["detected"] or {}).items())[:3]
        legend = "".join(chip(c, f"{short(c)} {p:.0%}") for c, p in parts)
    stats = [("Language", lang_value), ("Length", fmt_time(result["duration"])),
             ("Took", f"{result['elapsed']:.1f} s")]
    notes = []
    if result.get("rerouted"):
        notes.append(f'{result["rerouted"]} clear part(s) were transcribed by the second model '
                     f"({html.escape(DSFSI_MODEL)}, MIT licence).")
    if note:
        notes.append(html.escape(note))
    return summary_card(stats, title, parts, legend, " ".join(notes))


EMPTY_SUMMARY = summary_card([("Language", "—"), ("Length", "—"), ("Took", "—")], "Detection confidence", [], "",
                             "Your result appears here.")


def timeline_html(segs: list[dict], mixed: bool) -> str:
    rows = []
    for s in segs:
        meta = f'<span class="sa-time">{fmt_time(s["start"])}–{fmt_time(s["end"])}</span>'
        if s.get("language"):
            meta += chip(s["language"])
        if s.get("engine"):
            meta += '<span class="sa-tag">second model</span>'
        color = COLORS.get(s.get("language") or "", "#94A3B8") if mixed else "var(--sa-accent)"
        rows.append(f'<div class="sa-seg" style="--c:{color}"><div class="sa-seg-meta">{meta}</div>'
                    f'<p>{html.escape(s["text"])}</p></div>')
    return '<div class="sa-timeline">' + "".join(rows) + "</div>"


def transcribe(uploaded, recorded, lang_name, use_second, beams, extra=False):
    audio_path = recorded or uploaded
    if not audio_path:
        raise gr.Error("No audio yet. Upload a file, or record a voice message. If you recorded and the player "
                       "shows 0:00, the browser got no sound from the microphone: allow the microphone (icon at the "
                       "left of the address bar), check the right input device is selected, and record again.",
                       duration=None)
    wav = load_audio(Path(audio_path).read_bytes(), Path(audio_path).name)
    seconds = duration_seconds(wav)
    if seconds < 0.3:
        raise gr.Error("The audio is too short to transcribe.")
    if seconds > MAX_MINUTES * 60:
        raise gr.Error(f"Recordings up to {MAX_MINUTES} minutes only (this one is {fmt_time(seconds)}).")
    chosen = lang_name != AUTO
    rescore = bool(extra) and RESCORER is not None and seconds <= RESCORE_MINUTES * 60
    note = (f"Extra accuracy is used for recordings up to {RESCORE_MINUTES} minutes; this one was transcribed normally."
            if extra and RESCORER is not None and not rescore else "")
    try:
        result = run_model(wav, NAMES[lang_name] if chosen else None, use_second, int(beams), rescore)
    except Exception as e:
        if "quota" not in str(e).lower():
            raise
        raise gr.Error("The free GPU time for today is used up for this account (Hugging Face's ZeroGPU quota). It "
                       "refills over the day; shorter recordings need less of it, and the Extra accuracy and second "
                       f"model options more. Details: {e}", duration=None) from e

    shares = result.get("languages") or {}
    mixed = len(shares) > 1
    segs = [s for s in result["segments"] if s["text"]] or [
        {"start": 0.0, "end": result["duration"], "language": result["language"], "text": ""}]
    rows = [[f"{fmt_time(s['start'])}–{fmt_time(s['end'])}",
             registry[s["language"]].name if s.get("language") in LANGS else registry[LANGS[0]].name,
             s["text"]] for s in segs]
    txt = Path(tempfile.mkdtemp()) / "transcript.txt"
    txt.write_text(result["text"], encoding="utf-8")
    state = {"wav": wav, "segments": segs, "language": result["language"], "text": result["text"]}
    heard = sorted({r[1] for r in rows})
    # Everything below is already on the page and only its contents change (no showing, hiding or opening), so the
    # page keeps its height when the result arrives.
    return (summary_html(result, shares, chosen, note),
            result["text"] or "(no speech recognised)", gr.DownloadButton(value=str(txt), interactive=True),
            gr.Accordion(label=f"Timeline · {len(segs)} part{'s' if len(segs) > 1 else ''}"),
            timeline_html(segs, mixed), gr.Dataframe(value=rows, row_count=(len(rows), "fixed")),
            gr.update(value=heard), state, "", "")


def translate(state):
    if not state or not state["text"]:
        raise gr.Error("Transcribe something first.")
    if not any(can_translate(s.get("language")) or s.get("language") == "eng" for s in state["segments"]):
        raise gr.Error("No translation model covers this language.")
    return " ".join(p for p in translate_segments(state["segments"]) if p)


def _rows(table) -> list:
    return table.values.tolist() if hasattr(table, "values") else (table or [])


def words_html(words: list) -> str:
    if not words:
        return ""
    return '<div class="sa-words">' + "".join(chip(c, w) for w, c in words) + "</div>"


def refresh_words(table):
    """Every word of the corrected parts, each with its part's language (as the Streamlit word table)."""
    words = [[w, LOOKUP.get(str(lang_name).strip().lower())] for _, lang_name, text in _rows(table)
             for w in str(text).split()]
    return words, gr.CheckboxGroup(choices=[(w, i) for i, (w, _) in enumerate(words)], value=[]), words_html(words)


def set_word_lang(words, picked, lang_name, heard):
    if not picked:
        raise gr.Error("Tick the words first.")
    for i in picked:
        words[i][1] = NAMES[lang_name]
    return (words, gr.CheckboxGroup(value=[]), words_html(words),
            gr.update(value=sorted(set(heard or []) | {lang_name})))


def save(state, table, words, heard, allowed):
    if not state:
        raise gr.Error("Transcribe something first.")
    if not allowed:
        raise gr.Error("Tick the permission box first.")
    parts = []
    for seg, (_, lang_name, text) in zip(state["segments"], _rows(table)):
        lang_name = str(lang_name).strip()
        code = LOOKUP.get(lang_name.lower())
        if code is None:
            raise gr.Error(f"Unknown language '{lang_name}'. Use one of: {', '.join(NAMES)}")
        if str(text).strip():
            parts.append({"start": seg["start"], "end": seg["end"], "lang": code, "text": str(text).strip()})
    if not parts:
        raise gr.Error("The corrected text is empty.")
    text = " ".join(p["text"] for p in parts)
    seconds: dict[str, float] = {}
    for p in parts:
        seconds[p["lang"]] = seconds.get(p["lang"], 0.0) + p["end"] - p["start"]
    main = max(seconds, key=seconds.get)

    # every word takes its part's language, unless it was given another one in the word picker
    defaults = [(w, p["lang"]) for p in parts for w in p["text"].split()]
    picked = words if words and [w for w, _ in words] == [w for w, _ in defaults] else []
    word_langs = [(picked[i][1] if picked and picked[i][1] else lang) for i, (_, lang) in enumerate(defaults)]

    rec = save_correction(state["wav"], main, text, predicted_lang=state["language"], predicted_text=state["text"],
                          model=MODEL_ID, languages_present=[NAMES[n] for n in heard or []],
                          word_langs=word_langs, segments=parts)
    if FEEDBACK_REPO:
        try:
            upload_correction(rec, FEEDBACK_REPO)
        except Exception as e:
            raise gr.Error(f"Saved here, but couldn't copy it to {FEEDBACK_REPO} (it will be lost on restart): {e}")
    stats = summary()
    langs = ", ".join(registry[c].name for c in rec["languages_present"])
    return (f'<div class="sa-ok">✓ Saved ({html.escape(langs)}). {stats["count"]} correction(s) so far in this '
            f'session{f"; copied to {html.escape(FEEDBACK_REPO)}" if FEEDBACK_REPO else ""}. Thank you!</div>')


# While a transcription runs: an animated panel with the steps and a timer, the button disabled. All of it is CSS:
# changing the page from JavaScript makes the Space's iframe re-measure its height, which shook the page on phones.
BUSY = """<div class="sa-card sa-busy">
  <div class="sa-eq"><i></i><i></i><i></i><i></i><i></i></div>
  <div class="sa-busy-text"><b>Transcribing…</b>
    <div class="sa-steps"><em>Reading the audio</em><em>Detecting the language of each part</em>
      <em>Writing the transcript</em></div></div>
  <div class="sa-elapsed"></div>
  <div class="sa-progress"><i></i></div>
</div>"""
# On the Space page the app sits in an iframe that Gradio sizes through iframe-resizer (window.parentIFrame.size),
# with fractional heights measured partly from the iframe's own height. On iPhones that can feed back and the page
# shakes non-stop. This wraps size(): whole pixels, changes under 4 px dropped, and a height that bounces back within
# a second (A -> B -> A) keeps the larger value. Locally there is no parentIFrame and it does nothing.
HEAD = """<script>
(() => {
  if (window.__saHead) return;  // the page can carry this script twice (in the HTML and added again by Gradio)
  window.__saHead = true;
  const wrap = p => {
    if (!p || p.__saWrapped || typeof p.size !== 'function') return;
    const size = p.size.bind(p);
    let last = 0, prev = 0, changed = 0;
    p.size = (h, w) => {
      if (h == null) return size(h, w);
      h = Math.ceil(h);
      if (Math.abs(h - last) < 4) return;
      const now = Date.now();
      if (Math.abs(h - prev) < 4 && now - changed < 1000) h = Math.max(h, last);
      if (h === last) return;
      prev = last; last = h; changed = now;
      return size(h, w);
    };
    p.__saWrapped = true;
  };
  let tries = 0;
  const poll = setInterval(() => { wrap(window.parentIFrame); if (window.parentIFrame?.__saWrapped || ++tries > 150) clearInterval(poll); }, 100);
  // (No width lock: locking the app to the frame's width on load could catch an iPhone frame while it was too wide,
  // and since iOS sizes the frame to its content the two then held each other there with a quarter of the app cut
  // off. The shake's cause, the audio player sticking out past the screen, is fixed in the CSS instead.)
  // Inside the Space page's frame, show "Open full screen": the frame's address (with a private Space's signed
  // token) opened as its own page, where iOS Safari cannot stretch and shrink the frame.
  // TEMPORARY (iPhone shake hunt): a corner readout of width, zoom, page height and resize events per second, so a
  // screen recording shows which of them flips. Fixed position and outside the app, so it moves nothing itself.
  const readout = () => {
    const box = document.createElement('div');
    box.style.cssText = 'position:fixed;left:4px;top:4px;z-index:99999;pointer-events:none;font:11px/1.3 monospace;' +
      'background:rgba(0,0,0,.75);color:#0f0;padding:3px 5px;border-radius:4px;white-space:pre';
    document.body.appendChild(box);
    let rz = 0, vrz = 0, sent = 0;
    addEventListener('resize', () => rz++);
    window.visualViewport && visualViewport.addEventListener('resize', () => vrz++);
    const p = window.parentIFrame;
    if (p && p.size) { const s = p.size; p.size = (...a) => { sent++; return s(...a); }; }
    // the element sticking out furthest to the right, if anything does (what would widen the frame)
    const widest = () => {
      let best = null, right = innerWidth + 1;
      for (const el of document.body.querySelectorAll('*')) {
        const r = el.getBoundingClientRect().right;
        if (r > right) { right = r; best = el; }
      }
      return best ? `${best.tagName.toLowerCase()}.${[...best.classList].slice(0, 2).join('.')} r${Math.round(right)}` : 'none';
    };
    setInterval(() => {
      const v = window.visualViewport, d = document.documentElement;
      box.textContent = `${window.self !== window.top ? 'FRAME' : 'FULL'} w${innerWidth} cw${d.clientWidth} sw${d.scrollWidth}` +
        ` sc${screen.width}` +
        `\\nzoom ${v ? v.scale.toFixed(2) : '?'} vw${v ? Math.round(v.width) : '?'} h${d.scrollHeight}` +
        `\\nresize/s ${rz} vv/s ${vrz} size()/s ${sent}\\nwide: ${widest()}`;
      rz = vrz = sent = 0;
    }, 1000);
  };
  // Gradio may run this script after the page has loaded, so the load event can already be over
  document.readyState === 'complete' ? readout() : window.addEventListener('load', readout);
  if (window.self !== window.top) {
    document.documentElement.classList.add('sa-framed');
    document.addEventListener('click', e => {
      if (e.target.closest && e.target.closest('.sa-full')) { e.preventDefault(); window.open(location.href, '_blank'); }
    });
  }
})();
</script>"""
# The same box, at rest: the status slot always holds one of the two, so it never appears or disappears.
READY = """<div class="sa-card sa-busy sa-idle">
  <div class="sa-eq"><i></i><i></i><i></i><i></i><i></i></div>
  <div class="sa-busy-text"><b>Ready</b><div class="sa-steps"><em>The result appears below.</em></div></div>
</div>"""


def busy():
    return BUSY, gr.Button("Transcribing…", interactive=False)


def idle():
    return READY, gr.Button("Transcribe", interactive=True)


def step(n: int, title: str, hint: str = "") -> gr.HTML:
    return gr.HTML(f'<div class="sa-step"><span>{n}</span><div><b>{title}</b>'
                   + (f"<small>{hint}</small>" if hint else "") + "</div></div>")


HERO = (
    '<div class="sa-hero"><div class="sa-flag"></div>'
    '<div class="sa-badge">🇿🇦 11 official languages · code-switching aware</div>'
    '<div><a class="sa-full" href="#">⤢ Open full screen</a></div>'
    "<h1>Speak any South African language.<br><span>Get the words back.</span></h1>"
    "<p>Record a voice note or upload audio. Every part is transcribed in its own language, and you can "
    "translate it to English.</p>"
    '<div class="sa-langs">' + "".join(chip(c) for c in LANGS) + "</div></div>")

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.rose, neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg, spacing_size=gr.themes.sizes.spacing_md,
).set(
    button_primary_background_fill="linear-gradient(135deg, #FF4B4B 0%, #FF7A45 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #FF4B4B 0%, #FF7A45 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #FF3333 0%, #FF6A30 100%)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #FF3333 0%, #FF6A30 100%)",
    button_primary_text_color="white", button_primary_text_color_dark="white",
    button_primary_border_color="transparent", button_primary_border_color_dark="transparent",
    block_radius="18px", block_border_width="1px", button_large_radius="14px", button_small_radius="10px",
    input_radius="12px",
)
CSS = """
:root { --sa-accent: #FF4B4B; --sa-accent-2: #FF7A45; }
.gradio-container { max-width: 720px !important; margin: 0 auto !important; padding: 0 16px 48px !important;
  background: radial-gradient(900px 420px at 50% -120px, rgba(255, 75, 75, .16), transparent 70%) !important; }
.gradio-container .prose, .gradio-container .html-container { overflow: visible !important; }
/* iOS Safari widens an iframe to its content and enlarges text on its own; either one made the Space page's frame
   flip between two widths. Nothing may stick out sideways, and text keeps its size. */
html, body { overflow-x: hidden !important; max-width: 100%; -webkit-text-size-adjust: 100%; text-size-adjust: 100%; }
.gradio-container, .gradio-container * { min-width: 0; }
/* The audio player that appears after an upload or a recording: Gradio gives its box width 100% plus padding, so on
   a phone it stuck out past the screen, the iPhone frame widened, the waveform redrew at the new width and stuck out
   again (the shake started on upload). Padding counts inside the width, and nothing in the player sticks out past its
   box. */
.gradio-container .component-wrapper, .gradio-container .waveform-container, .gradio-container #waveform,
.gradio-container .timestamps, .gradio-container .controls { box-sizing: border-box !important; max-width: 100% !important; }
.gradio-container .component-wrapper, .gradio-container .waveform-container, .gradio-container #waveform {
  overflow: hidden !important; }
.gradio-container canvas, .gradio-container audio { max-width: 100% !important; }
.sa-full { display: none; margin-top: 10px; padding: 6px 14px; border-radius: 999px; font-size: .82rem; font-weight: 600;
  color: var(--sa-accent) !important; text-decoration: none !important; border: 1px solid var(--sa-accent); }
.sa-framed .sa-full { display: inline-block !important; }  /* Gradio also adds a stronger, prefixed copy of each rule */
footer { display: none !important; }

/* hero */
.sa-hero { position: relative; padding: 28px 0 8px; text-align: center; }
.sa-flag { height: 4px; width: 96px; margin: 0 auto 18px; border-radius: 4px;
  background: linear-gradient(90deg, #007A4D 0 25%, #FFB612 25% 45%, #DE3831 45% 70%, #002395 70% 100%); }
.sa-badge { display: inline-block; padding: 6px 14px; border-radius: 999px; font-size: .8rem; font-weight: 600;
  border: 1px solid var(--border-color-primary); background: var(--block-background-fill); }
.sa-hero h1 { font-size: clamp(1.7rem, 5.5vw, 2.7rem); line-height: 1.12; font-weight: 800; letter-spacing: -.02em;
  margin: 16px 0 10px; }
.sa-hero h1 span { background: linear-gradient(135deg, var(--sa-accent), var(--sa-accent-2));
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.sa-hero p { color: var(--body-text-color-subdued); max-width: 520px; margin: 0 auto 16px; font-size: 1rem; }
.sa-langs { display: flex; flex-wrap: wrap; justify-content: center; gap: 6px; }

/* language chips: a coloured dot and a tinted pill */
.sa-chip { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
  font-size: .78rem; font-weight: 600; white-space: nowrap;
  background: color-mix(in srgb, var(--c) 14%, transparent); border: 1px solid color-mix(in srgb, var(--c) 35%, transparent); }
.sa-chip i { width: 7px; height: 7px; border-radius: 50%; background: var(--c); }

/* numbered steps and cards */
.sa-step { display: flex; align-items: center; gap: 12px; margin: 14px 0 2px; }
.sa-step > span { flex: none; width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center;
  font-weight: 700; font-size: .85rem; color: white; background: linear-gradient(135deg, var(--sa-accent), var(--sa-accent-2)); }
.sa-step b { display: block; font-size: 1.05rem; }
.sa-step small { color: var(--body-text-color-subdued); font-size: .82rem; }
.sa-card { border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
  border-radius: 18px; padding: 18px; }
.sa-hint { color: var(--body-text-color-subdued); font-size: .82rem; margin: 2px 4px 0; }

/* the main button */
#sa-go { min-height: 56px; font-size: 1.1rem; font-weight: 700; margin-top: 10px;
  box-shadow: 0 10px 30px -10px rgba(255, 75, 75, .65); transition: transform .15s ease, box-shadow .15s ease; }
#sa-go:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 14px 34px -10px rgba(255, 75, 75, .8); }
#sa-go:disabled { opacity: .75; cursor: progress; }

/* while transcribing */
/* fixed size and transform/opacity-only animations, so nothing re-flows the page while it runs (phones shook) */
.sa-busy { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; align-items: center; gap: 16px;
  position: relative; overflow: hidden; margin-top: 4px; height: 76px; box-sizing: border-box; contain: strict; }
.sa-eq { display: flex; align-items: center; gap: 4px; height: 32px; }
.sa-eq i { width: 5px; height: 100%; border-radius: 3px; background: linear-gradient(var(--sa-accent), var(--sa-accent-2));
  animation: sa-eq 1s ease-in-out infinite; will-change: transform; }
.sa-eq i:nth-child(2) { animation-delay: -.8s } .sa-eq i:nth-child(3) { animation-delay: -.6s }
.sa-eq i:nth-child(4) { animation-delay: -.4s } .sa-eq i:nth-child(5) { animation-delay: -.2s }
@keyframes sa-eq { 0%, 100% { transform: scaleY(.25) } 50% { transform: scaleY(1) } }
.sa-busy-text { min-width: 0; }
.sa-busy-text b { font-size: 1.05rem; white-space: nowrap; }
.sa-steps { position: relative; height: 1.4em; overflow: hidden; color: var(--body-text-color-subdued); font-size: .88rem; }
.sa-steps em { position: absolute; left: 0; right: 0; top: 0; font-style: normal; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; opacity: 0; animation: sa-step 9s infinite; }
.sa-steps em:nth-child(2) { animation-delay: 3s } .sa-steps em:nth-child(3) { animation-delay: 6s }
@keyframes sa-step { 0% { opacity: 0 } 4%, 30% { opacity: 1 } 34%, 100% { opacity: 0 } }
@property --sa-t { syntax: "<integer>"; inherits: false; initial-value: 0; }
.sa-elapsed { font-variant-numeric: tabular-nums; font-weight: 700; color: var(--sa-accent); min-width: 5ch;
  text-align: right; white-space: nowrap; counter-reset: sa-t var(--sa-t); animation: sa-tick 900s steps(900) forwards; }
.sa-elapsed::after { content: counter(sa-t) " s"; }
@keyframes sa-tick { to { --sa-t: 900 } }
.sa-progress { position: absolute; left: 0; right: 0; bottom: 0; height: 3px; overflow: hidden; }
.sa-progress i { position: absolute; left: 0; height: 100%; width: 35%; border-radius: 3px;
  background: linear-gradient(90deg, transparent, var(--sa-accent), var(--sa-accent-2), transparent);
  animation: sa-slide 1.4s ease-in-out infinite; will-change: transform; }
@keyframes sa-slide { from { transform: translateX(-100%) } to { transform: translateX(290%) } }
@media (prefers-reduced-motion: reduce) { .sa-eq i, .sa-progress i { animation: none; } }
.sa-idle .sa-eq i { animation: none; transform: scaleY(.25); opacity: .5; }
.sa-idle .sa-steps em { opacity: 1; animation: none; }

/* results */
#sa-results { scroll-margin-top: 12px; }
.sa-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.sa-stat { padding: 12px 14px; border-radius: 14px; background: var(--background-fill-secondary); min-width: 0; }
.sa-stat span { display: block; font-size: .75rem; text-transform: uppercase; letter-spacing: .06em;
  color: var(--body-text-color-subdued); }
.sa-stat b { display: block; font-size: 1.35rem; font-weight: 700; margin-top: 2px; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.sa-label { margin: 16px 0 8px; font-size: .82rem; color: var(--body-text-color-subdued); }
.sa-bar { display: flex; height: 10px; border-radius: 999px; overflow: hidden; background: var(--background-fill-secondary); }
.sa-bar i { height: 100%; } .sa-bar i + i { margin-left: 2px; }
/* one line of chips (scrolls sideways inside itself if they do not fit) and a two-line note: the card's height is fixed */
.sa-legend { display: flex; flex-wrap: nowrap; gap: 6px; margin-top: 10px; overflow-x: auto; overflow-y: hidden;
  scrollbar-width: none; min-height: 26px; }
.sa-legend::-webkit-scrollbar { display: none; }
.sa-label { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sa-note { margin-top: 12px; font-size: .82rem; line-height: 1.4; height: 2.8em; overflow: hidden;
  color: var(--body-text-color-subdued); }
#sa-transcript textarea { font-size: 1.08rem !important; line-height: 1.6 !important; }
.sa-timeline { display: flex; flex-direction: column; gap: 10px; }
.sa-seg { border-left: 3px solid var(--c); padding: 4px 0 4px 12px; }
.sa-seg-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 4px; }
.sa-seg p { margin: 0; line-height: 1.55; }
.sa-time { font-variant-numeric: tabular-nums; font-size: .78rem; font-weight: 600; color: var(--body-text-color-subdued); }
.sa-tag { font-size: .72rem; padding: 2px 8px; border-radius: 999px; background: var(--background-fill-secondary); }
.sa-words { display: flex; flex-wrap: wrap; gap: 6px; padding: 4px 12px 12px; }
#sa-word-pick label { padding: 6px 12px !important; border-radius: 999px !important; }
.sa-ok { padding: 12px 14px; border-radius: 12px; font-weight: 600; color: #15803D;
  background: rgba(34, 197, 94, .12); border: 1px solid rgba(34, 197, 94, .35); }
.sa-foot { text-align: center; font-size: .78rem; color: var(--body-text-color-subdued); margin-top: 28px; }

/* phones */
@media (max-width: 640px) {
  .gradio-container { padding: 0 12px 40px !important; }
  .sa-hero { padding-top: 40px; }
  .sa-hero p { font-size: .92rem; }
  .sa-stats { gap: 6px; }
  .sa-stat { padding: 10px; }
  .sa-stat b { font-size: 1.02rem; }
  .sa-card { padding: 14px; border-radius: 16px; }
  .sa-busy { gap: 12px; padding: 14px; }
  #sa-go { min-height: 60px; }
}
"""

with gr.Blocks(title="South African Speech-to-Text") as demo:
    state = gr.State(None)
    words = gr.State([])   # [word, language code] for every word of the corrected parts
    with gr.Sidebar(open=False, label="Settings"):
        gr.Markdown(f"### Settings\n✅ Fine-tuned South African model, {len(LANGS)} languages  \n`{MODEL_ID}`")
        beams = gr.Slider(1, 5, value=1, step=1, label="Beam size", info="Higher is slightly more accurate and slower.")
        second = gr.Checkbox(
            value=False, visible=SECOND is not None, label="Second model for clear speech",
            info=f"For isiZulu, isiXhosa, Sesotho, Setswana, Xitsonga and Tshivenda: parts of a recording that are at "
                 f"least {MIN_SECONDS:.0f} s long and not telephone quality are transcribed again by {DSFSI_MODEL} "
                 "(MIT licence, trained on Swivuriso), about 5 points more accurate on such speech in our tests.")
        extra = gr.Checkbox(
            value=False, visible=RESCORER is not None, label="Extra accuracy (slower)",
            info=f"Whisper writes its 5 best versions of each part and a South African language model ({MZANSILM}) "
                 f"picks the most natural one: about 4% fewer word errors in our tests, 2-3x slower. Used for "
                 f"{', '.join(registry[c].name.split(' (')[0] for c in LAMBDAS)}; recordings up to "
                 f"{RESCORE_MINUTES} minutes.")

    gr.HTML(HERO)

    step(1, "Your audio", f"A voice note, a call or a meeting, up to {MAX_MINUTES} minutes.")
    with gr.Tabs():
        with gr.Tab("🎙️ Record"):
            recorded = gr.Audio(sources=["microphone"], type="filepath", show_label=False, buttons=["download"])
            gr.HTML('<div class="sa-hint">Tap the microphone, speak, then tap stop. Allow microphone access when '
                    "your browser asks.</div>")
        with gr.Tab("📁 Upload"):
            uploaded = gr.Audio(sources=["upload"], type="filepath", show_label=False, buttons=["download"])

    step(2, "Language", "Leave on automatic for mixed languages: it is detected part by part.")
    lang = gr.Dropdown([AUTO] + list(NAMES), value=AUTO, show_label=False, container=False)

    go = gr.Button("Transcribe", variant="primary", size="lg", elem_id="sa-go")
    status = gr.HTML(READY)

    # The whole result area is on the page from the start, in fixed-size boxes; a result only fills them in (like a
    # Streamlit page), so nothing appears, opens or grows when it arrives. Phones shook when the page changed size.
    with gr.Column(elem_id="sa-results"):
        step(3, "Result")
        summary_box = gr.HTML(EMPTY_SUMMARY)
        transcript = gr.Textbox(label="Transcript", lines=6, max_lines=6, autoscroll=False, buttons=["copy"],
                                placeholder="The transcript appears here.", elem_id="sa-transcript")
        download = gr.DownloadButton("⬇ Download .txt", interactive=False, variant="secondary", size="md")
        tr_btn = gr.Button("🌍 Translate to English", variant="secondary", size="md")
        translation = gr.Textbox(label="English translation", lines=3, max_lines=3, autoscroll=False,
                                 buttons=["copy"], placeholder="Press Translate to English after a transcription.")
        with gr.Accordion("Timeline", open=False) as seg_box:
            timeline = gr.HTML('<div class="sa-hint">Each part of the recording, with its language.</div>')

        with gr.Accordion("✏️ Wrong language or words? Help improve the model", open=False):
            gr.HTML('<div class="sa-hint">Fix the language and the text of each part. Saved corrections (audio + '
                    "text) are kept privately and used to improve the model. Only save recordings you have permission "
                    "to use.</div>")
            parts = gr.Dataframe(headers=["Time", "Language", "Text"], datatype=["str", "str", "str"],
                                 column_count=(3, "fixed"), interactive=True, wrap=True, label="Each part",
                                 column_widths=["16%", "24%", "60%"])
            gr.HTML('<div class="sa-hint">Languages: ' + " · ".join(NAMES) + "</div>")
            cs = gr.Checkbox(label="Some words are in another language (code-switching)")
            with gr.Group(visible=False) as cs_box:
                gr.HTML('<div class="sa-hint" style="padding:10px 12px 0">Tick the words that are in another '
                        "language (e.g. English words inside a Sepedi sentence), choose their language, then "
                        "press Set. Each word is shown in its language's colour.</div>")
                word_pick = gr.CheckboxGroup([], show_label=False, elem_id="sa-word-pick")
                with gr.Row(equal_height=True):
                    word_lang = gr.Dropdown(list(NAMES), value=registry["eng"].name if "eng" in LANGS
                                            else list(NAMES)[0], show_label=False, container=False, scale=3)
                    set_btn = gr.Button("Set", variant="secondary", scale=1)
                word_view = gr.HTML()
            heard = gr.Dropdown(list(NAMES), multiselect=True, label="Languages heard in this recording",
                                info="Filled in from the parts; add any other language you heard.")
            allowed = gr.Checkbox(label="I have permission to keep this recording (the speakers know it is saved "
                                        "and why).")
            save_btn = gr.Button("Save correction", variant="primary")
            saved = gr.HTML()

    gr.HTML(f'<div class="sa-foot">Model {html.escape(MODEL_ID)} · Whisper large-v3-turbo + LoRA, trained largely on '
            "Swivuriso (CC BY 4.0, DSFSI, University of Pretoria). See the Space page for all credits.</div>")

    go.click(busy, None, [status, go], queue=False, show_progress="hidden").then(
        transcribe, [uploaded, recorded, lang, second, beams, extra],
        [summary_box, transcript, download, seg_box, timeline, parts, heard, state, translation, saved],
        show_progress="hidden",
    ).then(idle, None, [status, go], queue=False)
    tr_btn.click(lambda: "Translating…", None, translation, queue=False).then(
        translate, state, translation, show_progress="hidden")
    # the word picker follows the corrected parts: editing a part's text or language resets it
    parts.change(refresh_words, parts, [words, word_pick, word_view], queue=False)
    cs.change(lambda on: gr.Group(visible=on), cs, cs_box, queue=False)
    set_btn.click(set_word_lang, [words, word_pick, word_lang, heard], [words, word_pick, word_view, heard],
                  queue=False)
    save_btn.click(save, [state, parts, words, heard, allowed], saved)


if __name__ == "__main__":
    # Server-side rendering (Gradio's Node proxy) can leave a private Space's page unstyled; plain serving works.
    demo.launch(ssr_mode=False, theme=THEME, css=CSS, head=HEAD, footer_links=[])
