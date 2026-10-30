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
from sa_asr.routing import DSFSI_MODEL, MIN_SECONDS, load_second_model
from sa_asr.textlid import DEFAULT_PATH as TEXT_LID_PATH, TextLID
from sa_asr.translate import can_translate, translate_segments

try:  # on a ZeroGPU Space; a no-op locally
    import spaces

    gpu = spaces.GPU
except ImportError:
    def gpu(duration=None):
        return lambda fn: fn

MODEL_ID = os.environ.get("SA_ASR_MODEL", "KekeCode/sa-asr-whisper-lora-v3")
FEEDBACK_REPO = os.environ.get("SA_ASR_FEEDBACK_REPO")
# The second model costs ~3 GB and start-up time; set SA_ASR_SECOND=0 to leave it out.
WITH_SECOND = os.environ.get("SA_ASR_SECOND", "1") != "0"
GPU_SECONDS = 120          # the most one transcription may hold the GPU (counts against the daily ZeroGPU quota)
MAX_MINUTES = 10           # longer recordings are refused, so a run fits in GPU_SECONDS
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


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def short(code: str | None) -> str:
    return registry[code].name.split(" (")[0] if code in registry else "Unknown"


def chip(code: str | None, text: str | None = None) -> str:
    return (f'<span class="sa-chip" style="--c:{COLORS.get(code or "", "#94A3B8")}"><i></i>'
            f'{html.escape(text or short(code))}</span>')


@gpu(duration=GPU_SECONDS)
def run_model(wav: np.ndarray, lang_code: str | None, use_second: bool, beams: int) -> dict:
    MODEL.num_beams = beams
    return transcribe_audio(MODEL, wav, registry, lang_code, mixed=lang_code is None, text_lid=TEXT_LID,
                            second_model=SECOND if use_second else None)


def summary_html(result: dict, shares: dict[str, float], chosen: bool) -> str:
    """Stat tiles, then a bar of the languages: share of speaking time if mixed, else detection confidence."""
    mixed = len(shares) > 1
    if mixed:
        lang_value, title, parts = f"{len(shares)} languages", "Language mix · share of speaking time", list(
            shares.items())
    else:
        lang_value = short(result["language"]) if result["language"] else "Chosen by Whisper"
        title = "You chose the language" if chosen else "Detection confidence"
        parts = [] if chosen else list((result["detected"] or {}).items())[:3]
    stats = [("Language", lang_value), ("Length", fmt_time(result["duration"])),
             ("Took", f"{result['elapsed']:.1f} s")]
    out = '<div class="sa-card sa-summary"><div class="sa-stats">' + "".join(
        f'<div class="sa-stat"><span>{k}</span><b>{html.escape(v)}</b></div>' for k, v in stats) + "</div>"
    if parts:
        bar = "".join(f'<i style="width:{100 * p:.1f}%;background:{COLORS.get(c, "#94A3B8")}"></i>'
                      for c, p in parts)
        out += (f'<div class="sa-label">{title}</div><div class="sa-bar">{bar}</div><div class="sa-legend">'
                + "".join(chip(c, f"{short(c)} {p:.0%}") for c, p in parts) + "</div>")
    if result.get("rerouted"):
        out += (f'<div class="sa-note">{result["rerouted"]} clear part(s) were transcribed by the second model '
                f"({html.escape(DSFSI_MODEL)}, MIT licence).</div>")
    return out + "</div>"


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


def transcribe(uploaded, recorded, lang_name, use_second, beams):
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
    result = run_model(wav, NAMES[lang_name] if chosen else None, use_second, int(beams))

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
    return (gr.Column(visible=True), summary_html(result, shares, chosen),
            result["text"] or "(no speech recognised)", gr.DownloadButton(value=str(txt), visible=True),
            gr.Accordion(label=f"Timeline · {len(segs)} parts", open=mixed, visible=len(segs) > 1),
            timeline_html(segs, mixed), gr.Dataframe(value=rows, row_count=(len(rows), "fixed")),
            gr.update(value=heard), state, gr.Textbox(value="", visible=False), "")


def translate(state):
    if not state or not state["text"]:
        raise gr.Error("Transcribe something first.")
    if not any(can_translate(s.get("language")) or s.get("language") == "eng" for s in state["segments"]):
        raise gr.Error("No translation model covers this language.")
    return gr.Textbox(value=" ".join(p for p in translate_segments(state["segments"]) if p), visible=True)


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
DONE_JS = "() => { document.getElementById('sa-results')?.scrollIntoView({behavior: 'smooth', block: 'start'}); }"


def busy():
    return BUSY, gr.Button("Transcribing…", interactive=False), gr.Column(visible=False)


def idle():
    return "", gr.Button("Transcribe", interactive=True)


def step(n: int, title: str, hint: str = "") -> gr.HTML:
    return gr.HTML(f'<div class="sa-step"><span>{n}</span><div><b>{title}</b>'
                   + (f"<small>{hint}</small>" if hint else "") + "</div></div>")


HERO = (
    '<div class="sa-hero"><div class="sa-flag"></div>'
    '<div class="sa-badge">🇿🇦 11 official languages · code-switching aware</div>'
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
.sa-legend { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.sa-note { margin-top: 12px; font-size: .82rem; color: var(--body-text-color-subdued); }
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
    status = gr.HTML()

    with gr.Column(visible=False, elem_id="sa-results") as results:
        step(3, "Result")
        summary_box = gr.HTML()
        transcript = gr.Textbox(label="Transcript", lines=5, buttons=["copy"], elem_id="sa-transcript")
        with gr.Row(equal_height=True):
            download = gr.DownloadButton("⬇ Download .txt", visible=False, variant="secondary", size="md")
            tr_btn = gr.Button("🌍 Translate to English", variant="secondary", size="md")
        translation = gr.Textbox(label="English translation", lines=3, buttons=["copy"], visible=False)
        with gr.Accordion("Timeline", open=False, visible=False) as seg_box:
            timeline = gr.HTML()

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

    go.click(busy, None, [status, go, results], queue=False, show_progress="hidden").then(
        transcribe, [uploaded, recorded, lang, second, beams],
        [results, summary_box, transcript, download, seg_box, timeline, parts, heard, state, translation, saved],
        show_progress="hidden",
    ).then(idle, None, [status, go], js=DONE_JS, queue=False)
    tr_btn.click(lambda: gr.Textbox(value="Translating…", visible=True), None, translation, queue=False).then(
        translate, state, translation)
    # the word picker follows the corrected parts: editing a part's text or language resets it
    parts.change(refresh_words, parts, [words, word_pick, word_view], queue=False)
    cs.change(lambda on: gr.Group(visible=on), cs, cs_box, queue=False)
    set_btn.click(set_word_lang, [words, word_pick, word_lang, heard], [words, word_pick, word_view, heard],
                  queue=False)
    save_btn.click(save, [state, parts, words, heard, allowed], saved)


if __name__ == "__main__":
    # Server-side rendering (Gradio's Node proxy) can leave a private Space's page unstyled; plain serving works.
    demo.launch(ssr_mode=False, theme=THEME, css=CSS, footer_links=[])
