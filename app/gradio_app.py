"""Speech-to-text for South African languages, Gradio version (for a Hugging Face Space on ZeroGPU).

    python app/gradio_app.py                      # locally (CPU, or GPU if there is one)
    python scripts/space.py push --repo <user>/sa-asr-app

Same engine as app/streamlit_app.py (which stays the local app): language detection part by part, the optional
second model for clear speech, translation to English, and corrections saved with the language of every part.
On ZeroGPU the models are loaded once at start-up and each transcription borrows a GPU for up to GPU_SECONDS.
Saved corrections are copied to the private dataset SA_ASR_FEEDBACK_REPO, because a Space's disk is wiped on restart.
"""

from __future__ import annotations

import os
import string
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

registry = load_languages()
model_ref, trained_langs = resolve_model(MODEL_ID)
LANGS = trained_langs or list(registry)
NAMES = {registry[c].name: c for c in LANGS}
# typed languages: the full name, its first word ("English" for "English (South African)") or the code, any case
LOOKUP = {**{n.split(" (")[0].lower(): c for n, c in NAMES.items()}, **{n.lower(): c for n, c in NAMES.items()},
          **{c: c for c in LANGS}}

# Loaded once at start-up (on ZeroGPU, "cuda" is available here and the weights move to the GPU on each call).
MODEL = load_transcriber(model_ref, trained_langs is not None)
TEXT_LID = TextLID.load() if TEXT_LID_PATH.exists() else None
SECOND = load_second_model() if WITH_SECOND else None


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


@gpu(duration=GPU_SECONDS)
def run_model(wav: np.ndarray, lang_code: str | None, use_second: bool, beams: int) -> dict:
    MODEL.num_beams = beams
    return transcribe_audio(MODEL, wav, registry, lang_code, mixed=lang_code is None, text_lid=TEXT_LID,
                            second_model=SECOND if use_second else None)


def transcribe(audio_path, lang_name, use_second, beams):
    if not audio_path:
        raise gr.Error("Upload or record some audio first.")
    wav = load_audio(Path(audio_path).read_bytes(), Path(audio_path).name)
    seconds = duration_seconds(wav)
    if seconds < 0.3:
        raise gr.Error("The audio is too short to transcribe.")
    if seconds > MAX_MINUTES * 60:
        raise gr.Error(f"Recordings up to {MAX_MINUTES} minutes only (this one is {fmt_time(seconds)}).")
    result = run_model(wav, None if lang_name == "Detect automatically" else NAMES[lang_name], use_second, int(beams))

    shares = result.get("languages") or {}
    if len(shares) > 1:
        summary_md = "**Languages:** " + " · ".join(f"{registry[c].name} {s:.0%}" for c, s in shares.items())
    elif result["detected"]:
        top = list(result["detected"].items())[:3]
        summary_md = f"**Language:** {registry[result['language']].name} (" + " · ".join(
            f"{registry[c].name} {p:.0%}" for c, p in top) + ")"
    else:
        summary_md = f"**Language:** {registry[result['language']].name}" if result["language"] else ""
    summary_md += f"  \n{fmt_time(result['duration'])} of audio in {result['elapsed']:.1f} s"
    if result.get("rerouted"):
        summary_md += f"; {result['rerouted']} clear part(s) transcribed by the second model ({DSFSI_MODEL}, MIT)"

    segs = [s for s in result["segments"] if s["text"]] or [
        {"start": 0.0, "end": result["duration"], "language": result["language"], "text": ""}]
    rows = [[f"{fmt_time(s['start'])}–{fmt_time(s['end'])}",
             registry[s["language"]].name if s.get("language") in LANGS else registry[LANGS[0]].name,
             s["text"]] for s in segs]
    txt = Path(tempfile.mkdtemp()) / "transcript.txt"
    txt.write_text(result["text"], encoding="utf-8")
    state = {"wav": wav, "segments": segs, "language": result["language"], "text": result["text"]}
    heard = sorted({r[1] for r in rows})
    return (summary_md, result["text"] or "(no speech recognised)", str(txt),
            gr.Dataframe(value=rows, row_count=(len(rows), "fixed")), gr.Dropdown(value=heard), state, "", "")


def translate(state):
    if not state or not state["text"]:
        raise gr.Error("Transcribe something first.")
    if not any(can_translate(s.get("language")) or s.get("language") == "eng" for s in state["segments"]):
        raise gr.Error("No translation model covers this language.")
    return " ".join(p for p in translate_segments(state["segments"]) if p)


def _word(w: str) -> str:
    return w.strip(string.punctuation + "“”‘’").lower()


def save(state, table, word_tags, heard, allowed):
    if not state:
        raise gr.Error("Transcribe something first.")
    if not allowed:
        raise gr.Error("Tick the permission box first.")
    rows = table.values.tolist() if hasattr(table, "values") else table
    parts = []
    for seg, (_, lang_name, text) in zip(state["segments"], rows):
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

    # every word takes its part's language, then the "word = language" tags override it
    words = [(w, p["lang"]) for p in parts for w in p["text"].split()]
    word_langs = [lang for _, lang in words]
    for tag in filter(None, (t.strip() for t in (word_tags or "").replace("\n", ",").split(","))):
        if "=" not in tag:
            raise gr.Error(f"Write word tags as word=Language, e.g. thank=English (got '{tag}')")
        w, lang_name = (x.strip() for x in tag.split("=", 1))
        code = LOOKUP.get(lang_name.lower())
        if code is None:
            raise gr.Error(f"Unknown language '{lang_name}' in '{tag}'. Use one of: {', '.join(NAMES)}")
        hits = [i for i, (x, _) in enumerate(words) if _word(x) == _word(w)]
        if not hits:
            raise gr.Error(f"The word '{w}' is not in the corrected text")
        for i in hits:
            word_langs[i] = code

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
    return (f"Saved ({langs}). {stats['count']} correction(s) so far in this session"
            + (f"; copied to {FEEDBACK_REPO}." if FEEDBACK_REPO else "."))


with gr.Blocks(title="South African Speech-to-Text") as demo:
    gr.Markdown("# 🎙️ South African Speech-to-Text\nEnglish · Afrikaans · isiZulu · isiXhosa · Sepedi · Sesotho · "
                "Setswana · Xitsonga · Tshivenda · siSwati · isiNdebele")
    state = gr.State(None)
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio (upload or record)")
            lang = gr.Dropdown(["Detect automatically"] + list(NAMES), value="Detect automatically", label="Language",
                               info="Detection works part by part (4-10 s), so mixed recordings get a language "
                                    "per part.")
            with gr.Accordion("Options", open=False):
                second = gr.Checkbox(
                    value=False, visible=SECOND is not None, label="Second model for clear speech",
                    info=f"isiZulu, isiXhosa, Sesotho, Setswana, Xitsonga, Tshivenda: parts of at least "
                         f"{MIN_SECONDS:.0f} s that are not telephone quality are transcribed again by {DSFSI_MODEL}.")
                beams = gr.Slider(1, 5, value=1, step=1, label="Beam size", info="Higher is a little more accurate.")
            go = gr.Button("Transcribe", variant="primary")
        with gr.Column():
            info = gr.Markdown()
            transcript = gr.Textbox(label="Transcript", lines=8)
            transcript_file = gr.File(label="Download transcript")
            tr_btn = gr.Button("Translate to English")
            translation = gr.Textbox(label="English translation", lines=5)

    with gr.Accordion("Wrong language or words? Correct it", open=False):
        gr.Markdown("Fix each part: its language and its text. Saved corrections (audio + text) are kept privately "
                    "to improve the model. **Only save recordings when everyone speaking knows it is kept and why.**")
        parts = gr.Dataframe(headers=["Time", "Language", "Text"], datatype=["str", "str", "str"],
                             col_count=(3, "fixed"), interactive=True, wrap=True, label="Parts",
                             column_widths=["12%", "18%", "70%"])
        gr.Markdown("Languages: " + ", ".join(NAMES))
        tags = gr.Textbox(label="Words in another language (optional)", placeholder="thank=English, you=English",
                          info="For code-switching inside a part: word=Language, separated by commas.")
        heard = gr.Dropdown(list(NAMES), multiselect=True, label="Languages heard in this recording")
        allowed = gr.Checkbox(label="I have permission to keep this recording (the speakers know it is saved and why).")
        save_btn = gr.Button("Save correction", variant="primary")
        saved = gr.Markdown()

    go.click(transcribe, [audio, lang, second, beams],
             [info, transcript, transcript_file, parts, heard, state, translation, saved])
    tr_btn.click(translate, state, translation)
    save_btn.click(save, [state, parts, tags, heard, allowed], saved)
    gr.Markdown(f"<small>Model: {MODEL_ID} (Whisper large-v3-turbo + LoRA, trained largely on Swivuriso, "
                "CC BY 4.0, DSFSI, University of Pretoria). See the Space page for all credits.</small>")


if __name__ == "__main__":
    demo.launch()
