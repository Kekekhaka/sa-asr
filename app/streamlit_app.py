"""Speech-to-text for South African languages: upload a file or record a voice message.

    streamlit run app/streamlit_app.py

Uses a model trained with scripts/train.py (outputs/.../final) when one exists, otherwise
zero-shot Whisper. Any local model folder or Hugging Face id can be entered in the sidebar.
"""

from __future__ import annotations

import streamlit as st
import torch

from sa_asr.audio import duration_seconds
from sa_asr.config import PROJECT_ROOT, load_languages
from sa_asr.inference import load_audio, load_transcriber, resolve_model, transcribe_audio

UPLOAD_TYPES = ["wav", "mp3", "flac", "ogg", "opus", "m4a", "aac", "webm", "mp4"]
DEFAULT_FINETUNED = PROJECT_ROOT / "outputs" / "whisper-turbo-sa-lora" / "final"
DEFAULT_ZERO_SHOT = "openai/whisper-large-v3-turbo"

st.set_page_config(page_title="South African Speech-to-Text", page_icon="🎙️", layout="centered")
registry = load_languages()


@st.cache_data(show_spinner="Looking up model…")
def cached_resolve(model: str):
    return resolve_model(model)


@st.cache_resource(show_spinner=False)
def get_model(model: str, finetuned: bool):
    return load_transcriber(model, finetuned)


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


# --- sidebar ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Model")
    default_model = str(DEFAULT_FINETUNED.relative_to(PROJECT_ROOT)) if DEFAULT_FINETUNED.exists() else DEFAULT_ZERO_SHOT
    model_input = st.text_input("Local folder or Hugging Face id", value=default_model)
    model_ref, trained_langs = cached_resolve(model_input.strip())
    finetuned = trained_langs is not None
    if finetuned:
        st.success(f"Fine-tuned South African model ({len(trained_langs)} languages)")
    else:
        st.warning(
            "Zero-shot model. Whisper has no language token for most South African languages, "
            "so expect poor results outside English and Afrikaans."
        )
    num_beams = st.slider("Beam size", 1, 5, 1, help="Higher is slightly more accurate and slower.")
    st.caption(f"Running on **{'GPU' if torch.cuda.is_available() else 'CPU'}**. On CPU, large models are slow.")

# --- main ------------------------------------------------------------------------------

st.title("🎙️ South African Speech-to-Text")
st.caption(
    "English · Afrikaans · isiZulu · isiXhosa · Sepedi · Sesotho · Setswana · "
    "Xitsonga · Tshivenda · siSwati · isiNdebele"
)

options = ["auto"] + [c for c in registry if not finetuned or c in trained_langs]
lang_choice = st.selectbox(
    "Language",
    options,
    format_func=lambda c: "Detect automatically" if c == "auto" else registry[c].name,
    help="Choosing the language usually gives better results than automatic detection.",
)

tab_upload, tab_record = st.tabs(["Upload a file", "Record"])
with tab_upload:
    uploaded = st.file_uploader("Audio file", type=UPLOAD_TYPES, help="Phone calls, voice notes, meetings.")
    if uploaded:
        st.audio(uploaded)
with tab_record:
    recorded = st.audio_input("Record a voice message")

source = recorded or uploaded
if source is not None:
    st.caption(f"Using: **{'your recording' if source is recorded else source.name}**")

if st.button("Transcribe", type="primary", disabled=source is None, width="stretch"):
    try:
        wav = load_audio(source.getvalue(), getattr(source, "name", "recording.wav"))
    except Exception as e:
        st.error(f"Couldn't read the audio: {e}")
        st.stop()
    if duration_seconds(wav) < 0.3:
        st.error("The audio is too short to transcribe.")
        st.stop()
    with st.spinner("Loading model…"):
        try:
            model = get_model(model_ref, finetuned)
        except Exception as e:
            st.error(f"Couldn't load model '{model_input}': {e}")
            st.stop()
    model.num_beams = num_beams
    with st.spinner(f"Transcribing {fmt_time(duration_seconds(wav))} of audio…"):
        st.session_state["result"] = transcribe_audio(
            model, wav, registry, None if lang_choice == "auto" else lang_choice
        )

result = st.session_state.get("result")
if result:
    st.divider()
    lang_name = registry[result["language"]].name if result["language"] else "chosen by Whisper"
    c1, c2, c3 = st.columns(3)
    c1.metric("Language", lang_name)
    c2.metric("Audio", fmt_time(result["duration"]))
    c3.metric("Processing time", f"{result['elapsed']:.1f} s")
    if result["detected"]:
        top = list(result["detected"].items())[:3]
        st.caption("Detection confidence: " + " · ".join(f"{registry[c].name} {p:.0%}" for c, p in top))

    st.subheader("Transcript")
    st.text_area("Transcript", result["text"] or "(no speech recognised)", height=200, label_visibility="collapsed")
    st.download_button("Download .txt", result["text"].encode("utf-8"), file_name="transcript.txt", mime="text/plain")

    if len(result["segments"]) > 1:
        with st.expander(f"Segments ({len(result['segments'])})"):
            for seg in result["segments"]:
                st.markdown(f"`{fmt_time(seg['start'])}–{fmt_time(seg['end'])}` {seg['text']}")
