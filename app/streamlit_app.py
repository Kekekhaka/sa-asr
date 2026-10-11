"""Speech-to-text for South African languages: upload a file or record a voice message.

    streamlit run app/streamlit_app.py

Uses a model trained with scripts/train.py: outputs/.../final when it exists locally, else
the trained model on the Hub (SA_ASR_MODEL, default KekeCode/sa-asr-whisper-lora). Any local
folder or Hugging Face id (e.g. openai/whisper-large-v3-turbo) can be entered in the sidebar.
"""

from __future__ import annotations

import os

import streamlit as st
import torch

from sa_asr.audio import duration_seconds
from sa_asr.config import PROJECT_ROOT, load_languages
from sa_asr.feedback import FEEDBACK_DIR, save_correction, summary
from sa_asr.inference import load_audio, load_transcriber, resolve_model, transcribe_audio
from sa_asr.translate import ROUTES, can_translate, translate_segments

UPLOAD_TYPES = ["wav", "mp3", "flac", "ogg", "opus", "m4a", "aac", "webm", "mp4"]
DEFAULT_FINETUNED = PROJECT_ROOT / "outputs" / "whisper-turbo-sa-lora" / "final"
# The trained model on the Hub (train.py --hub-repo); set SA_ASR_MODEL to use another one.
DEFAULT_HUB_MODEL = os.environ.get("SA_ASR_MODEL", "KekeCode/sa-asr-whisper-lora")

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
    default_model = str(DEFAULT_FINETUNED.relative_to(PROJECT_ROOT)) if DEFAULT_FINETUNED.exists() else DEFAULT_HUB_MODEL
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

AUTO_LABELS = {"auto": "Detect automatically (one language)", "mixed": "Detect automatically (mixed languages)"}
options = ["auto"] + (["mixed"] if finetuned else []) + [c for c in registry if not finetuned or c in trained_langs]
lang_choice = st.selectbox(
    "Language",
    options,
    format_func=lambda c: AUTO_LABELS.get(c) or registry[c].name,
    help="Choosing the language usually gives the best results. 'Mixed languages' detects the language "
    "part by part (4-10 s at a time), for recordings that switch language, like the national anthem.",
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
        result = transcribe_audio(model, wav, registry, None if lang_choice in AUTO_LABELS else lang_choice,
                                  mixed=lang_choice == "mixed")
    # The audio and the model's first answer are kept for corrections (what it got wrong).
    st.session_state.update(
        result=result, wav=wav, model_used=model_input.strip(), rid=st.session_state.get("rid", 0) + 1,
        first_prediction=(result["language"], result["text"]),
    )

result = st.session_state.get("result")
if result:
    st.divider()
    shares = result.get("languages") or {}
    mixed_result = len(shares) > 1
    if mixed_result:
        lang_label = f"Mixed ({len(shares)})"
    else:
        lang_label = registry[result["language"]].name if result["language"] else "chosen by Whisper"
    c1, c2, c3 = st.columns(3)
    c1.metric("Language", lang_label)
    c2.metric("Audio", fmt_time(result["duration"]))
    c3.metric("Processing time", f"{result['elapsed']:.1f} s")
    if mixed_result:
        st.caption("Languages: " + " · ".join(f"{registry[c].name} {s:.0%}" for c, s in shares.items()))
    elif result["detected"]:
        top = list(result["detected"].items())[:3]
        st.caption("Detection confidence: " + " · ".join(f"{registry[c].name} {p:.0%}" for c, p in top))

    st.subheader("Transcript")
    st.text_area("Transcript", result["text"] or "(no speech recognised)", height=200, label_visibility="collapsed")
    st.download_button("Download .txt", result["text"].encode("utf-8"), file_name="transcript.txt", mime="text/plain")

    if len(result["segments"]) > 1:
        with st.expander(f"Segments ({len(result['segments'])})", expanded=mixed_result):
            for seg in result["segments"]:
                tag = f" **{registry[seg['language']].name}**" if mixed_result and seg.get("language") else ""
                st.markdown(f"`{fmt_time(seg['start'])}–{fmt_time(seg['end'])}`{tag} {seg['text']}")

    # --- English translation -------------------------------------------------------------
    rid = st.session_state["rid"]
    lang = result["language"]
    seg_langs = {seg.get("language") for seg in result["segments"] if seg["text"]}
    if result["text"] and any(can_translate(c) for c in seg_langs):
        st.subheader("English translation")
        for c in seg_langs:
            if can_translate(c) and ROUTES[c].approximate:
                st.caption(f"No translation model covers {registry[c].name}; it is translated as isiZulu, so it is approximate.")
        translation = st.session_state.get("translation")
        if translation and translation[0] == rid:
            st.text_area("Translation", translation[1], height=150, label_visibility="collapsed")
            st.download_button("Download translation .txt", translation[1].encode("utf-8"),
                               file_name="translation.txt", mime="text/plain")
        elif st.button("Translate to English"):
            with st.spinner("Translating (the first time loads the translation model)…"):
                try:
                    parts = translate_segments(result["segments"])
                except Exception as e:
                    st.error(f"Couldn't translate: {e}")
                    st.stop()
            st.session_state["translation"] = (rid, " ".join(p for p in parts if p))
            st.rerun()

    # --- corrections -------------------------------------------------------------------
    with st.expander("Wrong language or words? Correct it"):
        st.caption(
            "Pick the right language and fix the text. Saved corrections (audio + text) are kept in "
            "`data/feedback/` and can be used to fine-tune the model further. Only save recordings you "
            "have permission to use."
        )
        fix_options = trained_langs if finetuned else list(registry)
        fix_lang = st.selectbox(
            "Correct language", fix_options, index=fix_options.index(lang) if lang in fix_options else 0,
            format_func=lambda c: registry[c].name, key=f"fix_lang_{rid}",
        )
        if fix_lang != lang and st.button(f"Transcribe again as {registry[fix_lang].name}"):
            with st.spinner(f"Transcribing as {registry[fix_lang].name}…"):
                model = get_model(model_ref, finetuned)
                model.num_beams = num_beams
                redo = transcribe_audio(model, st.session_state["wav"], registry, fix_lang)
            st.session_state.update(result=redo, rid=rid + 1)  # keeps first_prediction
            st.rerun()
        fix_text = st.text_area("Correct transcript", result["text"], height=150, key=f"fix_text_{rid}")
        if st.button("Save correction", type="primary", disabled=not fix_text.strip()):
            first_lang, first_text = st.session_state["first_prediction"]
            save_correction(st.session_state["wav"], fix_lang, fix_text, predicted_lang=first_lang,
                            predicted_text=first_text, model=st.session_state["model_used"])
            stats = summary()
            st.success(f"Saved. {stats['count']} corrections so far ({stats['hours'] * 60:.0f} min of audio) "
                       f"in {FEEDBACK_DIR.relative_to(PROJECT_ROOT)}.")
