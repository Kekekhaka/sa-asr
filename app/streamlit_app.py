"""Speech-to-text for South African languages: upload a file or record a voice message.

    streamlit run app/streamlit_app.py

Uses a model trained with scripts/train.py: outputs/.../final when it exists locally, else
the trained model on the Hub (SA_ASR_MODEL, default KekeCode/sa-asr-whisper-lora-v2). Any local
folder or Hugging Face id (e.g. openai/whisper-large-v3-turbo) can be entered in the sidebar.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# On a Hugging Face Space (scripts/space.py) the package is not installed: use src/ next to the app.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import streamlit as st
import torch

from sa_asr.audio import duration_seconds
from sa_asr.config import PROJECT_ROOT, load_languages
from sa_asr.feedback import FEEDBACK_DIR, save_correction, summary, upload_correction
from sa_asr.inference import load_audio, load_transcriber, resolve_model, transcribe_audio
from sa_asr.routing import DSFSI_MODEL, MIN_SECONDS, load_second_model
from sa_asr.textlid import DEFAULT_PATH as TEXT_LID_PATH, TextLID
from sa_asr.translate import ROUTES, can_translate, translate_segments

UPLOAD_TYPES = ["wav", "mp3", "flac", "ogg", "opus", "m4a", "aac", "webm", "mp4"]
DEFAULT_FINETUNED = PROJECT_ROOT / "outputs" / "whisper-turbo-sa-lora" / "final"
# The trained model on the Hub (train.py --hub-repo); set SA_ASR_MODEL to use another one.
DEFAULT_HUB_MODEL = os.environ.get("SA_ASR_MODEL", "KekeCode/sa-asr-whisper-lora-v5")
# Set on the Space (scripts/space.py): saved corrections are also copied to this private dataset.
FEEDBACK_REPO = os.environ.get("SA_ASR_FEEDBACK_REPO")

st.set_page_config(page_title="South African Speech-to-Text", page_icon="🎙️", layout="centered")
registry = load_languages()


@st.cache_data(show_spinner="Looking up model…")
def cached_resolve(model: str):
    return resolve_model(model)


@st.cache_resource(show_spinner=False)
def get_text_lid() -> TextLID | None:
    """The spelling-based second opinion on the language (scripts/train_text_lid.py); optional."""
    return TextLID.load() if TEXT_LID_PATH.exists() else None


@st.cache_resource(show_spinner=False)
def get_model(model: str, finetuned: bool):
    return load_transcriber(model, finetuned)


@st.cache_resource(show_spinner=False)
def get_second_model():
    return load_second_model()


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
    use_second = st.checkbox(
        "Second model for clear speech",
        value=False,
        help="For isiZulu, isiXhosa, Sesotho, Setswana, Xitsonga and Tshivenda: parts of a recording that are at "
        f"least {MIN_SECONDS:.0f} s long and not telephone quality are transcribed again by "
        f"{DSFSI_MODEL} (MIT licence, trained on Swivuriso), which was about 5 points more accurate on such "
        "speech in our tests. Needs about 3 GB more memory, is slower, and downloads the model the first time.",
    )

# --- main ------------------------------------------------------------------------------

st.title("🎙️ South African Speech-to-Text")
st.caption(
    "English · Afrikaans · isiZulu · isiXhosa · Sepedi · Sesotho · Setswana · "
    "Xitsonga · Tshivenda · siSwati · isiNdebele"
)

AUTO_LABELS = {"auto": "Detect automatically"}
options = ["auto"] + [c for c in registry if not finetuned or c in trained_langs]
lang_choice = st.selectbox(
    "Language",
    options,
    format_func=lambda c: AUTO_LABELS.get(c) or registry[c].name,
    help="Choosing the language usually gives the best results. 'Detect automatically' works for one "
    "language or several: it detects the language part by part (4-10 s at a time) and stays on one "
    "language unless the evidence for a change is strong, so a recording that switches language, "
    "like the national anthem, gets a language per part.",
)

tab_upload, tab_record = st.tabs(["Upload a file", "Record"])
with tab_upload:
    uploaded = st.file_uploader("Audio file", type=UPLOAD_TYPES, help="Phone calls, voice notes, meetings.")
    if uploaded:
        st.audio(uploaded)
with tab_record:
    recorded = st.audio_input("Record a voice message")
    if recorded:
        st.download_button("Download recording", recorded.getvalue(), mime="audio/wav",
                           file_name=f"recording-{time.strftime('%Y%m%d-%H%M%S')}.wav")

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
    # Streamlit can only interrupt a run when the script touches the page, so progress is reported
    # between batches of windows. Pressing Stop (or Transcribe again) then cancels within a few seconds.
    st.button("Stop", key="stop_transcribing", help="Cancel this transcription")
    bar = st.progress(0.0, text=f"Transcribing {fmt_time(duration_seconds(wav))} of audio…")

    def show_progress(what: str, done: int, total: int) -> None:
        bar.progress(min(done / max(total, 1), 1.0), text=f"{what}… ({done}/{total} parts done)")

    second = None
    if use_second:
        with st.spinner("Loading the second model (the first time this downloads about 3 GB)…"):
            try:
                second = get_second_model()
            except Exception as e:
                st.warning(f"Couldn't load the second model, using the main one only: {e}")
    result = transcribe_audio(model, wav, registry, None if lang_choice in AUTO_LABELS else lang_choice,
                              mixed=lang_choice == "auto", text_lid=get_text_lid(), on_progress=show_progress,
                              second_model=second)
    bar.empty()
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
    if result.get("rerouted"):
        st.caption(f"{result['rerouted']} clear part(s) were transcribed by the second model "
                   f"({DSFSI_MODEL}, MIT licence).")
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
                engine = " _(second model)_" if seg.get("engine") else ""
                st.markdown(f"`{fmt_time(seg['start'])}–{fmt_time(seg['end'])}`{tag}{engine} {seg['text']}")

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
    with st.expander("Wrong language or words? Correct it", expanded=False):
        st.caption(
            "Fix the language and the text. Saved corrections (audio + text) are kept in "
            "`data/feedback/` and can be used to fine-tune the model further. Only save recordings you "
            "have permission to use."
        )
        fix_options = trained_langs if finetuned else list(registry)
        names = {registry[c].name: c for c in fix_options}
        segs = [s for s in result["segments"] if s["text"]]
        fix_segments = None
        if len(segs) > 1:
            # One row per part: its own language and text, so a mixed recording is corrected part by part.
            st.markdown("**Correct each part** (language and text)")
            table = st.data_editor(
                pd.DataFrame({
                    "Time": [f"{fmt_time(s['start'])}–{fmt_time(s['end'])}" for s in segs],
                    "Language": [registry[s["language"]].name if s.get("language") in fix_options
                                 else registry[fix_options[0]].name for s in segs],
                    "Text": [s["text"] for s in segs],
                }),
                column_config={
                    "Time": st.column_config.TextColumn(disabled=True, width="small"),
                    "Language": st.column_config.SelectboxColumn(options=list(names), required=True, width="small"),
                    "Text": st.column_config.TextColumn(width="large"),
                },
                hide_index=True, width="stretch", key=f"segs_{rid}",
            )
            fix_segments = [
                {"start": s["start"], "end": s["end"], "lang": names[lg], "text": str(tx).strip()}
                for s, lg, tx in zip(segs, table["Language"].tolist(), table["Text"].tolist())
            ]
            fix_segments = [s for s in fix_segments if s["text"]]
            fix_text = " ".join(s["text"] for s in fix_segments)
            seconds: dict[str, float] = {}
            for s in fix_segments:
                seconds[s["lang"]] = seconds.get(s["lang"], 0.0) + s["end"] - s["start"]
            fix_lang = max(seconds, key=seconds.get) if seconds else lang
            default_word_langs = [s["lang"] for s in fix_segments for _ in s["text"].split()]
            st.caption(f"Main language (most time): {registry[fix_lang].name}")
        else:
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
            default_word_langs = [fix_lang] * len(fix_text.split())
        word_langs = default_word_langs if fix_segments else None
        words = fix_text.split()
        if words and st.checkbox("Some words are in another language (code-switching)", key=f"cs_{rid}"):
            st.caption("Set the language of each word that differs (e.g. English words inside a Sepedi sentence).")
            table = st.data_editor(
                pd.DataFrame({"Word": words, "Language": [registry[c].name for c in default_word_langs]}),
                column_config={
                    "Word": st.column_config.TextColumn(disabled=True),
                    "Language": st.column_config.SelectboxColumn(options=list(names), required=True),
                },
                hide_index=True, width="stretch",
                # a new key when the words or their default languages change, so the table matches them
                key=f"words_{rid}_{hash((fix_text, tuple(default_word_langs)))}",
            )
            word_langs = [names[n] for n in table["Language"].tolist()]
        heard = sorted(set(word_langs or [fix_lang]) | {fix_lang})
        present = st.multiselect(
            "Languages heard in this recording", fix_options, default=heard,
            format_func=lambda c: registry[c].name, key=f"present_{rid}_{'-'.join(heard)}",
            help="Filled in from the parts and words above; add any other language you heard. Used to test "
            "language detection.",
        )
        allowed = st.checkbox(
            "I have permission to keep this recording (the speakers know it is saved and why).",
            key=f"allowed_{rid}",
        )
        if st.button("Save correction", type="primary", disabled=not (fix_text.strip() and allowed)):
            first_lang, first_text = st.session_state["first_prediction"]
            rec = save_correction(st.session_state["wav"], fix_lang, fix_text, predicted_lang=first_lang,
                                  predicted_text=first_text, model=st.session_state["model_used"],
                                  languages_present=present, word_langs=word_langs, segments=fix_segments)
            # On a Space the local disk is wiped on restart, so each correction is also copied to a private dataset.
            if FEEDBACK_REPO:
                try:
                    upload_correction(rec, FEEDBACK_REPO)
                except Exception as e:
                    st.error(f"Saved here, but couldn't copy it to {FEEDBACK_REPO} (it will be lost on restart): {e}")
                    st.stop()
            stats = summary()
            st.success(f"Saved. {stats['count']} corrections so far ({stats['hours'] * 60:.0f} min of audio) "
                       f"in {FEEDBACK_DIR.relative_to(PROJECT_ROOT)}. Run `python scripts/feedback.py report` "
                       "to see how the model did on them.")
