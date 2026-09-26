# South African multilingual speech-to-text

Fine-tuning Whisper for the 11 official spoken languages of South Africa: English, Afrikaans,
isiZulu, isiXhosa, Sepedi, Sesotho, Setswana, Xitsonga, Tshivenda, siSwati and isiNdebele.
The target is all kinds of audio: phone calls, app and voice-note recordings, and meetings.

The workflow is **measure first, then train**. `scripts/evaluate.py` scores existing models on
held-out test sets. `scripts/train.py` fine-tunes Whisper (LoRA or full). The same evaluation
then shows whether the fine-tuned model actually beats the baseline. Deployment comes later.

## Layout

```
configs/
  languages.yaml        language registry: Whisper proxy tokens, MMS adapters, corpus configs
  eval_baseline.yaml    which models x which test sets to score
  eval_smoke.yaml       tiny CPU check of the eval pipeline
  train_lora.yaml       main fine-tuning config (A100)
  train_smoke.yaml      tiny CPU check of the training pipeline
src/sa_asr/
  data.py               corpus specs, streaming loaders, cached eval sets
  text.py               transcript cleanup (training) and normalization (scoring)
  backends.py           Whisper / MMS transcribers behind one interface
  training.py           labels, training stream mixing, collator, WER-reporting trainer
  inference.py          any-length transcription + language detection (used by the app)
  metrics.py, audio.py, config.py
scripts/
  evaluate.py           score models -> results/<name>/summary.md
  train.py              fine-tune -> outputs/<name>/final (--hub-repo: also push to HF)
  hub_sync.py           push/pull a results folder to a private HF repo
  package.py            build sa-asr.zip, or push the code to a private HF repo
  run_stage.py          one pipeline stage with GPU-appropriate settings (what the notebooks call)
  diagnose_memory.py    RAM profile of the training data pipeline, without a model
notebooks/
  colab_sa_asr.ipynb    runs all of the above on a Colab GPU (project on Google Drive)
  colab_no_drive.ipynb  same, where Google Drive is blocked: state is kept in private HF repos
  kaggle_sa_asr.ipynb   Kaggle version: pick a STAGE, "Save & Run All" runs it in the background
app/
  streamlit_app.py      web UI: upload an audio file or record a voice message
tests/
```

## Data

| Corpus | HF repo | Languages | Notes |
|---|---|---|---|
| NCHLT | `dsfsi-anv/multilingual-nchlt-dataset` | 10 (no eng) | read prompts, ~56 h/language, CC-BY-4.0 |
| Swivuriso (African Next Voices) | `dsfsi-anv/za-african-next-voices` | zul xho sot tsn tso ven nbl | scripted + spontaneous, 48 kHz, CC-BY-4.0, **gated**: accept the terms on the dataset page |
| NCHLT English | `danielshaps/nchlt_speech_eng` | eng | South African English, CC-BY-3.0; no validation split, so 5% of speakers are held out |
| Lwazi | `dsfsi/multilingual-lwazi-dataset` | 9 (no afr, eng) | **real telephone speech**; the phone-call benchmark. Licence not stated on the card: check before commercial use |
| FLEURS | `google/fleurs` | afr zul xho nso | read Wikipedia; **not used for training**, so it stays an unseen benchmark |

Everything is streamed. Nothing is downloaded in full. Evaluation sets are sampled once and cached
as 16 kHz FLAC under `data/eval_cache/`, so every model is scored on identical audio.

## Audio types

| Audio | How it is covered |
|---|---|
| Phone calls (8 kHz) | Lwazi (real phone audio) in training and eval. Also, 30% of wideband training audio is passed through a simulated phone line (8 kHz band limit + G.711 mu-law), set in `augment:` in the training config |
| App / voice notes | Swivuriso (varied phones and microphones, spontaneous speech), NCHLT; light noise augmentation |
| Meetings | Same acoustic model. Long recordings, speaker turns and overlapping speech are an inference-pipeline problem (VAD, chunking, diarization), part of the deployment phase |

Not covered yet: **code-switching** (e.g. English mixed into isiZulu mid-sentence). No open corpus
targets it. Swivuriso's spontaneous speech contains some. A small in-house test set of real
code-switched calls would be the best next data investment.

## Language tokens

Whisper has no language token for any South African language except English and Afrikaans. Each other
language borrows a distinct, unrelated Whisper token (isiZulu → `<|sn|>`, isiXhosa → `<|sw|>`, …;
see `configs/languages.yaml`). Fine-tuning teaches the model what each proxy means, so the
fine-tuned model can both transcribe and identify the language.

Do not change the mapping after training a model. The training run saves it in
`final/sa_asr_meta.json`, and evaluation refuses to run if the two disagree.

## Scoring

WER and CER are corpus-level (total errors / total reference words), computed after lowercasing
and removing punctuation and annotation tags such as `[noise]`. Diacritics are **kept**,
because Tshivenda and the Sotho-Tswana languages use them to tell words apart.

## Running

Training needs a GPU. The notebook `notebooks/colab_sa_asr.ipynb` covers the whole flow on Colab.

```bash
pip install -e ".[dev]"
hf auth login                                   # for Swivuriso

python -m pytest -q
python scripts/evaluate.py --config configs/eval_baseline.yaml
python scripts/train.py   --config configs/train_lora.yaml
```

### Kaggle (recommended on the free tier)

`notebooks/kaggle_sa_asr.ipynb`. About 30 free GPU hours a week, runs of up to 12 h that continue in
the background with the browser closed. Set `STAGE` in cell 1 (`baseline`, `trial`, `train`,
`compare`, `report`), then **Save Version → Save & Run All**. The HF token comes from Kaggle Secrets
(`HF_TOKEN`). The code comes from an attached Kaggle dataset made from `sa-asr.zip`, or, if none is
attached, from the private repo filled by `python scripts/package.py push --repo <you>/sa-asr-code`.
Results and checkpoints are synced to the same private HF repos as the Colab notebook.

After changing code: `python scripts/package.py zip`, then upload a **new version** of the Kaggle
dataset (or rerun `package.py push`).

### Colab without Google Drive

Use `notebooks/colab_no_drive.ipynb`. Upload `sa-asr.zip` (the project, minus data and outputs) when
the notebook asks. Anything that must outlive the Colab session goes to private repos under your
Hugging Face account. That covers baseline results (`<you>/sa-asr-results`), training checkpoints and
the final model (`<you>/sa-asr-whisper-lora`). After a disconnect, rerun the setup cells and the cell
you were on; finished work is pulled back and skipped. The Hugging Face token must have **Write**
permission.

Build the zip with `python scripts/package.py zip`.

Useful overrides: `--langs zul xho`, `--max-samples 50` (eval), `--max-steps 500` (train),
`--resume` (continue training after a disconnect).

## Web app

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

Upload a file (wav, mp3, flac, ogg; m4a/webm/mp4 need ffmpeg installed) or record from the
microphone. To use a model stored on Hugging Face, e.g. `<you>/sa-asr-whisper-lora`, enter the repo id in the
sidebar. For a private repo, log in once on the laptop first:
`python -c "import sa_asr; from huggingface_hub import login; login()"`. This goes through `sa_asr`
so the corporate-certificate fix applies; the plain `hf auth login` command fails on that network.

The app picks up `outputs/whisper-turbo-sa-lora/final` automatically once you have
trained it. Until then it falls back to zero-shot Whisper; any model folder or Hugging Face id can be
entered in the sidebar. With a fine-tuned model, "Detect automatically" picks among the model's own
languages and shows its confidence. Audio longer than 30 s is split at pauses and shown as timed
segments. Large models are slow on CPU, so for real use run the app on a GPU machine with the
trained `final/` folder.

On networks that re-sign HTTPS (e.g. corporate proxies), Python cannot verify the Hugging Face
certificate. The package uses `truststore` so Python trusts the Windows certificate store, as
curl and browsers do.

For local work on this Windows machine, keep the virtualenv **outside OneDrive** (e.g.
`C:\Users\<you>\.venvs\sa-asr`). A venv inside a synced folder means tens of thousands of files
syncing.

## Licences

NCHLT, Swivuriso and FLEURS are CC-BY (attribution required). Lwazi's licence is not stated on the
mirror, so check it before shipping a model trained on it. Whisper and the dsfsi-anv Whisper fine-tunes are
MIT. MMS (`facebook/mms-1b-all`) is **CC-BY-NC-4.0**, fine for benchmarking but not for a commercial
product. The same goes for `dsfsi/mms-1b-all-nchlt`, which is why it is not in the baseline.
