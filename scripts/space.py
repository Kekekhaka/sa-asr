"""Deploy the Gradio app (app/gradio_app.py) to a private Hugging Face Space on ZeroGPU (free).

    1. Create the Space once on huggingface.co/new-space: SDK Gradio (Blank), hardware ZeroGPU, Private.
       (Free accounts can only pick ZeroGPU there; the API cannot.)
    2. $env:HF_TOKEN = "hf_..."                                # a write token
       python scripts/space.py push --repo <user>/sa-asr-app

Uploads only what the app needs (app/, src/, configs/, the spelling model) plus a generated Space page with the
licences and credits, requirements.txt and packages.txt (ffmpeg, for m4a/webm/ogg uploads). Files on the Space
that are no longer part of the app are removed. Run it again after every app change.

The Space must be private, and your token is stored as its HF_TOKEN secret: the app needs it to load the
private model and to save corrections. Corrections go to the private dataset <user>/sa-asr-feedback (Space
variable SA_ASR_FEEDBACK_REPO), because a Space's own disk is wiped on every restart; fetch them with
`python scripts/feedback.py pull --repo <user>/sa-asr-feedback`.

ZeroGPU lends a GPU for each transcription (up to GPU_SECONDS in gradio_app.py), counted against a daily quota.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import sa_asr  # noqa: F401  (applies the corporate-certificate fix before any HF call)
from sa_asr.config import PROJECT_ROOT

APP_DIRS = ["app", "src", "configs"]
APP_FILES = ["models/text_lid.json.gz"]
SKIP_PARTS = {"__pycache__"}

# torch is left unpinned: ZeroGPU supports only some CUDA builds, and the newest one from PyPI is the safe choice.
REQUIREMENTS = """\
torch
transformers==5.20.2
peft==0.21.2
accelerate==1.15.0
datasets==5.1.0
huggingface_hub==2.2.0
jiwer==4.0.0
soundfile==0.14.0
soxr==1.1.0
pyyaml==6.0.3
pandas==3.0.6
tabulate==0.10.0
sentencepiece==0.2.2
truststore==0.10.5
psutil==7.2.2
"""

PACKAGES = "ffmpeg\n"

PAGE = """\
---
title: South African Speech-to-Text
emoji: 🎙️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: {gradio_version}
python_version: "3.12"
app_file: app/gradio_app.py
pinned: false
models:
  - {model}
  - openai/whisper-large-v3-turbo
  - dsfsi-anv/za-anv-multilingual-whisper-v3-turbo
  - facebook/nllb-200-distilled-600M
  - Helsinki-NLP/opus-mt-ve-en
datasets:
  - dsfsi-anv/za-african-next-voices
  - dsfsi-anv/multilingual-nchlt-dataset
  - danielshaps/nchlt_speech_eng
  - dsfsi/multilingual-lwazi-dataset
  - google/fleurs
---

# South African Speech-to-Text

Speech-to-text for the 11 official South African languages (English, Afrikaans, isiZulu, isiXhosa, Sepedi, Sesotho,
Setswana, Xitsonga, Tshivenda, siSwati, isiNdebele), with language detection, mixed-language recordings and
translation to English. Private research project, running on ZeroGPU.

**Recordings:** corrections you save (audio and text) are stored in a private dataset to improve the model. Only
save recordings when everyone speaking knows it is kept and why.

## Credits and licences

**Swivuriso (African Next Voices)**: the speech model was trained largely on this dataset, created by the
Data Science for Social Impact group (DSFSI), University of Pretoria, as part of the African Next Voices project,
[`dsfsi-anv/za-african-next-voices`](https://huggingface.co/datasets/dsfsi-anv/za-african-next-voices),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: used as training data for
fine-tuning; the recordings themselves are not redistributed here. The creators prohibit using the voices for
speech synthesis or voice cloning; this app does neither.

**Second model**: the optional "Second model for clear speech" is
[`dsfsi-anv/za-anv-multilingual-whisper-v3-turbo`](https://huggingface.co/dsfsi-anv/za-anv-multilingual-whisper-v3-turbo)
by DSFSI, University of Pretoria, used unchanged under the MIT licence.

**Other training data**: NCHLT speech corpus (CC BY 4.0), NCHLT South African English (CC BY 3.0), FLEURS
(CC BY 4.0), OpenSLR 32 (CC BY-SA 4.0) and the Lwazi telephone corpus (licence not stated on its dataset card).

**Models**: base model [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo)
(MIT), fine-tuned with LoRA as `{model}`. Translation:
[`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) (CC BY-NC 4.0,
non-commercial use only) and [`Helsinki-NLP/opus-mt-ve-en`](https://huggingface.co/Helsinki-NLP/opus-mt-ve-en)
(Apache 2.0).
"""


def app_files() -> dict[str, Path]:
    files = {}
    for d in APP_DIRS:
        for p in sorted((PROJECT_ROOT / d).rglob("*")):
            if p.is_file() and not SKIP_PARTS & set(p.parts) and not any(x.endswith(".egg-info") for x in p.parts):
                files[p.relative_to(PROJECT_ROOT).as_posix()] = p
    for f in APP_FILES:
        p = PROJECT_ROOT / f
        if not p.exists():
            raise SystemExit(f"Missing {f}: the app needs it (scripts/train_text_lid.py makes it)")
        files[f] = p
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push"])
    ap.add_argument("--repo", required=True, help="the Space, e.g. <user>/sa-asr-app")
    ap.add_argument("--model", default="KekeCode/sa-asr-whisper-lora-v3", help="the app's default model")
    ap.add_argument("--feedback-repo", help="private dataset for corrections (default <user>/sa-asr-feedback)")
    args = ap.parse_args()
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN to a write token first")
    feedback_repo = args.feedback_repo or f"{args.repo.split('/')[0]}/sa-asr-feedback"
    api = HfApi()
    if not api.repo_exists(args.repo, repo_type="space"):
        raise SystemExit(f"Create the Space first on https://huggingface.co/new-space: name {args.repo.split('/')[1]}, "
                         "SDK Gradio (Blank), hardware ZeroGPU, Private. Then run this again.")
    if not api.space_info(args.repo).private:
        raise SystemExit(f"{args.repo} is public: make it private in its Settings first (it saves recordings)")
    api.add_space_secret(args.repo, "HF_TOKEN", token)
    api.add_space_variable(args.repo, "SA_ASR_MODEL", args.model)
    api.add_space_variable(args.repo, "SA_ASR_FEEDBACK_REPO", feedback_repo)

    local = app_files()
    import gradio

    generated = {"README.md": PAGE.format(model=args.model, gradio_version=gradio.__version__), "requirements.txt": REQUIREMENTS,
                 "packages.txt": PACKAGES}
    ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)) for r, p in local.items()]
    ops += [CommitOperationAdd(path_in_repo=r, path_or_fileobj=t.encode()) for r, t in generated.items()]
    keep = set(local) | set(generated) | {".gitattributes"}
    remote = set(api.list_repo_files(args.repo, repo_type="space"))
    ops += [CommitOperationDelete(path_in_repo=r) for r in sorted(remote - keep)]
    api.create_commit(args.repo, repo_type="space", operations=ops, commit_message="update app")
    print(f"Pushed {len(local) + len(generated)} files to https://huggingface.co/spaces/{args.repo} (private).\n"
          f"Corrections will go to https://huggingface.co/datasets/{feedback_repo} (private). The first start "
          "installs packages and downloads the models, so it takes several minutes.")


if __name__ == "__main__":
    main()
