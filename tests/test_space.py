import importlib.util
import json
import shutil

import huggingface_hub
import numpy as np

from sa_asr.config import PROJECT_ROOT
from sa_asr.feedback import load_corrections, pull, save_correction


def clip(seed: int) -> np.ndarray:
    return (0.1 * np.random.RandomState(seed).randn(16000)).astype(np.float32)


def test_pull_adds_uploaded_corrections_once(tmp_path, monkeypatch):
    space, hub, local = tmp_path / "space", tmp_path / "hub", tmp_path / "local"
    recs = [save_correction(clip(i), "nso", f"ke a leboga {i}", predicted_lang="sot", predicted_text="x", model="m",
                            root=space) for i in range(2)]
    (hub / "raw").mkdir(parents=True)
    for r in recs:  # the layout upload_correction writes
        (hub / "raw" / f"{r['utt_id']}.json").write_text(json.dumps(r), encoding="utf-8")
        shutil.copyfile(space / "audio" / f"{r['utt_id']}.flac", hub / "raw" / f"{r['utt_id']}.flac")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(hub))
    assert pull("u/sa-asr-feedback", local) == 2
    assert pull("u/sa-asr-feedback", local) == 0  # nothing twice
    assert sorted(r["text"] for r in load_corrections(local)) == ["ke a leboga 0", "ke a leboga 1"]
    assert len(list((local / "audio").glob("*.flac"))) == 2


def test_space_upload_has_the_app_and_no_caches():
    spec = importlib.util.spec_from_file_location("space", PROJECT_ROOT / "scripts" / "space.py")
    space = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(space)
    files = space.app_files()
    assert {"app/streamlit_app.py", "src/sa_asr/inference.py", "configs/languages.yaml",
            "models/text_lid.json.gz"} <= set(files)
    assert not any("__pycache__" in f or f.startswith(("data/", "outputs/", "tests/")) for f in files)
    page = space.PAGE.format(model="u/m", gradio_version="5.0.0")
    assert "CC BY 4.0" in page and "MIT" in page and "app_file: app/gradio_app.py" in page
    assert "sdk: gradio" in page and "torch==" not in space.REQUIREMENTS
