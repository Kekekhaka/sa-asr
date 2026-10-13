import json

import numpy as np
import pytest
import torch

from sa_asr.training import lid_weighted_loss


def test_lid_loss_is_not_diluted_by_transcript_length():
    torch.manual_seed(0)
    vocab = 7
    short = torch.tensor([[3, 1, 2, -100, -100, -100]])
    long = torch.tensor([[3, 1, 2, 4, 5, 6]])
    logits = torch.randn(1, 6, vocab)
    lid_ce = torch.nn.functional.cross_entropy(logits[:, 0], torch.tensor([3]))
    for labels in (short, long):
        text_ce = torch.nn.functional.cross_entropy(logits[0, 1:], labels[0, 1:], ignore_index=-100)
        got = lid_weighted_loss(logits, labels, 2.0)
        assert torch.allclose(got, text_ce + 2.0 * lid_ce)  # the LID part is the same for both lengths


def _tiny_processor():
    from transformers import WhisperProcessor

    try:
        return WhisperProcessor.from_pretrained("openai/whisper-tiny", local_files_only=True)
    except Exception:
        pytest.skip("openai/whisper-tiny processor not cached locally")


def _tiny_model(processor):
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    cfg = WhisperConfig.from_pretrained("openai/whisper-tiny", local_files_only=True)
    cfg.encoder_layers = cfg.decoder_layers = 1
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(cfg)
    from transformers import GenerationConfig

    model.generation_config = GenerationConfig.from_pretrained("openai/whisper-tiny", local_files_only=True)
    return model


def test_init_adapter_loads_trainable_and_rejects_other_language_tokens(tmp_path):
    from peft import LoraConfig, get_peft_model

    from sa_asr.backends import META_FILE
    from sa_asr.config import load_languages
    from sa_asr.training import load_init_adapter

    proc = _tiny_processor()
    reg = load_languages()
    adapter = tmp_path / "adapter"
    get_peft_model(_tiny_model(proc), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])).save_pretrained(adapter)
    meta = {"languages": ["nso", "sot"], "whisper_tokens": {"nso": reg["nso"].whisper_token, "sot": reg["sot"].whisper_token}}
    (adapter / META_FILE).write_text(json.dumps(meta), encoding="utf-8")

    model = load_init_adapter(_tiny_model(proc), str(adapter), [reg["nso"], reg["sot"]])
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0

    meta["whisper_tokens"]["nso"] = "xx"
    (adapter / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="other language tokens"):
        load_init_adapter(_tiny_model(proc), str(adapter), [reg["nso"], reg["sot"]])


def test_trainer_with_lid_loss_trains_and_reports_lid_accuracy(tmp_path):
    import io

    import soundfile as sf
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import Seq2SeqTrainingArguments

    from sa_asr.config import load_languages
    from sa_asr.training import LabelBuilder, WhisperCollator, WhisperWERTrainer

    proc = _tiny_processor()
    reg = load_languages()
    lb = LabelBuilder(proc, [reg["nso"], reg["sot"]])

    def tone(code):
        t = np.arange(16000) / 16000
        return (0.3 * np.sin(2 * np.pi * (300 if code == "nso" else 1200) * t)).astype(np.float32)

    rows = [{"input_features": proc.feature_extractor(tone(c), sampling_rate=16000).input_features[0],
             "labels": lb("ke a leboga", c)} for c in ["nso", "sot"] * 4]
    flac = {}
    for c in ["nso", "sot"]:
        buf = io.BytesIO()
        sf.write(buf, tone(c), 16000, format="FLAC")
        flac[c] = buf.getvalue()
    wer_sets = {("nchlt", reg[c]): Dataset.from_dict({"flac": [flac[c]] * 2, "text": ["ke a leboga"] * 2})
                for c in ["nso", "sot"]}

    model = get_peft_model(_tiny_model(proc), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    args = Seq2SeqTrainingArguments(output_dir=str(tmp_path), max_steps=2, per_device_train_batch_size=4,
                                    report_to="none", remove_unused_columns=False, label_names=["labels"],
                                    save_strategy="no", use_cpu=True)
    trainer = WhisperWERTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
                                data_collator=WhisperCollator(), wer_sets=wer_sets, wer_processor=proc,
                                wer_batch_size=2, lid_token_ids=lb.lang_ids, lid_loss_weight=1.0)
    assert trainer.train().training_loss > 0
    metrics = trainer._wer_metrics("eval")
    assert {"eval_lid_acc", "eval_lid_family_acc", "eval_lid_nso", "eval_lid_sot"} <= set(metrics)
    assert metrics["eval_lid_family_acc"] == 1.0  # nso and sot are both Sotho-Tswana


def test_init_adapter_with_trainable_tokens_keeps_the_old_lora_weights(tmp_path):
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

    from sa_asr.backends import META_FILE
    from sa_asr.config import load_languages
    from sa_asr.training import load_init_adapter

    proc = _tiny_processor()
    reg = load_languages()
    old = get_peft_model(_tiny_model(proc), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():
        for n, p in old.named_parameters():
            if "lora_B" in n:
                p.normal_()  # non-zero, so copying is visible
    adapter = tmp_path / "adapter"
    old.save_pretrained(adapter)
    (adapter / META_FILE).write_text(json.dumps({"languages": ["nso"], "whisper_tokens": {"nso": reg["nso"].whisper_token}}))

    ids = [proc.tokenizer.convert_tokens_to_ids(f"<|{reg[c].whisper_token}|>") for c in ["nso", "sot"]]
    new = load_init_adapter(_tiny_model(proc), str(adapter), [reg["nso"]], trainable_token_ids=ids)
    trainable = [n for n, p in new.named_parameters() if p.requires_grad]
    assert any("trainable_tokens" in n for n in trainable)
    old_sd, new_sd = get_peft_model_state_dict(old), get_peft_model_state_dict(new)
    for k, v in old_sd.items():
        assert torch.equal(v, new_sd[k]), k


def test_trainable_tokens_separate_languages_whose_tokens_start_identical(tmp_path):
    """Whisper's proxies for nso and sot are near-identical vectors; with the token rows
    trainable (and the LID loss), the model learns to tell the two languages apart."""
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import Seq2SeqTrainingArguments

    from sa_asr.config import load_languages
    from sa_asr.training import LabelBuilder, WhisperCollator, WhisperWERTrainer

    proc = _tiny_processor()
    reg = load_languages()
    lb = LabelBuilder(proc, [reg["nso"], reg["sot"]])
    nso, sot = lb.lang_ids["nso"], lb.lang_ids["sot"]
    from transformers import WhisperForConditionalGeneration

    try:  # pretrained (not random) weights: like the real case, only the language decision is new
        base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", local_files_only=True)
    except Exception:
        pytest.skip("openai/whisper-tiny weights not cached locally")
    with torch.no_grad():  # make the two language tokens identical, like in Whisper
        base.model.decoder.embed_tokens.weight[sot] = base.model.decoder.embed_tokens.weight[nso]

    def tone(code, i):
        t = np.arange(16000) / 16000
        return (0.3 * np.sin(2 * np.pi * ((300 if code == "nso" else 1200) + 7 * i) * t)).astype(np.float32)

    rows = [{"input_features": proc.feature_extractor(tone(c, i), sampling_rate=16000).input_features[0],
             "labels": lb("ke a leboga", c)} for i in range(8) for c in ["nso", "sot"]]
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "fc1", "fc2"],
                                            trainable_token_indices={"embed_tokens": [nso, sot]}))
    args = Seq2SeqTrainingArguments(output_dir=str(tmp_path), max_steps=40, per_device_train_batch_size=8,
                                    learning_rate=1e-3, report_to="none", remove_unused_columns=False,
                                    label_names=["labels"], save_strategy="no", use_cpu=True, seed=0)
    WhisperWERTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows), data_collator=WhisperCollator(),
                      lid_token_ids=lb.lang_ids, lid_loss_weight=1.0).train()
    model.eval()
    sot_id = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    for code, want in [("nso", nso), ("sot", sot)]:
        f = torch.tensor(proc.feature_extractor(tone(code, 3.5), sampling_rate=16000).input_features)
        with torch.no_grad():
            logits = model(input_features=f, decoder_input_ids=torch.tensor([[sot_id]])).logits[0, -1]
        other = sot if want == nso else nso
        assert logits[want] > logits[other], code
