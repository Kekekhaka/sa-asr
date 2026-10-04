"""Fine-tune Whisper on South African languages (LoRA or full fine-tuning).

    python scripts/train.py --config configs/train_lora.yaml
    python scripts/train.py --config configs/train_lora.yaml --max-steps 20 --langs zul   # smoke test

Writes to output_dir:
    checkpoint-*/     Trainer checkpoints (resume with --resume)
    final/            the best checkpoint by validation WER: adapter (or full model),
                      processor and sa_asr_meta.json (the language-token mapping)

Without persistent disk (e.g. Colab without Google Drive), add --hub-repo <user>/<name>:
every checkpoint is also pushed to that private Hugging Face repo, --resume in a fresh
session pulls the latest one back, and final/ is uploaded to the repo root, so the repo id
can be used directly as a model in evaluate.py and the app.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from datasets import concatenate_datasets
from transformers import Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging as transformers_logging

from sa_asr.backends import META_FILE, native_bf16
from sa_asr.config import PROJECT_ROOT, load_languages, load_yaml, quiet_progress_bars, select_languages
from sa_asr.data import build_eval_set, check_access
from sa_asr.training import (
    EvalFeatureDataset,
    LabelBuilder,
    WhisperCollator,
    PeriodicSaveCallback,
    StallWatchdog,
    TimeLimitCallback,
    WhisperWERTrainer,
    build_train_stream,
    find_resume_checkpoint,
    make_example_fn,
    start_memory_monitor,
)

log = logging.getLogger("train")


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--langs", nargs="*", help="Override the config's languages")
    ap.add_argument("--max-steps", type=int, help="Override training.max_steps")
    ap.add_argument("--output-dir", help="Override output_dir")
    ap.add_argument("--batch-size", type=int, help="Override per-device train/eval batch size")
    ap.add_argument("--grad-accum", type=int, help="Override gradient_accumulation_steps")
    ap.add_argument("--resume", action="store_true", help="Resume from the last checkpoint (local, else --hub-repo)")
    ap.add_argument("--hub-repo", help="Private HF repo <user>/<name> to push checkpoints and the final model to")
    ap.add_argument("--max-hours", type=float,
                    help="Save and stop after this many hours (default: training.max_runtime_hours, e.g. 11 on Kaggle)")
    return ap.parse_args()


def main():
    started = time.time()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    transformers_logging.set_verbosity_error()  # per-batch generation warnings flood the output
    quiet_progress_bars()
    start_memory_monitor()
    args = parse_args()
    cfg = load_yaml(args.config)
    tcfg = dict(cfg["training"])
    if args.max_steps:
        tcfg["max_steps"] = args.max_steps
        # Short runs: evaluate and save at the halfway point too, not only at the very end.
        half = max(1, args.max_steps // 2)
        tcfg["eval_steps"] = min(tcfg["eval_steps"], half)
        tcfg["save_steps"] = min(tcfg["save_steps"], half)
    if args.batch_size:
        tcfg["per_device_train_batch_size"] = tcfg["per_device_eval_batch_size"] = args.batch_size
    if args.grad_accum:
        tcfg["gradient_accumulation_steps"] = args.grad_accum
    out_dir = resolve(args.output_dir or cfg["output_dir"])
    registry = load_languages()
    languages = select_languages(registry, args.langs or cfg.get("languages"))
    seed = tcfg.get("seed", 42)

    resume_from = find_resume_checkpoint(out_dir, args.hub_repo) if args.resume else None
    data_seed = seed
    if args.resume:
        if resume_from is None:
            log.warning("--resume: no checkpoint found locally%s; starting from scratch",
                        f" or in {args.hub_repo}" if args.hub_repo else "")
        else:
            # Re-streaming and skipping every batch already seen would take hours, so the
            # stream restarts with a new shuffle seed instead (ignore_data_skip below).
            step = json.loads((resume_from / "trainer_state.json").read_text(encoding="utf-8"))["global_step"]
            data_seed = seed + step
            log.info("Resuming from %s (step %d)", resume_from, step)

    # --- data ------------------------------------------------------------------------
    corpus_weights = {c: w for c, w in cfg["corpora"].items() if w and w > 0}
    usable = {c for c in corpus_weights if check_access(c)}
    missing = set(corpus_weights) - usable
    if missing:
        log.warning("Training WITHOUT %s (no access). Fix access and restart to include it.", sorted(missing))
    lang_weights = cfg.get("language_weights") or {}
    streams = [
        (corpus, lang, corpus_weights[corpus] * lang_weights.get(lang.code, 1.0))
        for lang in languages
        for corpus in corpus_weights
        if corpus in usable and corpus in lang.corpora
    ]
    # language_weights: {xxx: 0} switches a language off entirely (not trained, not validated).
    streams = [s for s in streams if s[2] > 0]
    if not streams:
        raise SystemExit("No training data: check languages, corpora weights and dataset access.")
    trained_langs = sorted({lang.code for _, lang, _ in streams})
    stream_desc = [(c, l.code, round(w, 3)) for c, l, w in streams]
    log.info("Training streams (corpus, lang, weight): %s", stream_desc)
    skipped_langs = [l.code for l in languages if l.code not in trained_langs]
    if skipped_langs:
        log.warning("No training data for %s", skipped_langs)

    processor = WhisperProcessor.from_pretrained(cfg["base_model"])
    labels = LabelBuilder(processor, [registry[c] for c in trained_langs], text_mode=cfg.get("text", "clean"))
    max_dur = cfg.get("max_duration_s", 30.0)
    max_label_tokens = cfg.get("max_label_tokens", 440)
    aug = cfg.get("augment") or {}
    example_fn = make_example_fn(
        processor, labels, max_dur, max_label_tokens,
        telephone_prob=aug.get("telephone_prob", 0.0), noise_prob=aug.get("noise_prob", 0.0), seed=seed,
    )
    train_ds = build_train_stream(
        streams, example_fn, seed=data_seed, shuffle_buffer=cfg.get("shuffle_buffer", 64), max_duration_s=max_dur
    )

    vcfg = cfg.get("validation", {})
    cache_dir = resolve(vcfg.get("cache_dir", "data/eval_cache"))
    wer_sets = {}
    for corpus, lang, _ in streams:
        wer_sets[(corpus, lang)] = build_eval_set(
            corpus, lang, "validation", cache_dir,
            max_samples=vcfg.get("max_samples_per_set", 32), seed=seed, max_duration_s=max_dur,
        )
    eval_ds = EvalFeatureDataset(concatenate_datasets(list(wer_sets.values())), processor, labels, max_label_tokens)

    # --- model -----------------------------------------------------------------------
    model = WhisperForConditionalGeneration.from_pretrained(cfg["base_model"])
    model.config.use_cache = False
    # Old checkpoints ship forced_decoder_ids that would override the language we pass.
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    lora = cfg.get("lora", {})
    if lora.get("enabled", True):
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=lora.get("r", 32),
                lora_alpha=lora.get("alpha", 64),
                lora_dropout=lora.get("dropout", 0.05),
                target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]),
                bias="none",
            ),
        )
        model.print_trainable_parameters()

    # --- trainer ---------------------------------------------------------------------
    cuda = torch.cuda.is_available()
    bf16 = native_bf16()  # fp16 on a T4: its bf16 is emulated and several times slower
    if cuda:
        log.info("Mixed precision: %s on %s", "bf16" if bf16 else "fp16", torch.cuda.get_device_name())
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        max_steps=tcfg["max_steps"],
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=tcfg.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 1),
        learning_rate=float(tcfg["learning_rate"]),
        warmup_steps=tcfg.get("warmup_steps", 0),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "linear"),
        weight_decay=tcfg.get("weight_decay", 0.0),
        bf16=bf16,
        fp16=cuda and not bf16,
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=tcfg["eval_steps"],
        save_strategy="steps",
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg.get("save_total_limit", 3),
        logging_steps=tcfg.get("logging_steps", 25),
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=False,
        prediction_loss_only=True,  # WER comes from WhisperWERTrainer; don't hold eval logits
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=tcfg.get("dataloader_num_workers", 2),
        report_to=tcfg.get("report_to", "none"),
        seed=seed,
        ignore_data_skip=True,
        push_to_hub=bool(args.hub_repo),
        hub_model_id=args.hub_repo,
        hub_private_repo=True,
        hub_strategy="checkpoint",  # pushes last-checkpoint/ at every save
    )
    trainer = WhisperWERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=WhisperCollator(),
        processing_class=processor.feature_extractor,
        wer_sets=wer_sets,
        wer_processor=processor,
        wer_batch_size=tcfg.get("per_device_eval_batch_size", 16),
    )
    max_hours = args.max_hours or tcfg.get("max_runtime_hours")
    time_limit = TimeLimitCallback(started + max_hours * 3600) if max_hours else None
    if time_limit:
        trainer.add_callback(time_limit)
        log.info("Will save and stop after %.1f h if not finished", max_hours)
    every_h = tcfg.get("checkpoint_every_hours")
    if every_h:
        trainer.add_callback(PeriodicSaveCallback(every_h * 3600))
        log.info("Will also save a checkpoint at least every %.1f h", every_h)
    stall_min = tcfg.get("stall_timeout_minutes")
    if stall_min:
        trainer.add_callback(StallWatchdog(stall_min * 60))
        log.info("Watchdog: will exit if training makes no progress for %d min", stall_min)
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    if time_limit and time_limit.hit:
        ckpt = find_resume_checkpoint(out_dir, None)
        if args.hub_repo and ckpt is not None:
            # Upload synchronously: the Trainer's own push runs in the background and may not
            # finish before the platform stops the machine.
            from huggingface_hub import HfApi

            HfApi().upload_folder(repo_id=args.hub_repo, folder_path=str(ckpt), path_in_repo="last-checkpoint",
                                  commit_message=f"checkpoint at step {trainer.state.global_step} (runtime limit)")
            log.info("Uploaded %s to %s/last-checkpoint", ckpt.name, args.hub_repo)
        log.warning("Stopped at the runtime limit (step %d of %d). Run the same command again with --resume "
                    "to continue; the final model is not written yet.", trainer.state.global_step, tcfg["max_steps"])
        return

    # --- save ------------------------------------------------------------------------
    final = out_dir / "final"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    meta = {
        "base_model": cfg["base_model"],
        "method": "lora" if lora.get("enabled", True) else "full",
        "languages": trained_langs,
        "whisper_tokens": {c: registry[c].whisper_token for c in trained_langs},
        "text": cfg.get("text", "clean"),
        "streams": stream_desc,
        "augment": aug,
        "best_metric_wer": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }
    (final / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Saved %s (best validation WER %s)", final, trainer.state.best_metric)
    if args.hub_repo:
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            repo_id=args.hub_repo, folder_path=str(final), commit_message="final model (best validation WER)"
        )
        log.info("Uploaded final model to https://huggingface.co/%s", args.hub_repo)


if __name__ == "__main__":
    main()
