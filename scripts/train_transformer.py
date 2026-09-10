"""Phase 4: GPU smoke test, full DistilBERT fine-tuning, then one test evaluation."""

import argparse
import gc
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import torch
from transformers import Trainer, TrainingArguments, TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_classical import load_saved_training_partition, write_json
from imdb_sentiment.data.loading import load_imdb_dataset
from imdb_sentiment.models.distilbert import (
    TransformerConfig, hardware_info, load_tokenizer, load_model, tokenize_split, collator, gpu_smoke_test)
from imdb_sentiment.evaluation.metrics import evaluate_predictions

METRICS = ROOT / "reports/metrics"
ARTIFACTS = ROOT / "artifacts/transformer"


class HistoryCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        write_json(METRICS / "transformer_training_history.json", state.log_history)


def compute_metrics(prediction):
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    metrics = evaluate_predictions(prediction.label_ids, np.argmax(logits, axis=-1))
    return {key: metrics[key] for key in ("accuracy", "precision", "recall", "f1")}


def training_arguments(config, hardware, output_dir):
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay,
        num_train_epochs=config.epochs, max_grad_norm=config.max_grad_norm,
        fp16=hardware["fp16"], gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        seed=config.seed, data_seed=config.seed,
        logging_steps=25, logging_first_step=True, report_to="none",
        dataloader_num_workers=0, dataloader_pin_memory=True,
        eval_accumulation_steps=16, optim="adamw_torch", disable_tqdm=True,
    )


def train_attempt(config, hardware, tokenizer, train_tokens, validation_tokens, checkpoint_dir):
    """No test dataset is accepted here; selection can only use validation."""
    set_seed(config.seed)
    trainer = Trainer(
        model=load_model(), args=training_arguments(config, hardware, checkpoint_dir),
        train_dataset=train_tokens, eval_dataset=validation_tokens,
        processing_class=tokenizer, data_collator=collator(tokenizer),
        compute_metrics=compute_metrics, callbacks=[HistoryCallback()],
    )
    resume = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    start = perf_counter()
    trainer.train(resume_from_checkpoint=resume)
    seconds = perf_counter() - start
    trainer.save_model(str(ARTIFACTS / "best_model"))
    tokenizer.save_pretrained(str(ARTIFACTS / "best_model"))
    return trainer, seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    METRICS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if (METRICS / "transformer_test_metrics.json").exists():
        raise SystemExit("Final Transformer results already exist; refusing repeated test evaluation.")
    hardware = hardware_info()
    print("HARDWARE:", json.dumps(hardware, indent=2), flush=True)
    write_json(METRICS / "transformer_hardware.json", hardware)
    if not hardware["cuda_available"]:
        raise SystemExit("CUDA is unavailable. Install compatible CUDA PyTorch; full training was not started on CPU.")
    config = TransformerConfig()
    set_seed(config.seed)
    metadata_path = METRICS / "split_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset = load_imdb_dataset()
    train, validation = load_saved_training_partition(dataset, metadata)
    assert (len(train), len(validation), len(dataset["test"])) == (20000, 5000, 25000)
    tokenizer = load_tokenizer()

    # 256 balances the original training median (174 whitespace words) and 4 GB VRAM.
    # WordPiece tokens differ from whitespace words; long reviews will be truncated.
    oom_changes = []
    try:
        smoke = gpu_smoke_test(tokenizer, list(train.select(range(8))["text"]), config, hardware["fp16"])
    except torch.cuda.OutOfMemoryError:
        smoke = None
    if smoke is None:
        gc.collect()
        torch.cuda.empty_cache()
        config.train_batch_size, config.eval_batch_size, config.gradient_accumulation_steps = 2, 4, 8
        oom_changes.append("Smoke OOM: cleared CUDA cache; train batch 4->2, eval 8->4, accumulation 4->8; all rows retained.")
        print(oom_changes[-1], flush=True)
        smoke = gpu_smoke_test(tokenizer, list(train.select(range(8))["text"]), config, hardware["fp16"])
    write_json(METRICS / "transformer_smoke_test.json", smoke)
    print("GPU SMOKE:", json.dumps(smoke), flush=True)
    if args.smoke_only:
        return

    train_tokens = tokenize_split(train, tokenizer, config.max_length)
    validation_tokens = tokenize_split(validation, tokenizer, config.max_length)
    configuration = {
        **config.to_dict(), "hardware": hardware, "oom_changes": oom_changes,
        "phase2_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "known_exact_text_overlap": metadata["text_overlap"],
        "versions": {name: version(name) for name in ("torch", "transformers", "accelerate", "datasets")},
        "gradient_checkpointing": True, "padding": "dynamic, multiple of 8",
        "train_sequences_at_max_length": sum(len(ids) == config.max_length for ids in train_tokens["input_ids"]),
        "length_note": "At-limit sequences include exact fits and truncated reviews; this is not an exact truncation count.",
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(dataset["test"])},
    }
    write_json(ARTIFACTS / "training_config.json", configuration)
    try:
        trainer, seconds = train_attempt(config, hardware, tokenizer, train_tokens, validation_tokens,
                                         ARTIFACTS / "checkpoints")
    except torch.cuda.OutOfMemoryError:
        trainer = None
    if trainer is None:
        gc.collect()
        torch.cuda.empty_cache()
        if config.train_batch_size == 2:
            raise RuntimeError("CUDA OOM persisted at batch 2; full data retained, training stopped.")
        config.train_batch_size, config.eval_batch_size, config.gradient_accumulation_steps = 2, 4, 8
        oom_changes.append("Training OOM: cleared CUDA cache; restarting full dataset with train batch 2, eval 4, accumulation 8.")
        print(oom_changes[-1], flush=True)
        configuration.update(config.to_dict())
        configuration["oom_changes"] = oom_changes
        write_json(ARTIFACTS / "training_config.json", configuration)
        trainer, seconds = train_attempt(config, hardware, tokenizer, train_tokens, validation_tokens,
                                         ARTIFACTS / "checkpoints_batch2")
    selection = {
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_f1": trainer.state.best_metric,
        "training_seconds": seconds,
        "epochs": [row for row in trainer.state.log_history if "eval_f1" in row],
        "history": trainer.state.log_history,
        "test_used_for_selection": False,
    }
    write_json(METRICS / "transformer_selection.json", selection)
    print("BEST CHECKPOINT:", json.dumps({key: value for key, value in selection.items() if key != "history"}, indent=2), flush=True)

    # Test tokenization/prediction happens only after selection and best-model saving.
    test_tokens = tokenize_split(dataset["test"], tokenizer, config.max_length)
    try:
        prediction = trainer.predict(test_tokens, metric_key_prefix="test")
    except torch.cuda.OutOfMemoryError:
        prediction = None
    if prediction is None:
        gc.collect()
        torch.cuda.empty_cache()
        if config.eval_batch_size <= 4:
            raise RuntimeError("Test prediction OOM at batch 4; no test metrics were used for tuning.")
        config.eval_batch_size = 4
        trainer.args.per_device_eval_batch_size = 4
        oom_changes.append("Test inference OOM: cleared CUDA cache and reduced evaluation batch 8->4; model unchanged.")
        configuration.update(config.to_dict())
        configuration["oom_changes"] = oom_changes
        write_json(ARTIFACTS / "training_config.json", configuration)
        print(oom_changes[-1], flush=True)
        prediction = trainer.predict(test_tokens, metric_key_prefix="test")
    logits = prediction.predictions
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    predicted = probabilities.argmax(axis=-1)
    np.savez_compressed(ARTIFACTS / "test_predictions.npz", labels=prediction.label_ids,
                        predictions=predicted, probabilities=probabilities)
    metrics = evaluate_predictions(prediction.label_ids, predicted)
    write_json(METRICS / "transformer_test_metrics.json", {
        **metrics, "test_evaluation_count": 1, "best_checkpoint": selection["best_checkpoint"]})
    print("FINAL TEST:", json.dumps(metrics, indent=2), flush=True)
    # Reporting reads stored outputs and does not perform another prediction pass.
    from imdb_sentiment.evaluation.transformer_reports import generate_reports
    generate_reports(ROOT, list(dataset["test"]["text"]))


if __name__ == "__main__":
    main()