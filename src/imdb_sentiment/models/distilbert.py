"""Memory-conscious DistilBERT setup; raw text goes directly to its tokenizer."""

from dataclasses import dataclass, asdict
import gc
from time import perf_counter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

MODEL_NAME = "distilbert-base-uncased"


@dataclass
class TransformerConfig:
    model_name: str = MODEL_NAME
    max_length: int = 256
    train_batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    seed: int = 42
    max_grad_norm: float = 1.0

    def to_dict(self):
        return {**asdict(self), "effective_batch_size":
                self.train_batch_size * self.gradient_accumulation_steps}


def hardware_info():
    available = torch.cuda.is_available()
    return {
        "torch_version": torch.__version__, "cuda_available": available,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "gpu_vram_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3 if available else None,
        "fp16": available and torch.cuda.get_device_capability(0)[0] >= 7,
    }


def load_tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")


def load_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )


def tokenize_split(split, tokenizer, max_length=256):
    # No classical preprocessing: retain negation, punctuation, and original text.
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)
    result = split.map(tokenize, batched=True, remove_columns=["text"])
    return result.rename_column("label", "labels")


def collator(tokenizer):
    # Pad each batch only to its longest sequence, rounded for Tensor Core efficiency.
    return DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)


def gpu_smoke_test(tokenizer, texts, config, fp16):
    """Exercise full-length batches, backward, Adam state, and evaluation before training."""
    model = load_model().cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    model.train()
    for step in range(2):
        inputs = tokenizer(texts[:config.train_batch_size], truncation=True,
                           padding="max_length", max_length=config.max_length, return_tensors="pt")
        inputs = {key: value.cuda() for key, value in inputs.items()}
        inputs["labels"] = torch.tensor([i % 2 for i in range(config.train_batch_size)], device="cuda")
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            output = model(**inputs)
        if not torch.isfinite(output.loss):
            raise ValueError("Smoke test loss is not finite.")
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        batch = tokenizer(texts[:config.eval_batch_size], truncation=True, padding="max_length",
                          max_length=config.max_length, return_tensors="pt").to("cuda")
        logits = model(**batch).logits
        assert logits.shape == (config.eval_batch_size, 2)
        assert torch.isfinite(logits).all()
    torch.cuda.synchronize()
    result = {"passed": True, "seconds": perf_counter() - start,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
              "train_batch_size": config.train_batch_size, "eval_batch_size": config.eval_batch_size,
              "max_length": config.max_length, "fp16": fp16}
    del model, optimizer, scaler, inputs, output, batch, logits
    gc.collect()
    torch.cuda.empty_cache()
    return result