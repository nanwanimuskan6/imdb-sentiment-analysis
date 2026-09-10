"""Small Transformer checks; tokenizer uses the model cache populated by the smoke run."""

import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, DistilBertConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import train_transformer
from imdb_sentiment.models.distilbert import TransformerConfig, load_tokenizer, tokenize_split, collator


class TransformerTests(unittest.TestCase):
    def test_tokenizer_labels_padding_and_model_shape(self):
        tokenizer = load_tokenizer()
        source = Dataset.from_dict({"text": ["A good film!", "not good " * 300], "label": [1, 0]})
        tokenized = tokenize_split(source, tokenizer, 256)
        self.assertEqual(list(tokenized["labels"]), [1, 0])
        self.assertEqual(len(tokenized[1]["input_ids"]), 256)
        batch = collator(tokenizer)([tokenized[i] for i in range(2)])
        self.assertEqual(batch["input_ids"].shape, (2, 256))
        self.assertEqual(batch["attention_mask"].shape, (2, 256))
        self.assertEqual(set(batch["labels"].tolist()), {0, 1})
        model = AutoModelForSequenceClassification.from_config(
            DistilBertConfig(vocab_size=tokenizer.vocab_size, n_layers=1, dim=32,
                            hidden_dim=64, n_heads=2, num_labels=2))
        with torch.no_grad():
            output = model(**batch)
        self.assertEqual(output.logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(source[0]["text"], "A good film!")

    def test_trainer_only_receives_training_and_validation(self):
        training, validation = object(), object()
        tokenizer = Mock()
        config = TransformerConfig()
        trainer = Mock()
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(train_transformer, "Trainer", return_value=trainer) as factory, \
                 patch.object(train_transformer, "load_model", return_value=object()), \
                 patch.object(train_transformer, "training_arguments", return_value=object()), \
                 patch.object(train_transformer, "collator", return_value=object()):
                train_transformer.train_attempt(config, {"fp16": False}, tokenizer,
                                                 training, validation, Path(folder) / "absent")
        self.assertIs(factory.call_args.kwargs["train_dataset"], training)
        self.assertIs(factory.call_args.kwargs["eval_dataset"], validation)
        self.assertNotIn("test_dataset", factory.call_args.kwargs)
        self.assertNotIn("test", inspect.signature(train_transformer.train_attempt).parameters)

    def test_saved_phase2_membership_and_selection_settings(self):
        metadata = json.loads((ROOT / "reports/metrics/split_metadata.json").read_text())
        train = metadata["membership"]["train_original_train_indices"]
        validation = metadata["membership"]["validation_original_train_indices"]
        self.assertEqual((len(train), len(validation)), (20000, 5000))
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), set(range(25000)))
        config = TransformerConfig()
        self.assertEqual(config.to_dict()["effective_batch_size"], 16)
        with tempfile.TemporaryDirectory() as folder:
            args = train_transformer.training_arguments(config, {"fp16": False}, folder)
        self.assertEqual(args.metric_for_best_model, "f1")
        self.assertTrue(args.load_best_model_at_end)
        self.assertEqual(args.eval_strategy.value, "epoch")


if __name__ == "__main__":
    unittest.main()