"""Offline tests for fitting boundaries, split identity, and binary outputs."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from train_classical import load_saved_training_partition
from imdb_sentiment.features.tfidf import build_vectorizer, fit_training_features
from imdb_sentiment.models.logistic_regression import build_model as logistic
from imdb_sentiment.models.linear_svc import build_model as svm
from imdb_sentiment.evaluation.comparison import select_best


class ClassicalTests(unittest.TestCase):
    def test_training_only_fit_shapes_and_binary_predictions(self):
        texts = ["wonderful good acting", "terrible bad acting", "great good story",
                 "awful bad story", "excellent good movie", "boring bad movie"] * 2
        labels = [1, 0, 1, 0, 1, 0] * 2
        validation = ["validationonly good", "validationonly bad"]
        test = ["testonly good", "testonly bad"]
        real = build_vectorizer()
        with patch("imdb_sentiment.features.tfidf.build_vectorizer", return_value=real):
            with patch.object(real, "fit_transform", wraps=real.fit_transform) as fitted:
                vectorizer, train_x, val_x = fit_training_features(texts, validation)
                fitted.assert_called_once_with(texts)
        vocab = dict(vectorizer.vocabulary_)
        idf = vectorizer.idf_.copy()
        test_x = vectorizer.transform(test)
        self.assertNotIn("validationonly", vocab)
        self.assertNotIn("testonly", vocab)
        self.assertEqual(vocab, vectorizer.vocabulary_)
        np.testing.assert_array_equal(idf, vectorizer.idf_)
        self.assertEqual(train_x.shape, (12, len(vocab)))
        self.assertEqual(val_x.shape, (2, len(vocab)))
        self.assertEqual(test_x.shape, (2, len(vocab)))
        for factory in (logistic, svm):
            model = factory().fit(train_x, labels)
            predictions = model.predict(test_x)
            self.assertEqual(predictions.shape, (2,))
            self.assertTrue(set(predictions).issubset({0, 1}))
            self.assertEqual(model.decision_function(test_x).shape, (2,))

    def test_saved_membership_does_not_mix_source_rows_or_test(self):
        features = Features({"text": Value("string"), "label": ClassLabel(names=["neg", "pos"])})
        train = Dataset.from_dict({"text": ["a", "b", "c", "d"], "label": [0, 1, 0, 1]}, features=features)
        test = Dataset.from_dict({"text": ["test"], "label": [0]}, features=features)
        dataset = DatasetDict({"train": train, "test": test})
        metadata = {"source_fingerprints": {"train": train._fingerprint, "test": test._fingerprint},
                    "membership": {"train_original_train_indices": [0, 1],
                                   "validation_original_train_indices": [2, 3]}}
        training, validation = load_saved_training_partition(dataset, metadata)
        self.assertEqual(training["text"], ["a", "b"])
        self.assertEqual(validation["text"], ["c", "d"])
        self.assertIs(dataset["test"], test)
        metadata["membership"]["validation_original_train_indices"] = [1, 3]
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_saved_training_partition(dataset, metadata)

    def test_selection_uses_f1_then_accuracy(self):
        self.assertEqual(select_best({"a": {"f1": .9, "accuracy": .8},
                                      "b": {"f1": .8, "accuracy": .99}}), "a")
        self.assertEqual(select_best({"a": {"f1": .9, "accuracy": .8},
                                      "b": {"f1": .9, "accuracy": .91}}), "b")


if __name__ == "__main__":
    unittest.main()