"""Small offline checks for Phase 2 statistics and split integrity."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from imdb_sentiment.analysis.eda import inspect_split, duplicate_overlap
from imdb_sentiment.data.splitting import prepare_splits

FEATURES = Features({"text": Value("string"), "label": ClassLabel(names=["neg", "pos"])})


class Phase2Tests(unittest.TestCase):
    def test_statistics_keep_null_empty_and_duplicate_definitions_distinct(self):
        split = Dataset.from_dict(
            {"text": [None, "", "  ", "hello world", "hello world"], "label": [0, 0, 1, 1, 0]},
            features=FEATURES,
        )
        stats, lengths = inspect_split(split)
        self.assertEqual((stats["null_reviews"], stats["empty_reviews"], stats["duplicate_reviews"]), (1, 2, 1))
        self.assertEqual(lengths["characters"], [0, 2, 11, 11])
        self.assertEqual(lengths["words"], [0, 0, 2, 2])
        self.assertEqual(stats["lengths"]["characters"]["median"], 6.5)
        self.assertIsNone(split[0]["text"])
        self.assertEqual(split[2]["text"], "  ")

    def test_split_reproducibility_coverage_balance_and_unchanged_text(self):
        train = Dataset.from_dict(
            {"text": [f"<br /> Review {i}  " for i in range(20)], "label": [0, 1] * 10},
            features=FEATURES,
        )
        test = Dataset.from_dict({"text": ["test negative", "test positive"], "label": [0, 1]}, features=FEATURES)
        source = DatasetDict({"train": train, "test": test})
        splits, indices = prepare_splits(source)
        _, repeated = prepare_splits(source)
        self.assertEqual(indices, repeated)
        self.assertEqual([len(splits[name]) for name in ("train", "validation", "test")], [16, 4, 2])
        self.assertIs(splits["test"], test)
        self.assertEqual(set(indices["train"]) & set(indices["validation"]), set())
        self.assertEqual(sorted(indices["train"] + indices["validation"]), list(range(20)))
        for name in ("train", "validation"):
            self.assertEqual(splits[name].to_dict(), train.select(indices[name]).to_dict())
        self.assertEqual(duplicate_overlap(splits), {"train__validation": 0, "train__test": 0, "validation__test": 0})


if __name__ == "__main__":
    unittest.main()