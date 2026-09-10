"""Offline policy checks for the experimental layer (no model training)."""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imdb_sentiment.nuanced_analysis import NuancedAnalyzer, evidence_label, review_units, UNCERTAIN


class NuancedTests(unittest.TestCase):
    def test_mixed_requires_conflicting_evidence(self):
        self.assertEqual(evidence_label(True, True), "Mixed")
        self.assertEqual(evidence_label(False, False), UNCERTAIN)
        self.assertEqual(evidence_label(True, False), "Positive")
        self.assertEqual(evidence_label(False, True), "Negative")

    def test_no_aspects_without_model_evidence(self):
        analyzer = object.__new__(NuancedAnalyzer)
        with patch.object(analyzer, "entailment", side_effect=lambda pairs: [0.1] * len(pairs)):
            result = analyzer.analyze("A review without a clear opinion.")
        self.assertEqual(result["aspects"], [])
        self.assertTrue(result["weak_evidence"])
        self.assertEqual(result["sentiment"], UNCERTAIN)


    def test_general_opinion_cannot_replace_aspect_evidence(self):
        analyzer = object.__new__(NuancedAnalyzer)
        # Even high NLI scores cannot invent a named aspect in generic praise.
        with patch.object(analyzer, "entailment", side_effect=lambda pairs: [0.99] * len(pairs)):
            result = analyzer.analyze("I loved it.")
        self.assertEqual(result["aspects"], [])
        self.assertEqual(result["sentiment"], UNCERTAIN)

    def test_supported_overall_opinion_without_aspects(self):
        analyzer = object.__new__(NuancedAnalyzer)
        for scores, expected in (([0.91, 0.03, 0.02], "Positive"),
                                 ([0.04, 0.92, 0.05], "Negative"),
                                 ([0.80, 0.05, 0.75], UNCERTAIN)):
            with self.subTest(scores=scores):
                with patch.object(analyzer, "entailment", side_effect=[
                    [0.0] * 8, [], scores
                ]):
                    result = analyzer.analyze("A general opinion.")
                self.assertEqual(result["sentiment"], expected)
                self.assertEqual(result["aspects"], [])

    def test_blank_input_and_explicit_limit(self):
        for text in ("", " \n\t"):
            with self.assertRaises(ValueError):
                review_units(text)
        units, limited = review_units("A sentence. " * 30)
        self.assertEqual(len(units), 24)
        self.assertTrue(limited)


if __name__ == "__main__":
    unittest.main()
