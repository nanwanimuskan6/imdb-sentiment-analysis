"""Run five illustrative reviews; not an evaluation of the official IMDb test set."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from imdb_sentiment.inference import SentimentPredictor
from imdb_sentiment.nuanced_analysis import NuancedAnalyzer, IronyAnalyzer

EXAMPLES = [
    "The movie was beautifully made. The acting was excellent, the story kept me engaged, and I loved the ending.",
    "The movie was painfully boring. The story made no sense, the acting was terrible, and I regretted watching it.",
    "The acting was excellent and the cinematography was beautiful, but the story was slow and the ending disappointed me.",
    "Amazing movie. I especially loved wasting two hours of my life waiting for something interesting to happen.",
    "it was ok ok i m confused what i felt",
]


def main():
    binary = SentimentPredictor()
    print("Loading CPU NLI model...", flush=True)
    nuanced = NuancedAnalyzer()
    print("Loading CPU irony model...", flush=True)
    irony = IronyAnalyzer()
    results = []
    for index, review in enumerate(EXAMPLES, 1):
        result = {"example": index, "review": review, "binary": binary.predict(review),
                  "nuanced": nuanced.analyze(review), "irony": irony.analyze(review)}
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    path = ROOT / "reports/errors/nuanced_examples.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("Saved:", path, flush=True)


if __name__ == "__main__":
    main()
