"""Save deterministic examples from the one final test prediction pass."""

import csv
from pathlib import Path


def save_examples(texts, true_labels, predictions, scores, output_dir, model_name, limit=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = {"correct_positive": (1, 1), "correct_negative": (0, 0),
              "false_positive": (0, 1), "false_negative": (1, 0)}
    saved = {}
    for category, (true, predicted) in groups.items():
        rows = []
        for index, (text, label, pred, score) in enumerate(zip(texts, true_labels, predictions, scores)):
            if int(label) == true and int(pred) == predicted:
                rows.append({
                    "test_row_index": index, "review": text,
                    "true_label": int(label), "predicted_label": int(pred),
                    "true_sentiment": "positive" if label else "negative",
                    "predicted_sentiment": "positive" if pred else "negative",
                    "decision_score": float(score), "model": model_name,
                })
                if len(rows) == limit:
                    break
        if len(rows) < limit:
            raise ValueError(f"Only {len(rows)} examples available for {category}.")
        path = output_dir / f"classical_{category}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        saved[category] = str(path)
    return saved