"""Generate Phase 4 plots/examples from saved predictions, never refit models."""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imdb_sentiment.evaluation.comparison import plot_confusion


def generate_reports(root, test_texts):
    root = Path(root)
    metrics_dir, figures = root / "reports/metrics", root / "reports/figures"
    read = lambda name: json.loads((metrics_dir / name).read_text(encoding="utf-8"))
    test = read("transformer_test_metrics.json")
    selection = read("transformer_selection.json")
    classical = read("classical_results.json")
    arrays = np.load(root / "artifacts/transformer/test_predictions.npz")
    labels, predicted, probabilities = arrays["labels"], arrays["predictions"], arrays["probabilities"]
    for category, pair in {"correct_positive": (1, 1), "correct_negative": (0, 0),
                           "false_positive": (0, 1), "false_negative": (1, 0)}.items():
        positions = np.flatnonzero((labels == pair[0]) & (predicted == pair[1]))[:10]
        rows = [{"test_row_index": int(i), "review": test_texts[i], "true_label": int(labels[i]),
                 "predicted_label": int(predicted[i]),
                 "predicted_probability": float(probabilities[i, predicted[i]]),
                 "positive_probability": float(probabilities[i, 1])} for i in positions]
        if len(rows) < 10:
            raise ValueError(f"Fewer than 10 {category} examples.")
        with (root / f"reports/errors/transformer_{category}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    plot_confusion(test, "DistilBERT - official test", figures / "transformer_test_confusion.png")
    epochs = selection["epochs"]
    best = max(epochs, key=lambda row: row["eval_f1"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    panels = [
        ("Validation: selection split", ["Logistic Regression", "LinearSVC", "DistilBERT"],
         [classical["validation"]["logistic_regression"], classical["validation"]["linear_svc"],
          {"accuracy": best["eval_accuracy"], "f1": best["eval_f1"]}]),
        ("Official test: selected models only", ["LinearSVC", "DistilBERT"], [classical["test"], test]),
    ]
    for ax, (title, names, results) in zip(axes, panels):
        x = np.arange(len(names))
        for offset, metric, color in [(-.18, "accuracy", "#2563EB"), (.18, "f1", "#D97706")]:
            bars = ax.bar(x + offset, [row[metric] for row in results], width=.36, label=metric.upper(), color=color)
            ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
        ax.set(xticks=x, xticklabels=names, ylim=(0, 1.08), ylabel="Score", title=title)
        ax.tick_params(axis="x", labelsize=9)
        ax.legend(loc="lower right")
    fig.savefig(figures / "classical_transformer_comparison.png", dpi=160)
    plt.close(fig)
    losses = [row for row in selection["history"] if "loss" in row]
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.plot([row["step"] for row in losses], [row["loss"] for row in losses], color="#2563EB")
    ax.set(xlabel="Optimizer step", ylabel="Logged training loss", title="DistilBERT training loss")
    ax.grid(alpha=.2)
    fig.savefig(figures / "transformer_training_loss.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for metric, color in [("f1", "#D97706"), ("accuracy", "#2563EB")]:
        ax.plot([row["epoch"] for row in epochs], [row[f"eval_{metric}"] for row in epochs],
                marker="o", label=metric.upper(), color=color)
    ax.set(xlabel="Epoch", ylabel="Validation score", title="DistilBERT validation by epoch",
           xticks=[row["epoch"] for row in epochs])
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(figures / "transformer_validation_epochs.png", dpi=160)
    plt.close(fig)