"""Validation-only model selection and report plots."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def select_best(validation_results):
    # Test metrics are never inputs to this decision; insertion order breaks exact ties.
    return max(validation_results, key=lambda name: (
        validation_results[name]["f1"], validation_results[name]["accuracy"]))


def plot_confusion(metrics, title, path):
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    ax.imshow(matrix, cmap="Blues", vmin=0)
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f"{value:,}", ha="center", va="center", fontsize=16,
                color="white" if value > matrix.max() / 2 else "#172033")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"],
           yticklabels=["Negative", "Positive"], xlabel="Predicted label",
           ylabel="True label", title=title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_comparison(results, path):
    names = list(results)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for offset, metric, color in [(-0.18, "accuracy", "#2563EB"), (0.18, "f1", "#D97706")]:
        bars = ax.bar(x + offset, [results[n][metric] for n in names],
                      width=0.36, label=metric.upper(), color=color)
        ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.set(xticks=x, xticklabels=[n.replace("_", " ").title() for n in names],
           ylim=(0, 1.08), ylabel="Score", title="Classical baselines: validation only")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=160)
    plt.close(fig)