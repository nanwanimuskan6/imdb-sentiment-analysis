"""Descriptive EDA on original review text; no cleaning or feature fitting."""

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np


def length_statistics(values):
    """Summarize lengths with NumPy's linear percentile interpolation."""
    if not values:
        return dict.fromkeys(("min", "max", "mean", "median", "p90", "p95"))
    return {
        "min": int(min(values)), "max": int(max(values)),
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def inspect_split(split):
    """Count duplicate rows after the first occurrence, using exact original text."""
    texts = list(split["text"])
    valid = [text for text in texts if text is not None]
    if any(not isinstance(text, str) for text in valid):
        raise ValueError("Non-null reviews must be strings.")
    counts = Counter(valid)
    label_counts = Counter(split["label"])
    features = split.features["label"]
    lengths = {
        "characters": [len(text) for text in valid],
        "words": [len(text.split()) for text in valid],
    }
    return {
        "samples": len(split),
        "class_counts": {features.int2str(k): v for k, v in sorted(label_counts.items())},
        "null_reviews": len(texts) - len(valid),
        # strip() is used only for checking emptiness; saved text is never modified.
        "empty_reviews": sum(not text.strip() for text in valid),
        "duplicate_reviews": sum(count - 1 for count in counts.values()),
        "duplicate_text_groups": sum(count > 1 for count in counts.values()),
        "lengths": {unit: length_statistics(values) for unit, values in lengths.items()},
    }, lengths


def duplicate_overlap(splits):
    """Report unique exact review texts shared by each pair of splits."""
    texts = {name: {text for text in split["text"] if text is not None}
             for name, split in splits.items()}
    return {
        f"{left}__{right}": len(texts[left] & texts[right])
        for left, right in combinations(texts, 2)
    }


def save_plots(statistics, lengths, output_dir):
    """Save full-range distributions; log frequency keeps long tails visible."""
    import matplotlib
    matplotlib.use("Agg")  # File output works on machines without a display.
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"train": "#2563EB", "test": "#D97706"}
    paths = []
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.titleweight": "bold",
    }):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for ax, name in zip(axes, statistics):
            values = [statistics[name]["class_counts"][label] for label in ("neg", "pos")]
            bars = ax.bar(["Negative", "Positive"], values, color=colors[name], width=0.55)
            ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=5)
            ax.set(title=f"Official {name} ({statistics[name]['samples']:,} reviews)",
                   ylabel="Reviews", ylim=(0, max(values) * 1.18))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        fig.suptitle("IMDb sentiment class distribution")
        path = output_dir / "class_distribution.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
        for unit in ("words", "characters"):
            maximum = max(max(lengths[name][unit]) for name in lengths)
            bins = np.linspace(0, max(1, maximum), 81)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True,
                                     sharey=True, layout="constrained")
            for ax, name in zip(axes, lengths):
                ax.hist(lengths[name][unit], bins=bins, color=colors[name],
                        alpha=0.85, edgecolor="white", linewidth=0.3)
                ax.set_yscale("log")
                ax.set(title=f"Official {name}", xlabel=f"Review length ({unit})",
                       ylabel="Reviews (log scale)")
                ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
                ax.grid(axis="y", alpha=0.2)
            fig.suptitle(f"IMDb review lengths: {unit}\nOriginal text; full range, shared bins")
            path = output_dir / f"review_length_{unit}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
    return paths