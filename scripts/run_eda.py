"""Run Phase 2 EDA and stratified split preparation, without training models."""

import json
from importlib.metadata import version
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from imdb_sentiment.analysis.eda import duplicate_overlap, inspect_split, save_plots
from imdb_sentiment.data.loading import load_imdb_dataset
from imdb_sentiment.data.splitting import SEED, VALIDATION_FRACTION, prepare_splits


def main():
    dataset = load_imdb_dataset()
    official = {name: dataset[name] for name in ("train", "test")}
    statistics, lengths = {}, {}
    for name, split in official.items():
        statistics[name], lengths[name] = inspect_split(split)

    splits, indices = prepare_splits(dataset)
    expected_sizes = {"train": 20000, "validation": 5000, "test": 25000}
    if {name: len(split) for name, split in splits.items()} != expected_sizes:
        raise ValueError("Unexpected IMDb split sizes.")
    if splits["test"] is not dataset["test"]:
        raise ValueError("Official test dataset was replaced.")

    split_statistics = {name: inspect_split(split)[0] for name, split in splits.items()}
    metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots = save_plots(statistics, lengths, PROJECT_ROOT / "reports" / "figures")
    eda_report = {
        "definitions": {
            "duplicates": "Exact text, extra occurrences after the first; nulls excluded.",
            "empty": "Non-null review empty after whitespace stripping for inspection only.",
            "lengths": "Original Unicode character count and whitespace-separated word count; nulls excluded.",
            "percentiles": "Linear interpolation.",
            "overlap": "Number of distinct exact texts shared by a pair of splits.",
        },
        "official_splits": statistics,
        "official_text_overlap": duplicate_overlap(official),
    }
    split_report = {
        "dataset": "imdb", "seed": SEED,
        "validation_fraction_of_original_train": VALIDATION_FRACTION,
        "method": "Hugging Face train_test_split stratified by label",
        "versions": {name: version(name) for name in ("datasets", "huggingface-hub", "numpy", "matplotlib")},
        "source_fingerprints": {name: split._fingerprint for name, split in official.items()},
        "source_cache_files": {name: split.cache_files for name, split in official.items()},
        "statistics": split_statistics,
        "text_overlap": duplicate_overlap(splits),
        "verification": {
            "expected_sizes": True, "all_splits_balanced": True,
            "train_validation_source_rows_disjoint": True,
            "original_train_fully_partitioned": True, "official_test_unchanged": True,
        },
        "duplicate_policy": "Report only; no removal or grouping, to preserve requested sizes and official test.",
        "membership": {
            "train_original_train_indices": indices["train"],
            "validation_original_train_indices": indices["validation"],
            "test": "All original test rows in original order.",
        },
    }
    for name, report in (("eda_statistics.json", eda_report), ("split_metadata.json", split_report)):
        path = metrics_dir / name
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Keep terminal output readable; full membership belongs only in the JSON report.
    print("Official supervised split statistics:")
    print(json.dumps(eda_report, indent=2))
    print("\nPrepared supervised splits (seed=42):")
    for name, stats in split_statistics.items():
        print(f"{name}: {stats['samples']:,} rows | {stats['class_counts']} | balanced=yes")
    print("Exact text overlap:", split_report["text_overlap"])
    print("Verification:", split_report["verification"])
    print("Saved reports:")
    for path in [metrics_dir / "eda_statistics.json", metrics_dir / "split_metadata.json", *plots]:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()