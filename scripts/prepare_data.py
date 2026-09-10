"""Load and inspect IMDb without cleaning text or training models."""

from collections import Counter
from pathlib import Path
import sys

# Support direct execution before packaging is introduced in a later phase.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imdb_sentiment.data.loading import load_imdb_dataset


def main() -> None:
    """Print split metadata, original review samples, and class counts."""
    dataset = load_imdb_dataset()
    print(f"Available splits: {list(dataset.keys())}")
    for name, split in dataset.items():
        print(f"\n{name}: {len(split):,} examples")
        print(f"Columns: {split.column_names}")
        print(f"Features: {split.features}")

    # Metadata provides the label encoding; do not guess what integer IDs mean.
    label_feature = dataset["train"].features["label"]
    print(f"Label names: {label_feature.names}")
    label_mapping = {
        label_id: label_feature.int2str(label_id) for label_id in range(len(label_feature.names))
    }
    print("Label mapping: 0 = negative, 1 = positive")
    print(f"Dataset label IDs: {label_mapping}")
    readable_labels = {"neg": "negative", "pos": "positive"}

    print("\nThree sample reviews (train, original text):")
    for index, review in enumerate(dataset["train"].select(range(3)), start=1):
        label_name = label_feature.int2str(review["label"])
        print(f"\nSample {index} | sentiment: {readable_labels[label_name]}")
        print(review["text"])

    print("\nClass balance:")
    all_balanced = True
    # Unsupervised reviews have no sentiment labels and cannot measure class balance.
    for name in ("train", "test"):
        split = dataset[name]
        counts = Counter(split["label"])
        labels = split.features["label"]
        negative_id = labels.str2int("neg")
        positive_id = labels.str2int("pos")
        if set(counts) - {negative_id, positive_id}:
            raise ValueError(f"Unexpected sentiment labels in {name}: {counts}")
        negative, positive = counts[negative_id], counts[positive_id]
        balanced = negative == positive and len(split) > 0
        all_balanced = all_balanced and balanced
        print(
            f"{name}: negative={negative:,}, positive={positive:,} | "
            f"balanced: {'yes' if balanced else 'no'}"
        )
    print(f"Both labeled splits balanced: {'yes' if all_balanced else 'no'}")
    if "unsupervised" in dataset:
        print("Unsupervised split excluded from sentiment counts (unlabeled reviews).")


if __name__ == "__main__":
    main()