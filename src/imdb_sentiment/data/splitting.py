"""Reproducible stratified preparation without changing text or official test rows."""

from collections import Counter

from datasets import Dataset, DatasetDict, Features, Value

SEED = 42
VALIDATION_FRACTION = 0.2


def prepare_splits(dataset: DatasetDict, seed: int = SEED):
    """Return supervised splits and original-training row indices for reproducibility."""
    # Split lightweight row IDs, avoiding a second copy of the full review text.
    indexed = Dataset.from_dict(
        {"label": list(dataset["train"]["label"]),
         "source_index": list(range(len(dataset["train"])))},
        features=Features({"label": dataset["train"].features["label"],
                           "source_index": Value("int64")}),
    )
    partition = indexed.train_test_split(
        test_size=VALIDATION_FRACTION, stratify_by_column="label",
        seed=seed, keep_in_memory=True,
    )
    indices = {
        "train": list(partition["train"]["source_index"]),
        "validation": list(partition["test"]["source_index"]),
    }
    splits = DatasetDict({
        "train": dataset["train"].select(indices["train"], keep_in_memory=True),
        "validation": dataset["train"].select(indices["validation"], keep_in_memory=True),
        "test": dataset["test"],
    })
    # Row disjointness is distinct from duplicate text, which EDA reports separately.
    if set(indices["train"]) & set(indices["validation"]):
        raise ValueError("Training and validation row indices overlap.")
    if set(indices["train"]) | set(indices["validation"]) != set(range(len(dataset["train"]))):
        raise ValueError("Training and validation do not cover the original training split.")
    for name, split in splits.items():
        counts = Counter(split["label"])
        if set(counts) != {0, 1} or counts[0] != counts[1]:
            raise ValueError(f"{name} is not balanced: {counts}")
    return splits, indices