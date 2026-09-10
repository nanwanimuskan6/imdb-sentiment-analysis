"""Load the original IMDb splits without modifying review text."""

from datasets import DatasetDict, load_dataset


def load_imdb_dataset() -> DatasetDict:
    """Download IMDb on first use and reuse Hugging Face's cache afterward."""
    # Preserve official splits so inspection does not change the experiment data.
    return load_dataset("imdb")