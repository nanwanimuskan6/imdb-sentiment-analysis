"""Phase 3: train two baselines, select on validation, evaluate winner once."""

import json
from pathlib import Path
from time import perf_counter
from importlib.metadata import version
import hashlib
import sys
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imdb_sentiment.data.loading import load_imdb_dataset
from imdb_sentiment.features.tfidf import fit_training_features
from imdb_sentiment.models.logistic_regression import build_model as logistic_regression
from imdb_sentiment.models.linear_svc import build_model as linear_svc
from imdb_sentiment.evaluation.metrics import evaluate_predictions
from imdb_sentiment.evaluation.comparison import select_best, plot_confusion, plot_comparison
from imdb_sentiment.evaluation.error_analysis import save_examples


def load_saved_training_partition(dataset, metadata):
    """Verify source identity and reuse Phase 2 row membership exactly."""
    for name in ("train", "test"):
        if dataset[name]._fingerprint != metadata["source_fingerprints"][name]:
            raise ValueError(f"{name} source fingerprint differs from Phase 2.")
    member = metadata["membership"]
    train_ids = member["train_original_train_indices"]
    val_ids = member["validation_original_train_indices"]
    if len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids)):
        raise ValueError("Duplicate source row indices.")
    if set(train_ids) & set(val_ids):
        raise ValueError("Train/validation source row overlap.")
    if set(train_ids) | set(val_ids) != set(range(len(dataset["train"]))):
        raise ValueError("Source rows are missing or out of range.")
    return dataset["train"].select(train_ids), dataset["train"].select(val_ids)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    metrics_dir = ROOT / "reports/metrics"
    result_path = metrics_dir / "classical_results.json"
    # A rerun must not silently repeat the final test evaluation.
    if result_path.exists():
        raise SystemExit("Final results already exist. Inspect saved results; test evaluation will not be repeated.")
    metadata_path = metrics_dir / "split_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset = load_imdb_dataset()
    train, validation = load_saved_training_partition(dataset, metadata)
    if (len(train), len(validation), len(dataset["test"])) != (20000, 5000, 25000):
        raise ValueError("Unexpected Phase 2 split sizes.")
    train_texts, validation_texts = list(train["text"]), list(validation["text"])
    train_labels, validation_labels = np.asarray(train["label"]), np.asarray(validation["label"])

    start = perf_counter()
    vectorizer, x_train, x_validation = fit_training_features(train_texts, validation_texts)
    vectorizer_seconds = perf_counter() - start
    print(f"TF-IDF fit/validation transform: {vectorizer_seconds:.2f}s | {x_train.shape}", flush=True)

    artifact_dir = ROOT / "artifacts/classical"
    figures_dir = ROOT / "reports/figures"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models, validation_results, times = {}, {}, {}
    # Treat nonconvergence as a failure before accessing test labels.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for name, factory in (("logistic_regression", logistic_regression), ("linear_svc", linear_svc)):
            model = factory()
            start = perf_counter()
            model.fit(x_train, train_labels)
            times[name] = perf_counter() - start
            predictions = model.predict(x_validation)
            validation_results[name] = evaluate_predictions(validation_labels, predictions)
            models[name] = model
            print(f"{name}: fit={times[name]:.2f}s | validation={validation_results[name]}", flush=True)
            joblib.dump(model, artifact_dir / f"{name}.joblib", compress=3)
            plot_confusion(validation_results[name], f"{name.replace('_', ' ').title()} - validation",
                           figures_dir / f"{name}_validation_confusion.png")

    winner = select_best(validation_results)
    joblib.dump(vectorizer, artifact_dir / "tfidf_vectorizer.joblib", compress=3)
    plot_comparison(validation_results, figures_dir / "classical_validation_comparison.png")
    selection = {
        "winner": winner, "criterion": "Validation F1, then validation accuracy",
        "validation": validation_results, "fit_seconds": times,
        "vectorizer_fit_and_validation_transform_seconds": vectorizer_seconds,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "model_parameters": {name: model.get_params() for name, model in models.items()},
        "tfidf_parameters": {key: vectorizer.get_params()[key] for key in (
            "lowercase", "strip_accents", "ngram_range", "sublinear_tf", "min_df",
            "max_df", "max_features", "stop_words", "norm")},
        "versions": {name: version(name) for name in ("scikit-learn", "numpy", "joblib", "datasets")},
        "phase2_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "known_exact_text_overlap": metadata["text_overlap"],
        "training_rows": len(train), "validation_rows": len(validation),
        "refit_on_train_plus_validation": False,
    }
    # Persist the selection before inspecting test text/labels or computing predictions.
    write_json(metrics_dir / "classical_selection.json", selection)
    print(f"Selected using validation only: {winner}", flush=True)

    test_texts = list(dataset["test"]["text"])
    test_labels = np.asarray(dataset["test"]["label"])
    x_test = vectorizer.transform(test_texts)
    predictions = models[winner].predict(x_test)  # The single final test prediction pass.
    scores = models[winner].decision_function(x_test)
    results = {**selection, "test": evaluate_predictions(test_labels, predictions),
               "test_rows": len(test_labels), "test_evaluation_count": 1,
               "decision_score_note": "Positive favors label 1; negative favors label 0. Not a calibrated probability."}
    write_json(result_path, results)
    # Save predictions so reporting fixes never require re-evaluating the test set.
    np.savez_compressed(artifact_dir / "best_test_predictions.npz",
                        true_labels=test_labels, predictions=predictions, decision_scores=scores)
    plot_confusion(results["test"], f"Best classical: {winner.replace('_', ' ').title()} - test",
                   figures_dir / "best_classical_test_confusion.png")
    save_examples(test_texts, test_labels, predictions, scores, ROOT / "reports/errors", winner)
    print("FINAL TEST:", json.dumps(results["test"], indent=2), flush=True)
    print(f"Saved results: {result_path}", flush=True)


if __name__ == "__main__":
    main()