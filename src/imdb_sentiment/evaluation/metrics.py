"""Binary sentiment metrics with a fixed positive label and matrix order."""

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


def evaluate_predictions(true_labels, predictions):
    if not set(predictions).issubset({0, 1}):
        raise ValueError("Predictions must be binary sentiment labels.")
    return {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "precision": float(precision_score(true_labels, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(true_labels, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(true_labels, predictions, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(true_labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            true_labels, predictions, labels=[0, 1],
            target_names=["negative", "positive"], output_dict=True, zero_division=0),
    }