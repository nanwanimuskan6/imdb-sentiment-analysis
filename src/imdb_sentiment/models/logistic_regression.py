"""Reproducible binary logistic regression for sparse text."""

from sklearn.linear_model import LogisticRegression


def build_model():
    # LIBLINEAR supports sparse binary classification; L2 is the default penalty.
    return LogisticRegression(C=1.0, solver="liblinear", max_iter=3000, random_state=42)