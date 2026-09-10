"""Linear SVM baseline for sparse high-dimensional TF-IDF."""

from sklearn.svm import LinearSVC


def build_model():
    # A linear margin classifier works well when sparse features outnumber reviews.
    return LinearSVC(C=1.0, dual="auto", max_iter=10000, random_state=42)