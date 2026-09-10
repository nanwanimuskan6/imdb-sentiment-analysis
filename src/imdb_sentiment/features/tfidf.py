"""Training-only sparse features for the classical baselines."""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer():
    # IDF downweights ubiquitous terms; sublinear TF reduces repetition's influence.
    # Unigrams capture individual cues; bigrams capture local phrases like "not good".
    # Keep stopwords: negation and short connecting words can carry sentiment.
    # Stemming/lemmatization can erase useful distinctions; benefits need validation.
    return TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, max_df=0.95, max_features=None,
        stop_words=None,
    )


def fit_training_features(train_texts, validation_texts):
    vectorizer = build_vectorizer()
    train_matrix = vectorizer.fit_transform(train_texts)
    validation_matrix = vectorizer.transform(validation_texts)
    return vectorizer, train_matrix, validation_matrix