import nltk
import random
import numpy as np

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from numpy.typing import NDArray
from typing import List, Mapping, Optional, Sequence
import gensim
import gensim.downloader
import nltk
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

import os
import pandas as pd
from pandas import DataFrame

FloatArray = NDArray[np.float64]


# Creating function to read synthetic data:


def read_data(path1, path2):
    # Read reviews and titles from file
    titles = []
    reviews = []
    title_to_tag = {}

    with open(path1, "r") as file:
        lines = file.readlines()
        for line in lines:
            split_line = line.split("):")
            if len(split_line) > 1:
                title = split_line[0].strip()
                review = ":".join(split_line[1:]).strip()
                titles.append(title)
                reviews.append(review)

    # Read tags from file and map titles to tags
    with open(path2, "r") as file:
        lines = file.readlines()
        for line in lines:
            split_line = line.split("):")
            if len(split_line) > 1:
                title = split_line[0].strip()
                tag = ":".join(split_line[1:]).strip()
                title_to_tag[title] = tag

    # Create labels corresponding to reviews
    labels = [title_to_tag[title] for title in titles]

    return titles, reviews, labels


# Creating function to change synthetic data to dataframe:
def data_to_df(titles, reviews, labels):
    # Create dataframe
    df = pd.DataFrame(
        list(zip(titles, reviews, labels)), columns=["title", "review", "label"]
    )

    return df


# Writing a function to create documents for winners, nominees, and no recognition:
def create_documents(df):
    # Create documents for winners, nominees, and no recognition
    winners = df[df["label"] == "Winner"]
    winners = winners["review"].tolist()
    nominees = df[df["label"] == "Nominee"]
    nominees = nominees["review"].tolist()
    no_recognition = df[df["label"] == "No Recognition"]
    no_recognition = no_recognition["review"].tolist()

    return winners, nominees, no_recognition


def create_doc_list(winner_list, nominee_list, no_rec_list):
    documents = [winner_list, nominee_list, no_rec_list]
    return documents


# Create vocabulary map
def create_vocabulary_map(documents):
    stop_words = set(stopwords.words("english"))
    # Create vocabulary from all tokens in documents
    vocabulary = sorted(
        set(
            token
            for document in documents
            for token in document
            if token not in stop_words
        )
    ) + [None]
    # Create vocabulary map with token indices
    vocabulary_map = {token: idx for idx, token in enumerate(vocabulary)}

    return vocabulary_map


def onehot(
    vocabulary_map: Mapping[Optional[str], int], token: Optional[str]
) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary_map),))
    idx = vocabulary_map.get(token, len(vocabulary_map) - 1)
    embedding[idx] = 1
    return embedding


def sum_token_embeddings(
    token_embeddings: Sequence[FloatArray],
) -> FloatArray:
    """Sum the token embeddings."""
    total: FloatArray = np.array(token_embeddings).sum(axis=0)
    return total


def split_train_test(
    X: FloatArray, y: FloatArray, test_percent: float = 10
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Split data into training and testing sets."""

    N = len(y)
    data_idx = list(range(N))
    random.shuffle(data_idx)
    break_idx = round(test_percent / 100 * N)
    training_idx = data_idx[break_idx:]
    testing_idx = data_idx[:break_idx]
    X_train = X[training_idx, :]
    y_train = y[training_idx]
    X_test = X[testing_idx, :]
    y_test = y[testing_idx]
    return X_train, y_train, X_test, y_test


def generate_data_token_counts(
    winner_doc: list[list[str]],
    nominee_doc: list[list[str]],
    no_rec_doc: list[list[str]],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate training and testing data with raw token counts."""
    documents = create_doc_list(winner_doc, nominee_doc, no_rec_doc)
    vocabulary_map = create_vocabulary_map(documents)
    X: FloatArray = np.array(
        [
            sum_token_embeddings([onehot(vocabulary_map, token) for token in sentence])
            for sentence in winner_doc
        ]
        + [
            sum_token_embeddings([onehot(vocabulary_map, token) for token in sentence])
            for sentence in nominee_doc
        ]
        + [
            sum_token_embeddings([onehot(vocabulary_map, token) for token in sentence])
            for sentence in no_rec_doc
        ]
    )
    y: FloatArray = np.array(
        [0 for sentence in winner_doc]
        + [1 for sentence in nominee_doc]
        + [-1 for sentence in no_rec_doc]
    )
    return split_train_test(X, y)


def generate_data_tfidf(
    winner_doc: list[list[str]],
    nominee_doc: list[list[str]],
    no_rec_doc: list[list[str]],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate training and testing data with TF-IDF scaling."""
    X_train, y_train, X_test, y_test = generate_data_token_counts(
        winner_doc, nominee_doc, no_rec_doc
    )
    tfidf = TfidfTransformer(norm=None).fit(X_train)
    X_train = tfidf.transform(X_train)
    X_test = tfidf.transform(X_test)
    return X_train, y_train, X_test, y_test


# Defining function for Naive Bayes classification:
def naive_bayes(X_train, X_test, y_train, y_test):
    # Initialize and train a classifier (e.g., Naive Bayes)
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)

    # Predict on the test set
    predictions = nb_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy score: {accuracy:.3f}")

    return predictions, accuracy


def prep_review(review: str) -> FloatArray:
    stop_words = set(stopwords.words("english"))
    # Create vocabulary map
    vocabulary = sorted(set(word for word in reviews if word not in stop_words)) + [
        None
    ]  # Assuming 'reviews' contains all words from your training data
    vocabulary_map = {word: idx for idx, word in enumerate(vocabulary)}
    tokenized_review = word_tokenize(review.lower())

    # Create one-hot encoding for review using the provided vocabulary_map
    one_hot_review = [onehot(vocabulary_map, word) for word in tokenized_review]

    # Sum token embeddings
    sum_review = sum_token_embeddings(one_hot_review)

    # Reshape review
    reshaped_review = sum_review.reshape(1, -1)
    return reshaped_review


def classify_review(df: DataFrame, review: str) -> str:
    winners, nominees, no_recognition = create_documents(df)
    X_train, y_train, X_test, y_test = generate_data_tfidf(
        winners, nominees, no_recognition
    )
    _, accuracy = naive_bayes(X_train, X_test, y_train, y_test)

    tokenized_review = prep_review(review)

    best_accuracy = 0
    best_alpha = 0

    for alpha in [0.1, 0.5, 1.0, 1.5, 2.0]:
        nb_classifier = MultinomialNB(alpha=alpha)
        nb_classifier.fit(X_train, y_train)
        predictions = nb_classifier.predict(X_test)
        current_accuracy = accuracy_score(y_test, predictions)

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_alpha = alpha

    print(f"Best accuracy: {best_accuracy:.3f} with alpha: {best_alpha}")
    return classify_review_with_model(tokenized_review, X_train, y_train, best_alpha)


def classify_review_with_model(tokenized_review, X_train, y_train, best_alpha):
    nb_classifier = MultinomialNB(alpha=best_alpha)
    nb_classifier.fit(X_train, y_train)
    classification = nb_classifier.predict(tokenized_review)
    return classification


if __name__ == "__main__":
    movies, reviews, labels = read_data(
        "../../00_data/LDA_test_texts/reviews_test.txt",
        "../../00_data/LDA_test_texts/oscar_status.txt",
    )
    oscars_df = data_to_df(movies, reviews, labels)
    input_review = "This movie was so good! I loved it!"

    classification = classify_review(oscars_df, input_review)
    print(f"Classification: {classification}")
