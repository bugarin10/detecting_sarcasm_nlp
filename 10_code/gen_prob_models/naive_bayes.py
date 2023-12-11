import numpy as np, pandas as pd
import seaborn as sns
import random
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

sns.set()

import pandas as pd

data = pd.read_csv("../../00_data/final_data.csv")

df = pd.DataFrame(data)

# Splitting the 'top_critics' column into separate rows
df["top_critics"] = df["top_critics"].str.split(r"\., |\. ,")
df = df.explode("top_critics").reset_index(drop=True)
df["top_critics"] = df["top_critics"].apply(lambda x: str(x).lower())


nominees = df[(df["winner"] == 2) | (df["winner"] == 1)]
nominee_reviews = nominees["top_critics"].tolist()
no_recognition = df[df["winner"] == 0]
no_recognition_reviews = no_recognition["top_critics"].tolist()

# Combine all reviews and create labels
all_reviews = nominee_reviews + no_recognition_reviews
labels = ["nominee"] * len(nominee_reviews) + ["no_recognition"] * len(
    no_recognition_reviews
)

categories = ["nominee", "no_recognition"]

X_train, X_test, y_train, y_test = train_test_split(
    all_reviews, labels, test_size=0.2, random_state=42
)

model_pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

model_pipeline.fit(X_train, y_train)
predicted_categories = model_pipeline.predict(X_test)

mat = confusion_matrix(y_test, predicted_categories)
sns.heatmap(mat.T, square=True, annot=True, fmt="d", cbar=False)
# xticklabels = ["nominee", "no_recognition"]
# yticklabels = ["nominee", "no_recognition"]
plt.xlabel("true label")
plt.ylabel("predicted label")
# plt.xticks(np.arange(2), xticklabels, rotation=45)
plt.show()

accuracy_score_1 = accuracy_score(y_test, predicted_categories)
print(f"Accuracy: {accuracy_score_1}")

classification_report_real = classification_report(y_test, predicted_categories)
print(classification_report_real)

# SENSITIVITY ANALYSIS (50/(50+17))

# Specificity()(acodring to the confusoion matrix it should be low)

# F1 score (accord)

# Predicting the label of a new review
new_review = "This movie was not remarkable, touching, or superb in any way"
new_review_1 = "Oscar, oscar, oscar for sure"

prediction = model_pipeline.predict([new_review_1])

print(f"This movie is predicted to be a {prediction[0]}")

# GENERATING SYNTHETIC DATA WITH TFIDF AND NAIVE BAYES

vectorizer = model_pipeline.named_steps["tfidfvectorizer"]
clf = model_pipeline.named_steps["multinomialnb"]

log_probs = clf.feature_log_prob_
# exponentiate with numpy = actual_probs
actual_probs = np.exp(log_probs)

feature_names = vectorizer.get_feature_names_out()

df_probs = pd.DataFrame(actual_probs.T, columns=["nominee", "no_recognition"])
df_probs["Words"] = feature_names
df_log_probs = pd.DataFrame(log_probs.T, columns=["nominee", "no_recognition"])
df_log_probs["Words"] = feature_names

# Iterate over the actual dataset, set the length of the review to be generated to the length of the review in the current row of actual dataset
# For each word in the review, randomly sample a word from the actual_probs dataframe
# Add the sampled word to the generated review
# Repeat until the generated review is the same length as the review in the actual dataset
# Add the generated review to a dataframe of generated reviews with nominee or no_recognition label
# Repeat for all reviews in the actual dataset


def sample_word(category):
    if category == "nominee":
        return np.random.choice(
            df_probs[df_probs["nominee"] > 0]["Words"],
            p=df_probs[df_probs["nominee"] > 0]["nominee"],
        )
    elif category == "no_recognition":
        return np.random.choice(
            df_probs[df_probs["no_recognition"] > 0]["Words"],
            p=df_probs[df_probs["no_recognition"] > 0]["no_recognition"],
        )
    else:
        # Handle cases where category is undefined or not recognized
        return ""


# Replace the section generating synthetic reviews with sampling based on category probabilities
df["generated_reviews"] = df.apply(
    lambda row: " ".join(
        [
            sample_word("nominee")
            if row["winner"] == 2 or row["winner"] == 1
            else sample_word("no_recognition")
            for _ in row["top_critics"].split()
        ]
    ),
    axis=1,
)
# df["generated_reviews"] = None  # Create the "generated_reviews" column

# df["generated_reviews"] = df.apply(
#     lambda row: " ".join(
#         [
#             np.random.choice(df_probs["Words"], p=df_probs["nominee"])
#             if row["winner"] == 2 or row["winner"] == 1
#             else np.random.choice(df_probs["Words"], p=df_probs["no_recognition"])
#             for _ in row["top_critics"].split()
#         ]
#     ),
#     axis=1,
# )

df.to_csv("../../00_data/generated_reviews.csv", index=False)
synth_nominees = df[(df["winner"] == 2) | (df["winner"] == 1)]
synth_nominee_reviews = synth_nominees["generated_reviews"].tolist()
synth_no_recognition = df[df["winner"] == 0]
synth_no_recognition_reviews = synth_no_recognition["generated_reviews"].tolist()

# Combine all reviews and create labels
synth_all_reviews = synth_nominee_reviews + synth_no_recognition_reviews
synth_labels = ["nominee"] * len(synth_nominee_reviews) + ["no_recognition"] * len(
    synth_no_recognition_reviews
)

categories = ["no_recognition", "nominee"]

X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(
    synth_all_reviews, synth_labels, test_size=0.2, random_state=42
)

model_pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

model_pipeline.fit(X_train_synth, y_train_synth)
predicted_categories = model_pipeline.predict(X_test_synth)

# Display the confusion matrix
mat = confusion_matrix(y_test_synth, predicted_categories)
sns.heatmap(mat.T, square=True, annot=True, fmt="d", cbar=False)
plt.xlabel("true label")
plt.ylabel("predicted label")
plt.show()

from sklearn.metrics import accuracy_score  # Ensure using the correct function

acc = accuracy_score(y_test_synth, predicted_categories)
print(f"Accuracy: {acc}")

classification_report_synth = classification_report(y_test_synth, predicted_categories)
print(classification_report_synth)
