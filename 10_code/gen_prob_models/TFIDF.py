import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download("punkt")

# Your data
titles = []
reviews = []
title_to_tag = {}

# Read reviews and titles from file
with open("../00_data/LDA_test_texts/reviews_test.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        split_line = line.split("):")
        if len(split_line) > 1:
            title = split_line[0].strip()
            review = ":".join(split_line[1:]).strip()
            titles.append(title)
            reviews.append(review)

# Read tags from file and map titles to tags
with open("../00_data/LDA_test_texts/oscar_status.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        split_line = line.split("):")
        if len(split_line) > 1:
            title = split_line[0].strip()
            tag = ":".join(split_line[1:]).strip()
            title_to_tag[title] = tag

# Create labels corresponding to reviews
labels = [title_to_tag[title] for title in titles]

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Transform the reviews into TF-IDF features
X = tfidf_vectorizer.fit_transform(reviews)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Initialize and train a classifier (e.g., Naive Bayes)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predict on the test set
predictions = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


def classify_review(review):
    # Vectorize the input review using the pre-trained TfidfVectorizer
    review_vec = tfidf_vectorizer.transform([review])

    # Predict the class of the input review using the trained classifier
    prediction = nb_classifier.predict(review_vec)[0]

    return prediction


# Example usage:
# user_review = input("Enter your review: ")
user_review = "I loved this movie! The acting was great, plot was wonderful, and there were pyrotechnics!"
classification = classify_review(user_review)

# Map the prediction to the corresponding category (winner, nominee, or no recognition)
if classification == "Winner":
    print("This review is classified as a Winner!")
elif classification == "Nominee":
    print("This review is classified as a Nominee!")
else:
    print("This review has no recognition.")
