from nltk.tokenize import word_tokenize
from gensim import corpora
import gensim

# Your data
titles = []
reviews = []
title_to_tag = {}

# Read reviews and titles from file
with open("LDA_test_texts/reviews_test.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        split_line = line.split("):")
        if len(split_line) > 1:
            title = split_line[0].strip()
            review = ":".join(split_line[1:]).strip()
            titles.append(title)
            reviews.append(review)

# Read tags from file and map titles to tags
with open("LDA_test_texts/oscar_status.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        split_line = line.split("):")
        if len(split_line) > 1:
            title = split_line[0].strip()
            tag = ":".join(split_line[1:]).strip()
            title_to_tag[title] = tag

# Create labels corresponding to reviews
labels = [title_to_tag[title] for title in titles]

# Tokenize the reviews
tokenized_reviews = [word_tokenize(review.lower()) for review in reviews]

# Create a dictionary and corpus
dictionary = corpora.Dictionary(tokenized_reviews)
corpus = [dictionary.doc2bow(review) for review in tokenized_reviews]

# Build the LDA model
lda_model = gensim.models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=10,  # Number of topics to extract (adjust as needed)
    passes=15,  # Number of passes through the corpus during training
    random_state=42,
)

# Adding each review's topic distribution to a list and adding the probabilities so each review has a probability
topic_distributions = []
for review in corpus:
    topic_distribution = lda_model[review]
    topic_distributions.append(topic_distribution)

for i in range(len(topic_distributions)):
    topic_distribution = topic_distributions[i]
    total_probability = 0
    for topic in topic_distribution:
        total_probability += topic[1]
    for j in range(len(topic_distribution)):
        topic_distribution[j] = (
            topic_distribution[j][0],
            topic_distribution[j][1] / total_probability,
        )
    topic_distributions[i] = topic_distribution

# Matching probabilities with winner, nominee, or no recognition tags
topic_probabilities_with_tags = []
for i in range(len(topic_distributions)):
    topic_probabilities_with_tags.append((topic_distributions[i], labels[i]))

print(topic_probabilities_with_tags)

import pyLDAvis
import pyLDAvis.gensim

vis = pyLDAvis.gensim.prepare(
    topic_model=lda_model, corpus=corpus, dictionary=dictionary
)
pyLDAvis.enable_notebook()
pyLDAvis.display(vis)


# Classiyfing new review based on topic probabilities with tags
def classify_new_review(review: str) -> list[float]:
    """Classify a new review based on topic probabilities with tags."""
    # Tokenize the review
    tokenized_review = word_tokenize(review.lower())

    # Create a dictionary and corpus
    dictionary = corpora.Dictionary([tokenized_review])
    corpus = [dictionary.doc2bow(tokenized_review)]

    # Extract the topic distribution for the new review
    topic_distribution = lda_model[corpus[0]]

    # Extract the probabilities for each topic for the new review
    topic_probability = []
    for topic in topic_probabilities_with_tags:
        topic_probability.append(topic[1])

    return topic_probability


# Example usage:
user_input = "This movie was to die for"
topic_dist = classify_new_review(user_input)
# print("Topic Distribution for the New Review:", topic_probabilities)
