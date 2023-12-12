import pandas as pd
import random
from collections import Counter
from torchtext.data.utils import get_tokenizer

# Read the data
data = pd.read_csv("../../00_data/final_data.csv")

df = pd.DataFrame(data)

# Splitting the 'top_critics' column into separate rows
df["top_critics"] = df["top_critics"].str.split(r"\., |\. ,")
df = df.explode("top_critics").reset_index(drop=True)


tokenizer = get_tokenizer("basic_english")
data["top_critics"] = data["top_critics"].apply(lambda x: tokenizer(x))

# Separate into nominees and no recognition categories
nominees = data[data["winner"] == 2 | (data["winner"] == 1)]
no_recognition = data[data["winner"] == 0]

min_review_length_nominee = min(nominees["top_critics"].apply(len))
min_review_length_no_recognition = min(no_recognition["top_critics"].apply(len))
max_review_length_nominee = max(nominees["top_critics"].apply(len))
max_review_length_no_recognition = max(no_recognition["top_critics"].apply(len))

# Count word sequences for each category
nominee_tokens = Counter(
    word for sublist in nominees["top_critics"] for word in sublist
)
no_recognition_tokens = Counter(
    word for sublist in no_recognition["top_critics"] for word in sublist
)

# Probability of nominee word sequences
nominee_total = sum(nominee_tokens.values())
for key in nominee_tokens:
    nominee_tokens[key] /= nominee_total

# Probability of no recognition word sequences
no_recognition_total = sum(no_recognition_tokens.values())
for key in no_recognition_tokens:
    no_recognition_tokens[key] /= no_recognition_total

# Generate synthetic data based on word sequence probabilities
nominee_data_len = nominees.shape[0]
no_recognition_data_len = no_recognition.shape[0]

rewarded_synth_data = []
for _ in range(nominee_data_len):
    review_list_len = random.randint(
        min_review_length_nominee, max_review_length_nominee
    )
    review = []
    for i in range(review_list_len):
        # Randomly sample word sequences based on probabilities for nominees
        word = random.choices(
            list(nominee_tokens.keys()), weights=list(nominee_tokens.values())
        )
        review.append(word)
    rewarded_synth_data.append(review)

# unrewarded_synth_data = []
# for _ in range(no_recognition_data_len):
#     review_list_len = random.randint(
#         min_review_length_no_recognition, max_review_length_no_recognition
#     )
#     review = []
#     for i in range(review_list_len):
#         # Randomly sample word sequences based on probabilities for nominees
#         word = random.choices(
#             list(no_recognition_tokens.keys()),
#             weights=list(no_recognition_tokens.values()),
#         )
#         review.append(word)
#     unrewarded_synth_data.append(review)

print(rewarded_synth_data)
# print(unrewarded_synth_data)


log_probs = clf.feature_log_prob_
# exponentiate with numpy = actual_probs
feature_names = vectorizer.get_feature_names_out()

df_probs = pd.DataFrame(actual_probs.T, columns=["nominee", "no_recognition"])
df_probs["feature_names"] = feature_names
df_log_probs = pd.DataFrame(log_probs.T, columns=["nominee", "no_recognition"])
df_log_probs["feature_names"] = feature_names
