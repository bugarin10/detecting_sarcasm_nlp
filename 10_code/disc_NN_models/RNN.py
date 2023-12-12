# %% [markdown]
# ## Recurrent Neural Network (RNN) Model 

# %%
# Install tensorflow
%pip install tensorflow_datasets==4.9.2
%pip install tensorflow

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall

tfds.disable_progress_bar()

# %% [markdown]
# ### Data Preparation

# %%
# Define what data to use for model
data_type = 'REAL'

# %%
# Import and clean Real or Synthetic data

if data_type == 'REAL':
    data = pd.read_csv("../../00_data/final_data.csv")
    df = pd.DataFrame(data)
    df["top_critics"] = df["top_critics"].str.split(r"\., |\. ,")
    df = df.explode("top_critics").reset_index(drop=True)
    df['winner'] = df['winner'].apply(lambda x: 1 if x > 0 else 0)
    reviews = df['top_critics'].astype(str)
    tags = df['winner'].astype(int)

elif data_type == 'SYNTHETIC':
    data = pd.read_csv("../../00_data/generated_reviews.csv")
    df = pd.DataFrame(data)
    df['winner'] = df['winner'].apply(lambda x: 1 if x > 0 else 0)
    reviews = df['generated_reviews']
    tags = df['winner'].astype(int)



# %%
# Transform data to tensorflow compatible datasets
combined_data = list(zip(reviews, tags))
def generator():
    for review, tag in combined_data:
        yield review, tag
output_types = (tf.string, tf.int32)
output_shapes = (tf.TensorShape([]), tf.TensorShape([]))
new_dataset = tf.data.Dataset.from_generator(
    generator, 
    output_types=output_types, 
    output_shapes=output_shapes
)

# %%
# Test one sample review
for example, label in new_dataset.take(1):
  print('text: ', example.numpy())
  print('label: ', label.numpy())

# %%
# Verify the TensorSpecs
new_dataset.element_spec

# %%
# Determine the total number of elements in the dataset
DATASET_SIZE = sum(1 for _ in new_dataset)

# Calculate split sizes for 90/10 train/test split
TRAIN_SIZE = int(0.9 * DATASET_SIZE)
TEST_SIZE = DATASET_SIZE - TRAIN_SIZE

# Shuffle the dataset
SHUFFLE_BUFFER_SIZE = DATASET_SIZE  # Adjust this as needed for your dataset size and memory constraints
shuffled_dataset = new_dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

# Create train dataset
new_train_dataset = shuffled_dataset.take(TRAIN_SIZE)

# Create test dataset
new_test_dataset = shuffled_dataset.skip(TRAIN_SIZE)

# %%
# Count the number of elements in the train dataset
train_dataset_size = sum(1 for _ in new_train_dataset)

# Count the number of elements in the test dataset
test_dataset_size = sum(1 for _ in new_test_dataset)

# Print number of elements in each dataset
print(f"Number of elements in the train dataset: {train_dataset_size}")
print(f"Number of elements in the test dataset: {test_dataset_size}")

# %%
# Establish buffer and batch sizes 
BUFFER_SIZE = 10000
BATCH_SIZE = 64

# %%
# Pre-batch the training and test datasets
train_dataset = new_train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = new_test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# %% [markdown]
# ### Establish Model Parameters and Run RNN Model

# %%
# Define vocab size and encoder function

VOCAB_SIZE = 10000

encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

vocab = np.array(encoder.get_vocabulary())

# %%
# Define RNN model layers
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

# %%
# Compile model with Binary Cross Entropy Loss Function
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

# %%
# Prepare plot
def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

# %%
# Fit model and store the loss and accuracy for plotting
history = model.fit(train_dataset,
                    epochs=25,
                    validation_data=test_dataset,
                    validation_steps=30)

# %%
# Print Loss and Accuracy
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# %%
# Plot Loss and Accuracy
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.title("Accuracy vs Epochs - Synthetic Data")
plot_graphs(history, 'accuracy')
plt.subplot(1, 2, 2)
plt.title("Loss vs Epochs - Synthetic Data")
plot_graphs(history, 'loss')

# %% [markdown]
# ### Generate Confusion Matrix, Precision, Recall, and F1 Score

# %%
from sklearn.metrics import classification_report
import numpy as np

# Extract test data from Tensor Object
predictions = []
labels = []
for new_inputs, new_labels in test_dataset:
    results = model(new_inputs)
    predictions.append(results)
    labels.append(new_labels)

# Initialize an empty list to store the floats
predicted_floats = []

# Iterate through each prediction tensor
for tensor in predictions:
    
    # Convert the tensor to a numpy array and flatten it
    numpy_array = tensor.numpy().flatten()
    
    # Extend the all_floats list with the elements from the numpy array
    predicted_floats.extend(numpy_array.tolist())

# Convert each tensor to a numpy array and concatenate them
flattened_array = np.concatenate([tensor.numpy() for tensor in labels])

# Convert the numpy array of labels to a list
labels_list = flattened_array.tolist()

# Apply sigmoid function to predicted probabilities
probabilities = 1 / (1 + np.exp(-np.array(predicted_floats)))

# Apply threshold to logit transformed probabilities
classification = [1 if prob >= 0.5 else 0 for prob in probabilities]

print("Neural Network Model - Confustion Matrix")
print(classification_report(labels_list, classification))


