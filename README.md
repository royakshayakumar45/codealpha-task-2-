# codealpha-task-2-
new repo
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# Step 1: Load Dataset (for simplicity, using MNIST dataset for handwritten digits)
# Extend to alphabets using datasets like EMNIST for handwritten characters
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Step 2: Data Preparation
# Reshape the data to fit the model input format (28x28x1 for grayscale images)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 3: Model Development
# Define a Convolutional Neural Network (CNN) model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Step 4: Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=1)

# Step 6: Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))

# Step 7: Save the Model
model.save('handwritten_character_recognition.h5')

# Step 8: Extend for Handwritten Characters/Words
# To recognize characters/words, use EMNIST or similar datasets and preprocess them accordingly.
# Add techniques like sequence-to-sequence models (e.g., RNNs or Transformers) for full sentences.
