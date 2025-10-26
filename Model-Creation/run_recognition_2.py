#!/usr/bin/env python3
# Recognition_2.ipynb - Train CNN model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

print("Loading pickle files...")
# Loading pickle files
file_path_X = open(os.path.join(r"X.pickle"), 'rb')
file_path_Y = open(os.path.join(r"y.pickle"), 'rb')
X = pickle.load(file_path_X)
y = pickle.load(file_path_Y)
y = np.array(y)  # Convert to numpy array

print(f"X shape: {X.shape}")
print(f"y length: {len(y)}")

number_of_classes = max(y) + 1  # Number of classes
print(f"Number of classes: {number_of_classes}")

# Normalising the images
X = X / 255.0

print("\nBuilding CNN model...")
# Building the model
model = Sequential()

# First convolutional layer
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolutional layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output
model.add(Flatten())

# First dense layer
model.add(Dense(64))
model.add(Activation("relu"))

# Output layer
model.add(Dense(number_of_classes))
model.add(Activation("softmax"))

# Compile the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

print("\nModel summary:")
model.summary()

print("\nTraining the model...")
# Train the model
history = model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1, verbose=1)

print("\nSaving the model...")
# Saving the model in local machine
model.save_weights("model.weights.h5")
print("✓ Saved model weights to model.weights.h5")

# Save in Keras format
model.save('CNN.model.keras')
print("✓ Saved complete model to CNN.model.keras")

# Create a symbolic link for backward compatibility
import shutil
if os.path.exists('CNN.model'):
    if os.path.isfile('CNN.model'):
        os.remove('CNN.model')
    elif os.path.isdir('CNN.model'):
        shutil.rmtree('CNN.model')
shutil.copy('CNN.model.keras', 'CNN.model')
print("✓ Created CNN.model for backward compatibility")

# Plot training history
print("\nPlotting training history...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
print("✓ Saved training history plot to training_history.png")

print("\n✓ Model training completed successfully!")

