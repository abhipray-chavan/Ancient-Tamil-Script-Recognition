#!/usr/bin/env python3
"""
Improved CNN Model with Data Augmentation and Better Architecture
Targets: 60-70% accuracy improvement over baseline (36.84%)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, 
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split

print("=" * 80)
print("IMPROVED CNN MODEL TRAINING WITH DATA AUGMENTATION")
print("=" * 80)

# Load data
print("\n[1/6] Loading training data...")
with open('X.pickle', 'rb') as f:
    X = pickle.load(f)
with open('y.pickle', 'rb') as f:
    y = pickle.load(f)

y = np.array(y)
num_classes = len(np.unique(y))

print(f"✓ Loaded data: X shape {X.shape}, y length {len(y)}")
print(f"✓ Number of classes: {num_classes}")

# Normalize images
print("\n[2/6] Preprocessing images...")
X = X / 255.0
print(f"✓ Normalized images to [0, 1]")

# Split data (without stratify due to small dataset)
print("\n[3/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✓ Training set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")

# Data augmentation
print("\n[4/6] Setting up data augmentation...")
train_datagen = ImageDataGenerator(
    rotation_range=20,           # Random rotation
    width_shift_range=0.1,       # Random horizontal shift
    height_shift_range=0.1,      # Random vertical shift
    zoom_range=0.15,             # Random zoom
    shear_range=0.1,             # Shear transformation
    brightness_range=[0.8, 1.2], # Brightness adjustment
    fill_mode='nearest'
)
print("✓ Data augmentation configured:")
print("  - Rotation: ±20°")
print("  - Shift: ±10%")
print("  - Zoom: ±15%")
print("  - Brightness: 80-120%")

# Build improved model
print("\n[5/6] Building improved CNN model...")
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(32, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Block 2
    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Block 3
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Global pooling and dense layers
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✓ Model built successfully")
print(f"✓ Total parameters: {model.count_params():,}")
print(f"\nModel Architecture:")
model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)
checkpoint = ModelCheckpoint(
    'improved_model_best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train model with augmentation
print("\n[6/6] Training model with data augmentation...")
print("Note: This may take 2-5 minutes...")

history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=8),
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# Evaluate
print("\n" + "=" * 80)
print("TRAINING RESULTS")
print("=" * 80)
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"Training Accuracy:   {train_acc*100:.2f}%")
print(f"Test Accuracy:       {test_acc*100:.2f}%")
print(f"Training Loss:       {train_loss:.4f}")
print(f"Test Loss:           {test_loss:.4f}")

# Calculate improvement
baseline_acc = 0.3684
improvement = ((test_acc - baseline_acc) / baseline_acc) * 100
print(f"\nImprovement over baseline: {improvement:+.1f}%")

# Save model
print("\nSaving model...")
model.save('improved_cnn_model.keras')
print("✓ Saved to: improved_cnn_model.keras")

# Save history
np.save('improved_training_history.npy', history.history)
print("✓ Saved history to: improved_training_history.npy")

# Plot results
print("\nPlotting training history...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Model Accuracy (Improved CNN)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_title('Model Loss (Improved CNN)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('improved_training_history.png', dpi=100, bbox_inches='tight')
print("✓ Saved plot to: improved_training_history.png")

print("\n" + "=" * 80)
print("✓ IMPROVED MODEL TRAINING COMPLETED!")
print("=" * 80)

