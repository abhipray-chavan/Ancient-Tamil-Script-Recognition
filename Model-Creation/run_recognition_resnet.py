#!/usr/bin/env python3
"""
ResNet50 with Transfer Learning for Tamil Character Recognition
Improves upon the basic CNN model by leveraging pre-trained ImageNet features
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

print("=" * 70)
print("ResNet50 Transfer Learning for Tamil Character Recognition")
print("=" * 70)

# Load training data
print("\n[1/6] Loading training data...")
try:
    with open('X.pickle', 'rb') as f:
        X = pickle.load(f)
    with open('y.pickle', 'rb') as f:
        y = pickle.load(f)
    
    y = np.array(y)
    print(f"✓ Loaded X shape: {X.shape}")
    print(f"✓ Loaded y length: {len(y)}")
    print(f"✓ Number of classes: {len(np.unique(y))}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

# Normalize and resize images to 224x224 (ResNet50 requirement)
print("\n[2/6] Preprocessing images for ResNet50...")
IMG_SIZE = 224
X_resized = np.zeros((X.shape[0], IMG_SIZE, IMG_SIZE, 3))

for i in range(X.shape[0]):
    # Resize from 50x50 to 224x224
    img = X[i, :, :, 0]  # Get single channel
    img_resized = tf.image.resize(img[np.newaxis, :, :, np.newaxis], 
                                   (IMG_SIZE, IMG_SIZE))
    # Convert grayscale to RGB by repeating channels
    X_resized[i] = np.repeat(img_resized[0], 3, axis=-1)

# Normalize to [0, 1]
X_resized = X_resized / 255.0
print(f"✓ Resized images to: {X_resized.shape}")

# Split data into train and validation
print("\n[3/6] Splitting data (80% train, 20% validation)...")
from sklearn.model_selection import train_test_split
# Use stratify=None for small datasets with imbalanced classes
X_train, X_val, y_train, y_val = train_test_split(
    X_resized, y, test_size=0.2, random_state=42, stratify=None
)
print(f"✓ Training set: {X_train.shape}")
print(f"✓ Validation set: {X_val.shape}")

# Data augmentation to prevent overfitting
print("\n[4/6] Building ResNet50 model with transfer learning...")
num_classes = len(np.unique(y))

# Load pre-trained ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, 
                      input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze base model layers
base_model.trainable = False
print(f"✓ Loaded pre-trained ResNet50")
print(f"✓ Frozen {len(base_model.layers)} base layers")

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✓ Model compiled successfully")
print(f"✓ Total parameters: {model.count_params():,}")

# Model summary
print("\nModel Architecture:")
model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

# Train model
print("\n[5/6] Training ResNet50 model...")
print("Note: First epoch may be slower due to model compilation")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=8,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\n✓ Training completed!")

# Evaluate on validation set
print("\n[6/6] Evaluating model...")
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)

print(f"\n{'='*70}")
print("TRAINING RESULTS")
print(f"{'='*70}")
print(f"Training Accuracy:   {train_accuracy*100:.2f}%")
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"Training Loss:       {train_loss:.4f}")
print(f"Validation Loss:     {val_loss:.4f}")
print(f"{'='*70}")

# Save model
print("\nSaving model...")
model.save('ResNet50_Tamil.keras')
print("✓ Saved model to: ResNet50_Tamil.keras")

# Save training history
print("Saving training history...")
np.save('resnet_history.npy', history.history)
print("✓ Saved history to: resnet_history.npy")

# Plot training history
print("Plotting training history...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Model Accuracy (ResNet50)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_title('Model Loss (ResNet50)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resnet_training_history.png', dpi=100, bbox_inches='tight')
print("✓ Saved plot to: resnet_training_history.png")

print("\n" + "="*70)
print("✓ ResNet50 MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nNext Steps:")
print("1. Compare with current CNN model using: python3 compare_models.py")
print("2. Use ResNet50 for predictions: python3 run_recognition_resnet_predict.py")
print("3. Deploy to production when satisfied with accuracy")

