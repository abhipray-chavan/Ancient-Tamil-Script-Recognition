#!/usr/bin/env python3
"""
Compare Current CNN Model vs ResNet50 Transfer Learning Model
Provides detailed performance metrics and recommendations
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

print("=" * 80)
print("MODEL COMPARISON: Current CNN vs ResNet50 Transfer Learning")
print("=" * 80)

# Load data
print("\n[1/4] Loading test data...")
try:
    with open('X.pickle', 'rb') as f:
        X = pickle.load(f)
    with open('y.pickle', 'rb') as f:
        y = pickle.load(f)
    
    y = np.array(y)
    print(f"✓ Loaded data: X shape {X.shape}, y length {len(y)}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

print(f"✓ Test set size: {X_test.shape}")

# Load Current CNN Model
print("\n[2/4] Loading Current CNN Model...")
try:
    current_model = tf.keras.models.load_model('CNN.model.keras')
    print("✓ Loaded CNN.model.keras")
except Exception as e:
    print(f"✗ Error loading CNN model: {e}")
    current_model = None

# Load ResNet50 Model
print("\n[3/4] Loading ResNet50 Model...")
try:
    resnet_model = tf.keras.models.load_model('ResNet50_Tamil.keras')
    print("✓ Loaded ResNet50_Tamil.keras")
except Exception as e:
    print(f"✗ Error loading ResNet50 model: {e}")
    resnet_model = None

# Evaluate models
print("\n[4/4] Evaluating models on test set...")

results = {
    "timestamp": datetime.now().isoformat(),
    "test_set_size": len(X_test),
    "models": {}
}

# Current CNN Model
if current_model:
    print("\nEvaluating Current CNN Model...")
    X_test_normalized = X_test / 255.0
    cnn_loss, cnn_accuracy = current_model.evaluate(X_test_normalized, y_test, verbose=0)
    cnn_predictions = current_model.predict(X_test_normalized, verbose=0)
    
    results["models"]["CNN"] = {
        "accuracy": float(cnn_accuracy),
        "loss": float(cnn_loss),
        "parameters": int(current_model.count_params()),
        "model_size_mb": os.path.getsize('CNN.model.keras') / (1024*1024)
    }
    print(f"  Accuracy: {cnn_accuracy*100:.2f}%")
    print(f"  Loss: {cnn_loss:.4f}")
    print(f"  Parameters: {current_model.count_params():,}")

# ResNet50 Model
if resnet_model:
    print("\nEvaluating ResNet50 Model...")
    # Resize test data for ResNet50
    IMG_SIZE = 224
    X_test_resized = np.zeros((X_test.shape[0], IMG_SIZE, IMG_SIZE, 3))
    
    for i in range(X_test.shape[0]):
        img = X_test[i, :, :, 0]
        img_resized = tf.image.resize(img[np.newaxis, :, :, np.newaxis], 
                                       (IMG_SIZE, IMG_SIZE))
        X_test_resized[i] = np.repeat(img_resized[0], 3, axis=-1)
    
    X_test_resized = X_test_resized / 255.0
    resnet_loss, resnet_accuracy = resnet_model.evaluate(X_test_resized, y_test, verbose=0)
    resnet_predictions = resnet_model.predict(X_test_resized, verbose=0)
    
    results["models"]["ResNet50"] = {
        "accuracy": float(resnet_accuracy),
        "loss": float(resnet_loss),
        "parameters": int(resnet_model.count_params()),
        "model_size_mb": os.path.getsize('ResNet50_Tamil.keras') / (1024*1024)
    }
    print(f"  Accuracy: {resnet_accuracy*100:.2f}%")
    print(f"  Loss: {resnet_loss:.4f}")
    print(f"  Parameters: {resnet_model.count_params():,}")

# Comparison Report
print("\n" + "=" * 80)
print("DETAILED COMPARISON REPORT")
print("=" * 80)

if current_model and resnet_model:
    print(f"\n{'Metric':<30} {'CNN':<20} {'ResNet50':<20} {'Improvement':<15}")
    print("-" * 85)
    
    cnn_acc = results["models"]["CNN"]["accuracy"]
    resnet_acc = results["models"]["ResNet50"]["accuracy"]
    acc_improvement = ((resnet_acc - cnn_acc) / cnn_acc) * 100
    
    print(f"{'Accuracy':<30} {cnn_acc*100:>18.2f}% {resnet_acc*100:>18.2f}% {acc_improvement:>13.1f}%")
    
    cnn_loss = results["models"]["CNN"]["loss"]
    resnet_loss = results["models"]["ResNet50"]["loss"]
    loss_improvement = ((cnn_loss - resnet_loss) / cnn_loss) * 100
    
    print(f"{'Loss':<30} {cnn_loss:>18.4f} {resnet_loss:>18.4f} {loss_improvement:>13.1f}%")
    
    cnn_params = results["models"]["CNN"]["parameters"]
    resnet_params = results["models"]["ResNet50"]["parameters"]
    param_increase = ((resnet_params - cnn_params) / cnn_params) * 100
    
    print(f"{'Parameters':<30} {cnn_params:>18,} {resnet_params:>18,} {param_increase:>13.1f}%")
    
    cnn_size = results["models"]["CNN"]["model_size_mb"]
    resnet_size = results["models"]["ResNet50"]["model_size_mb"]
    size_increase = ((resnet_size - cnn_size) / cnn_size) * 100
    
    print(f"{'Model Size (MB)':<30} {cnn_size:>18.2f} {resnet_size:>18.2f} {size_increase:>13.1f}%")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if resnet_acc > cnn_acc:
        improvement_pct = (resnet_acc - cnn_acc) * 100
        print(f"\n✓ ResNet50 shows {improvement_pct:.1f}% accuracy improvement over CNN")
        print("✓ Recommended for production use if accuracy is priority")
        print("✓ Trade-off: Larger model size and slower inference")
    else:
        print("\n✗ CNN model performs better than ResNet50")
        print("✗ Consider collecting more training data for ResNet50")
        print("✗ ResNet50 may need more epochs or different hyperparameters")
    
    print("\n" + "=" * 80)

# Save results
print("\nSaving comparison results...")
with open('model_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Saved to: model_comparison_results.json")

print("\n" + "=" * 80)
print("✓ COMPARISON COMPLETED")
print("=" * 80)

