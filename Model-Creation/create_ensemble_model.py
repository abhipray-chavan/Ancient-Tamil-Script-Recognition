#!/usr/bin/env python3
"""
Ensemble Model Combining Multiple Approaches
Uses voting and confidence-based selection for better accuracy
"""

import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter

print("=" * 80)
print("ENSEMBLE MODEL STRATEGY")
print("=" * 80)

# Load data
print("\n[1/5] Loading data...")
with open('X.pickle', 'rb') as f:
    X = pickle.load(f)
with open('y.pickle', 'rb') as f:
    y = pickle.load(f)

y = np.array(y)
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")

# Load models
print("\n[2/5] Loading available models...")
models = {}
try:
    models['cnn'] = tf.keras.models.load_model('CNN.model.keras')
    print("✓ Loaded CNN model")
except:
    print("✗ CNN model not found")

try:
    models['resnet50'] = tf.keras.models.load_model('ResNet50_Tamil.keras')
    print("✓ Loaded ResNet50 model")
except:
    print("✗ ResNet50 model not found")

try:
    models['improved'] = tf.keras.models.load_model('improved_model_best.keras')
    print("✓ Loaded Improved model")
except:
    print("✗ Improved model not found")

if len(models) < 2:
    print("\n⚠️  Need at least 2 models for ensemble. Using available models...")

# Evaluate individual models
print("\n[3/5] Evaluating individual models...")
results = {}

for name, model in models.items():
    print(f"\n  {name.upper()} Model:")
    
    # Prepare input based on model requirements
    if name == 'resnet50':
        # ResNet50 needs 224x224 RGB images
        X_test_resized = np.zeros((X_test.shape[0], 224, 224, 3))
        for i in range(X_test.shape[0]):
            img = X_test[i, :, :, 0]
            img_resized = tf.image.resize(img[np.newaxis, :, :, np.newaxis], (224, 224))
            X_test_resized[i] = np.repeat(img_resized[0], 3, axis=-1)
        test_data = X_test_resized
    else:
        test_data = X_test
    
    loss, acc = model.evaluate(test_data, y_test, verbose=0)
    predictions = model.predict(test_data, verbose=0)
    
    results[name] = {
        'accuracy': acc,
        'loss': loss,
        'predictions': predictions
    }
    
    print(f"    Accuracy: {acc*100:.2f}%")
    print(f"    Loss: {loss:.4f}")

# Ensemble voting
print("\n[4/5] Creating ensemble predictions...")
ensemble_predictions = np.zeros_like(results[list(models.keys())[0]]['predictions'])

for name in models.keys():
    ensemble_predictions += results[name]['predictions']

ensemble_predictions /= len(models)
ensemble_pred_classes = np.argmax(ensemble_predictions, axis=1)
ensemble_accuracy = np.mean(ensemble_pred_classes == y_test)

print(f"\n  Ensemble (Voting) Accuracy: {ensemble_accuracy*100:.2f}%")

# Confidence-based selection
print("\n[5/5] Confidence-based ensemble...")
confidence_predictions = np.zeros(len(y_test), dtype=int)

for i in range(len(y_test)):
    confidences = []
    predictions_list = []
    
    for name in models.keys():
        pred = np.argmax(results[name]['predictions'][i])
        conf = np.max(results[name]['predictions'][i])
        predictions_list.append(pred)
        confidences.append(conf)
    
    # Use prediction with highest confidence
    best_idx = np.argmax(confidences)
    confidence_predictions[i] = predictions_list[best_idx]

confidence_accuracy = np.mean(confidence_predictions == y_test)
print(f"  Confidence-based Accuracy: {confidence_accuracy*100:.2f}%")

# Summary
print("\n" + "=" * 80)
print("ENSEMBLE RESULTS SUMMARY")
print("=" * 80)

print("\nIndividual Model Accuracies:")
for name, result in results.items():
    print(f"  {name.upper():15s}: {result['accuracy']*100:6.2f}%")

print(f"\nEnsemble Methods:")
print(f"  Voting:          {ensemble_accuracy*100:6.2f}%")
print(f"  Confidence-based: {confidence_accuracy*100:6.2f}%")

# Find best approach
best_acc = max(ensemble_accuracy, confidence_accuracy)
best_method = "Voting" if ensemble_accuracy >= confidence_accuracy else "Confidence-based"

print(f"\n✓ Best Ensemble Method: {best_method} ({best_acc*100:.2f}%)")

# Improvement calculation
baseline = 0.3684
improvement = ((best_acc - baseline) / baseline) * 100
print(f"✓ Improvement over baseline: {improvement:+.1f}%")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print("""
1. ✅ Use Ensemble Method for better accuracy
2. ✅ Combine predictions from multiple models
3. ✅ Use confidence-based selection for robustness
4. ⚠️  Collect more training data (500+ samples per class)
5. ⚠️  Use data augmentation during training
6. ⚠️  Consider transfer learning with pre-trained models
""")

print("=" * 80)

