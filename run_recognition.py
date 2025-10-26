#!/usr/bin/env python3
# Character Recognition - Predict characters from segmented images

import os
import cv2
import tensorflow as tf
import glob

print("Loading categories...")
# Append all the categories we want to read
CATEGORIES = []
files = ['1 - Multipart', '2 - Unknown']
DATADIR = r'Labels/Labelled Dataset - Fig 51'

for directoryfile in os.listdir(DATADIR):
    if directoryfile in files:
        continue
    CATEGORIES.append(directoryfile)

print(f"Number of categories: {len(CATEGORIES)}")
print(f"Categories: {CATEGORIES}")

# The function prepare(file) allows us to use an image of any size
def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if img_array is None:
        return None
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("\nLoading trained model...")
# Loading pre-trained model from local machine
try:
    model = tf.keras.models.load_model("Model-Creation/CNN.model.keras")
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying alternative model path...")
    try:
        model = tf.keras.models.load_model("Model-Creation/CNN.model")
        print("✓ Model loaded successfully!")
    except Exception as e2:
        print(f"Error: Could not load model: {e2}")
        exit(1)

print("\nProcessing segmented characters...")
# Get all character images
image_files = sorted(glob.glob("Images/roi*.png"))
print(f"Found {len(image_files)} character images")

if len(image_files) == 0:
    print("Error: No character images found in Images/ folder")
    print("Please run the segmentation script first!")
    exit(1)

# Process each character
results = []
print("\nPredicting characters:")
print("-" * 60)

for img_file in image_files[:20]:  # Process first 20 characters as a sample
    image = prepare(img_file)
    if image is None:
        print(f"Warning: Could not read {img_file}")
        continue
    
    # Predict
    prediction = model.predict([image], verbose=0)
    prediction = list(prediction[0])
    
    # Get the predicted character
    predicted_class = prediction.index(max(prediction))
    predicted_char = CATEGORIES[predicted_class]
    confidence = max(prediction) * 100
    
    results.append({
        'file': os.path.basename(img_file),
        'character': predicted_char,
        'confidence': confidence
    })
    
    print(f"{os.path.basename(img_file):15s} -> {predicted_char:10s} (confidence: {confidence:.2f}%)")

print("-" * 60)
print(f"\n✓ Processed {len(results)} characters")

# Save results to file
print("\nSaving results to recognition_results.txt...")
with open("recognition_results.txt", "w", encoding="utf-8") as f:
    f.write("Character Recognition Results\n")
    f.write("=" * 60 + "\n\n")
    for result in results:
        f.write(f"{result['file']:15s} -> {result['character']:10s} (confidence: {result['confidence']:.2f}%)\n")
    f.write("\n" + "=" * 60 + "\n")
    f.write(f"Total characters processed: {len(results)}\n")

print("✓ Results saved to recognition_results.txt")

# Create a summary
print("\n" + "=" * 60)
print("RECOGNITION SUMMARY")
print("=" * 60)
char_counts = {}
for result in results:
    char = result['character']
    char_counts[char] = char_counts.get(char, 0) + 1

print("\nCharacter frequency:")
for char, count in sorted(char_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {char:10s}: {count} occurrences")

print("\n✓ Character recognition completed successfully!")

