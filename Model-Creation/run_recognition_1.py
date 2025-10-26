#!/usr/bin/env python3
# Recognition_1.ipynb - Prepare training data and create pickle files

import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle

file_list = []
class_list = []
IMG_SIZE = 50  # The size of images the network will work on

# All the categories that should be detect in neural network
CATEGORIES = []
ignore_files = ['1 - Multipart', '2 - Unknown']  # This files/folder should not include in neural network
DATADIR = r'../Labels/Labelled Dataset - Fig 51'

print(f"Loading categories from: {DATADIR}")
for directoryfile in os.listdir(DATADIR):
    if directoryfile in ignore_files:
        continue
    CATEGORIES.append(directoryfile)  # Append all the character's name as label in 'CATEGORIES'

print(f"Number of categories: {len(CATEGORIES)}")
print(f"Categories: {CATEGORIES}")

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)  # Character's name label as class number
        print(f"Processing category: {category} (class {class_num})")
        count = 0
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Converting image into grayscale
                if img_array is None:
                    print(f"Warning: Could not read {os.path.join(path, img)}")
                    continue
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # Resizing all images into same size (50,50)
                training_data.append([new_array, class_num])  # append all images with their respective class name
                count += 1
            except Exception as e:
                print(f"Error processing {path}/{img}: {e}")
                pass
        print(f"  Loaded {count} images for category {category}")

print("\nCreating training data...")
create_training_data()
print(f"Total training samples: {len(training_data)}")

# Storing features in X and labels in Y using numpy
print("\nShuffling and preparing data...")
random.shuffle(training_data)
X = []  # features
y = []  # labels

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print(f"X shape: {X.shape}")
print(f"y length: {len(y)}")

# Saving features and label in pickle files
print("\nSaving pickle files...")
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print("✓ Successfully created X.pickle and y.pickle files!")

