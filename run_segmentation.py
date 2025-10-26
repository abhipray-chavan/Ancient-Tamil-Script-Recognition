#!/usr/bin/env python3
# Character Segmentation - Extract individual characters from preprocessed image

import cv2
import numpy as np
import imutils

print("Reading preprocessed image...")
image = cv2.imread("ImagePreProcessingFinal.jpg")
if image is None:
    print("Error: Could not read ImagePreProcessingFinal.jpg")
    print("Please run the preprocessing script first!")
    exit(1)

print(f"Image shape: {image.shape}")

print("Converting to grayscale and applying Gaussian blur...")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

print("Applying threshold...")
ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

print("Dilating the image...")
dilate = cv2.dilate(thresh1, None, iterations=2)

print("Finding contours...")
cnts, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(cnts)} contours")

# Filter out empty contours
cnts = [c for c in cnts if len(c) > 0]
print(f"Valid contours: {len(cnts)}")

if len(cnts) == 0:
    print("Warning: No valid contours found!")
    print("The image might be too clean or the threshold needs adjustment.")
    exit(1)

print("Sorting contours from left to right, top to bottom...")
sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * image.shape[1])

orig = image.copy()
i = 0
saved_count = 0

print("Extracting and saving character ROIs...")
for cnt in sorted_ctrs:
    # Check the area of contour, if it is very small ignore it
    area = cv2.contourArea(cnt)
    if area < 200:
        continue

    # Filtered contours are detected
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Taking ROI of the contour
    roi = image[y:y+h, x:x+w]
    
    # Mark them on the image if you want
    cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Save your contours or characters
    cv2.imwrite(f"Images/roi{i}.png", roi)
    saved_count += 1
    i = i + 1

print(f"✓ Saved {saved_count} character images to Images/ folder")

print("Saving image with bounding boxes...")
cv2.imwrite("box.jpg", orig)
print("✓ Saved bounding box visualization to box.jpg")

print("\n✓ Character segmentation completed successfully!")
print(f"✓ Total characters extracted: {saved_count}")

