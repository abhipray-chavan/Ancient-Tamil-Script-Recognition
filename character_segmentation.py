#!/usr/bin/env python3
"""
Character Segmentation and Recognition
Extracts individual characters from an image and recognizes them
"""

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle

class CharacterSegmentation:
    """
    Segments an image into individual characters and recognizes them
    """

    def __init__(self, model_path, character_names, use_ensemble=True, confidence_threshold=0.15, model_choice='ensemble'):
        """
        Initialize with model and character names

        Args:
            model_path: Path to trained model
            character_names: List of character class names
            use_ensemble: Whether to use ensemble model for better accuracy
            confidence_threshold: Minimum confidence to accept a prediction (0.0-1.0)
            model_choice: 'ensemble' or 'cnn' - which model to use for predictions
        """
        self.model = tf.keras.models.load_model(model_path)
        self.character_names = character_names
        self.use_ensemble = use_ensemble
        self.confidence_threshold = confidence_threshold
        self.ensemble_model = None
        self.model_choice = model_choice

        # Try to load ensemble model if requested
        if use_ensemble:
            try:
                from ensemble_predictor import EnsemblePredictor
                self.ensemble_model = EnsemblePredictor()
            except Exception:
                self.use_ensemble = False
    
    def find_contours(self, image):
        """
        Find character contours in image with preprocessing

        Args:
            image: Input image (grayscale)

        Returns:
            Tuple of (contours, binary_image)
        """
        # Denoise the image
        denoised = cv2.fastNlMeansDenoising(image, h=10)

        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

        # Use Otsu's thresholding for better automatic threshold selection
        # This works better than fixed threshold for different image types
        _, binary = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours, binary
    
    def extract_characters(self, image_path):
        """
        Extract individual characters from image

        Args:
            image_path: Path to image file

        Returns:
            List of character images and their bounding boxes
        """
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return []

        # Find contours and get binary image
        contours, binary = self.find_contours(img)

        # Extract characters
        characters = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate solidity (how filled the contour is)
            area = cv2.contourArea(contour)
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / hull_area if hull_area > 0 else 0

            # Calculate aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 0

            # Filter by multiple criteria:
            # 1. Size: 15x15 to 80x80 (more restrictive)
            # 2. Solidity: > 0.5 (real characters are filled, noise is sparse)
            # 3. Aspect ratio: 0.3 to 3.0 (characters aren't too elongated)
            # 4. Area: at least 150 pixels (avoid tiny noise)
            if (w > 15 and h > 15 and w < 80 and h < 80 and
                solidity > 0.5 and
                0.3 < aspect_ratio < 3.0 and
                area > 150):
                # Extract from binary image (preprocessed)
                char_img = binary[y:y+h, x:x+w]
                characters.append({
                    'image': char_img,
                    'bbox': (x, y, w, h),
                    'area': w * h,
                    'solidity': solidity
                })

        # Sort by x-coordinate (left to right)
        characters.sort(key=lambda c: c['bbox'][0])

        # Sanity check: if we extracted too many characters, it's likely noise
        # Real inscriptions typically have 5-100 characters per image
        # If we get more than 100, reject all as it's probably a photo or noise
        if len(characters) > 100:
            return []

        return characters
    
    def recognize_character(self, char_image):
        """
        Recognize a single character using ensemble or CNN model

        Args:
            char_image: Character image

        Returns:
            Tuple of (character_name, confidence, predictions, pred_idx)
        """
        # Resize to model input size
        char_resized = cv2.resize(char_image, (50, 50))

        # Ensure image is in correct format (0-255 range)
        if char_resized.dtype != np.uint8:
            char_resized = np.uint8(char_resized)

        # Normalize to 0-1 range
        char_norm = char_resized.astype(np.float32) / 255.0
        X = np.array([char_norm]).reshape(1, 50, 50, 1)

        # Use selected model
        if self.model_choice == 'ensemble' and self.use_ensemble and self.ensemble_model:
            try:
                result = self.ensemble_model.predict_single(X[0], use_ensemble=True)
                pred_idx = result['class_id']
                confidence = result['confidence']
                predictions = np.array(result.get('all_predictions', []))
                if len(predictions) == 0:
                    # Fallback if all_predictions is empty
                    predictions = self.ensemble_model.models['cnn'].predict(X, verbose=0)
            except Exception:
                # Fallback to CNN
                predictions = self.model.predict(X, verbose=0)
                confidence = np.max(predictions[0])
                pred_idx = np.argmax(predictions[0])
        else:
            # Use CNN model
            predictions = self.model.predict(X, verbose=0)
            confidence = np.max(predictions[0])
            pred_idx = np.argmax(predictions[0])

        if pred_idx < len(self.character_names):
            char_name = self.character_names[pred_idx]
        else:
            char_name = f"unknown_{pred_idx}"

        return char_name, confidence, predictions, pred_idx
    
    def recognize_all_characters(self, characters):
        """
        Recognize all extracted characters, filtering by confidence

        Args:
            characters: List of character images

        Returns:
            List of recognized characters with confidence
        """
        results = []
        skipped = 0

        for i, char_data in enumerate(characters):
            try:
                char_name, confidence, predictions, pred_idx = self.recognize_character(char_data['image'])

                # Apply confidence threshold filtering
                if confidence < self.confidence_threshold:
                    skipped += 1
                    continue

                results.append({
                    'index': i,
                    'character': char_name,
                    'confidence': float(confidence),
                    'predictions': predictions,
                    'bbox': char_data['bbox'],
                    'pred_idx': int(pred_idx)
                })
            except Exception:
                # Skip characters that fail to recognize
                skipped += 1
                continue

        return results
    
    def process_image(self, image_path):
        """
        Process complete image: extract and recognize characters

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with results
        """
        characters = self.extract_characters(image_path)

        if not characters:
            return {
                'status': 'failed',
                'message': 'No characters found in image',
                'characters': []
            }

        results = self.recognize_all_characters(characters)

        char_sequence = [r['character'] for r in results]
        confidences = [r['confidence'] for r in results]
        avg_confidence = np.mean(confidences) if confidences else 0

        # Sanity check: if average confidence is very low (< 20%),
        # it means the model is essentially guessing
        # This likely indicates the image is not a Tamil inscription
        if avg_confidence < 0.20 and len(results) > 0:
            # Check if we have any high-confidence predictions (> 50%)
            high_conf_results = [r for r in results if r['confidence'] > 0.50]
            if not high_conf_results:
                # No high-confidence predictions, reject all
                return {
                    'status': 'failed',
                    'message': f'Low confidence predictions (avg: {avg_confidence*100:.1f}%). Image may not be a Tamil inscription.',
                    'characters': [],
                    'num_characters': 0
                }

        return {
            'status': 'success',
            'character_sequence': char_sequence,
            'confidences': confidences,
            'average_confidence': float(avg_confidence),
            'results': results,
            'num_characters': len(results)
        }


# Example usage
if __name__ == "__main__":
    import json
    
    # Character names
    character_names = [
        'a', 'ai', 'c', 'e', 'i', 'k', 'l', 'l5', 'l5u', 'l5u4', 
        'n', 'n1', 'n1u4', 'n2', 'n2u4', 'n3', 'n5', 'o', 'p', 'pu4', 
        'r', 'r5', 'r5i', 'ru', 't', 'y'
    ]
    
    # Initialize segmentation
    segmenter = CharacterSegmentation('Model-Creation/CNN.model.keras', character_names)
    
    # Process sample image
    image_path = 'Input Images/Inscriptions - Wiki1/8.jpg'
    print(f"\n{'='*70}")
    print(f"Processing: {image_path}")
    print(f"{'='*70}")
    
    result = segmenter.process_image(image_path)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Status: {result['status']}")
    print(f"Characters found: {result['num_characters']}")
    print(f"Character sequence: {' → '.join(result['character_sequence'])}")
    print(f"Average confidence: {result['average_confidence']*100:.2f}%")
    print(f"\nDetailed results:")
    for r in result['results']:
        print(f"  [{r['index']+1}] {r['character']}: {r['confidence']*100:.2f}%")
    
    # Save results (convert numpy arrays to lists)
    result_to_save = result.copy()
    result_to_save['confidences'] = [float(c) for c in result_to_save['confidences']]
    for r in result_to_save['results']:
        r['predictions'] = r['predictions'].tolist()

    with open('segmentation_results.json', 'w', encoding='utf-8') as f:
        json.dump(result_to_save, f, indent=2)
    print(f"\n✓ Results saved to segmentation_results.json")

