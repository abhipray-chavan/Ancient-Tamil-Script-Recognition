#!/usr/bin/env python3
"""
Streamlit Web Application for Ancient Tamil Script Recognition
Full end-to-end pipeline with single and multi-character recognition
"""

# Suppress TensorFlow warnings and info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import sys
from contextlib import contextmanager
from io import StringIO

sys.path.insert(0, '.')
from tamil_sentence_generator import TamilSentenceGenerator
from character_segmentation import CharacterSegmentation
from tamil_english_translator import TamilEnglishTranslator
from ensemble_predictor import EnsemblePredictor

# Context manager to suppress stdout/stderr
@contextmanager
def suppress_output():
    """Suppress stdout and stderr"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# Page configuration
st.set_page_config(
    page_title="Tamil Script Recognition",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .success-box { background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #28a745; }
    .error-box { background-color: #f8d7da; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #dc3545; }
    .result-box { background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0; }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_info = {}

    with suppress_output():
        # Ensemble Model (Best Performance)
        try:
            ensemble_obj = EnsemblePredictor()
            models['ensemble'] = ensemble_obj
            model_info['ensemble'] = {
                'name': 'Ensemble Voting ⭐ BEST',
                'accuracy': '42.11%',
                'type': 'Ensemble (Voting)',
                'description': 'Combines CNN, ResNet50, and Improved CNN using voting mechanism. Best overall accuracy (+14.3% improvement over baseline)'
            }
        except Exception:
            pass

        # CNN Model (Baseline)
        try:
            models['cnn'] = tf.keras.models.load_model('Model-Creation/CNN.model.keras')
            model_info['cnn'] = {
                'name': 'CNN (Baseline)',
                'accuracy': '36.84%',
                'type': 'Baseline CNN',
                'description': 'Original CNN model trained on 95 samples'
            }
        except:
            pass

        # ResNet50 Model (Transfer Learning)
        try:
            models['resnet50'] = tf.keras.models.load_model('Model-Creation/ResNet50_Tamil.keras')
            model_info['resnet50'] = {
                'name': 'ResNet50 (Transfer Learning)',
                'accuracy': '10.53%',
                'type': 'Transfer Learning (Experimental)',
                'description': 'ResNet50 pre-trained on ImageNet, fine-tuned for Tamil. Note: Lower accuracy due to limited training data (95 samples). Better with 500+ samples.'
            }
        except:
            pass

        # Improved CNN Model
        try:
            models['improved_cnn'] = tf.keras.models.load_model('Model-Creation/improved_cnn_model.keras')
            model_info['improved_cnn'] = {
                'name': 'Improved CNN',
                'accuracy': '10.53%',
                'type': 'Enhanced Architecture (Experimental)',
                'description': 'CNN with batch normalization and data augmentation. Note: Lower accuracy due to limited training data (95 samples). Better with 500+ samples.'
            }
        except:
            pass

        # Best Improved Model
        try:
            models['improved_best'] = tf.keras.models.load_model('Model-Creation/improved_model_best.keras')
            model_info['improved_best'] = {
                'name': 'Improved Model (Best)',
                'accuracy': '10.53%',
                'type': 'Enhanced Architecture (Experimental)',
                'description': 'Best performing improved model with optimizations. Note: Lower accuracy due to limited training data (95 samples). Better with 500+ samples.'
            }
        except:
            pass

    return models, model_info

@st.cache_resource
def load_character_classes():
    """Load character class names"""
    try:
        character_names = [
            'a', 'ai', 'c', 'e', 'i', 'k', 'l', 'l5', 'l5u', 'l5u4',
            'n', 'n1', 'n1u4', 'n2', 'n2u4', 'n3', 'n5', 'o', 'p', 'pu4',
            'r', 'r5', 'r5i', 'ru', 't', 'y'
        ]
        return character_names
    except:
        return None

@st.cache_resource
def load_character_mapping():
    """Load character name to Tamil Unicode mapping"""
    return {
        'a': 'அ', 'ai': 'ஐ', 'c': 'ச', 'e': 'எ', 'i': 'இ', 'k': 'க',
        'l': 'ல', 'l5': 'ள', 'l5u': 'ளு', 'l5u4': 'ளு',
        'n': 'ண', 'n1': 'ன', 'n1u4': 'னு', 'n2': 'ந', 'n2u4': 'நு', 'n3': 'ண', 'n5': 'ன',
        'o': 'ஒ', 'p': 'ப', 'pu4': 'பு',
        'r': 'ர', 'r5': 'ற', 'r5i': 'றி', 'ru': 'ரு', 't': 'ட', 'y': 'ய'
    }

@st.cache_resource
def load_sentence_generator():
    try:
        return TamilSentenceGenerator()
    except Exception as e:
        st.warning(f"Could not load sentence generator: {e}")
        return None

@st.cache_resource
def load_character_segmentation():
    try:
        character_names = [
            'a', 'ai', 'c', 'e', 'i', 'k', 'l', 'l5', 'l5u', 'l5u4',
            'n', 'n1', 'n1u4', 'n2', 'n2u4', 'n3', 'n5', 'o', 'p', 'pu4',
            'r', 'r5', 'r5i', 'ru', 't', 'y'
        ]
        # Use ensemble model with 15% confidence threshold (can be adjusted by user)
        return CharacterSegmentation(
            'Model-Creation/CNN.model.keras',
            character_names,
            use_ensemble=True,
            confidence_threshold=0.15
        )
    except Exception as e:
        st.warning(f"Could not load character segmentation: {e}")
        return None

@st.cache_resource
def load_translator():
    try:
        return TamilEnglishTranslator()
    except Exception as e:
        st.warning(f"Could not load translator: {e}")
        return None

# Load resources
models, model_info = load_models()
character_classes = load_character_classes()
character_mapping = load_character_mapping()
sentence_generator = load_sentence_generator()
segmentation = load_character_segmentation()
translator = load_translator()

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'segmentation_results' not in st.session_state:
    st.session_state.segmentation_results = None

# Title
st.title("🏛️ Ancient Tamil Script Recognition System")
st.markdown("---")

# Sidebar
st.sidebar.title("Configuration")

# Model selection with descriptions
st.sidebar.subheader("🤖 Model Selection")
available_models = list(model_info.keys())
model_labels = [f"{model_info[m]['name']} ({model_info[m]['accuracy']})" for m in available_models]
model_choice_idx = st.sidebar.selectbox(
    "Select Model:",
    range(len(available_models)),
    format_func=lambda x: model_labels[x],
    help="Choose which model to use for character recognition"
)
model_choice = available_models[model_choice_idx]

# Display selected model info
with st.sidebar.expander("📊 Model Details", expanded=False):
    info = model_info[model_choice]
    st.write(f"**Name:** {info['name']}")
    st.write(f"**Type:** {info['type']}")
    st.write(f"**Accuracy:** {info['accuracy']}")
    st.write(f"**Description:** {info['description']}")

confidence_threshold = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.3)

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Single Character", "Full Text (Multi-Char)", "Results", "Translation", "Literal Meaning", "Model Info", "Help"])

# TAB 1: Single Character Recognition
with tab1:
    st.header("Single Character Recognition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"], key="single")
    
    with col2:
        st.subheader("Or Use Sample")
        use_sample = st.checkbox("Use sample image", key="single_sample")
        sample_path = None
        if use_sample:
            for dir_path in ["Input Images/Inscriptions - Wiki1", "Input Images"]:
                if os.path.exists(dir_path):
                    images = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        sample_path = os.path.join(dir_path, images[0])
                        st.info(f"Using: {images[0]}")
                        break
    
    if st.button("Process Single Character"):
        image_path = None
        if uploaded_file:
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_path = "temp_image.jpg"
        elif use_sample and sample_path:
            image_path = sample_path
        else:
            st.error("Please upload or select sample image")
        
        if image_path:
            try:
                with st.spinner(f"Processing with {model_info[model_choice]['name']}..."):
                    model = models.get(model_choice)
                    if not model:
                        st.error(f"{model_info[model_choice]['name']} not loaded")
                    else:
                        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                        # Preprocess image to match training data
                        img_denoised = cv2.fastNlMeansDenoising(img, h=10)
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                        img_morph = cv2.morphologyEx(img_denoised, cv2.MORPH_CLOSE, kernel)
                        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel)

                        img_resized = cv2.resize(img_morph, (50, 50))
                        img_norm = img_resized / 255.0
                        X = np.array([img_norm]).reshape(1, 50, 50, 1)

                        # Handle ensemble model differently
                        predictions = None

                        # Suppress all output during model prediction
                        with suppress_output():
                            if model_choice == 'ensemble':
                                try:
                                    result = model.predict_single(X[0], use_ensemble=True)
                                    pred_idx = result['class_id']
                                    confidence = result['confidence']
                                    all_preds = result.get('all_predictions', None)
                                    if all_preds:
                                        predictions = np.array(all_preds)
                                    else:
                                        # Fallback if all_predictions is empty
                                        predictions = model.models['cnn'].predict(X, verbose=0)
                                except Exception:
                                    predictions = model.models['cnn'].predict(X, verbose=0)
                                    confidence = np.max(predictions[0])
                                    pred_idx = np.argmax(predictions[0])
                            else:
                                predictions = model.predict(X, verbose=0)
                                confidence = np.max(predictions[0])
                                pred_idx = np.argmax(predictions[0])

                        # Ensure predictions is set
                        if predictions is None:
                            st.error("Failed to get predictions")
                        else:
                            pred_class = character_classes[pred_idx]
                            tamil_char = character_mapping.get(pred_class, pred_class)

                            st.session_state.results = {
                                'character': pred_class,
                                'tamil_character': tamil_char,
                                'confidence': confidence,
                                'predictions': predictions[0] if isinstance(predictions, np.ndarray) and len(predictions.shape) > 1 else predictions,
                                'model': model_info[model_choice]['name'],
                                'model_key': model_choice,
                                'image_path': image_path,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }

                            st.success(f"Recognized: **{tamil_char}** ({confidence*100:.2f}%)")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# TAB 2: Multi-Character Recognition
with tab2:
    st.header("Full Text Recognition (Multi-Character)")
    st.info("This mode extracts and recognizes ALL characters in an image to generate a complete sentence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file_multi = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"], key="multi")
    
    with col2:
        st.subheader("Or Use Sample")
        use_sample_multi = st.checkbox("Use sample image", key="multi_sample")
        sample_path_multi = None
        if use_sample_multi:
            for dir_path in ["Input Images/Inscriptions - Wiki1", "Input Images"]:
                if os.path.exists(dir_path):
                    images = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        sample_path_multi = os.path.join(dir_path, images[0])
                        st.info(f"Using: {images[0]}")
                        break
    
    # Add confidence threshold control for multi-character
    st.subheader("Recognition Settings")

    col1, col2 = st.columns(2)
    with col1:
        multi_model_choice = st.selectbox(
            "Model for Multi-Character Recognition:",
            ['cnn', 'ensemble'],
            format_func=lambda x: 'CNN (Faster, Baseline)' if x == 'cnn' else 'Ensemble (Slower, Better)',
            key='multi_model_choice'
        )

    with col2:
        multi_confidence_threshold = st.slider(
            "Confidence Threshold:",
            0.0, 1.0, 0.15,
            step=0.05,
            help="Only accept character predictions with confidence above this threshold. Higher = fewer but more reliable characters. Recommended: 0.10-0.30"
        )

    if st.button("Extract and Recognize All Characters"):
        image_path_multi = None
        if uploaded_file_multi:
            with open("temp_image_multi.jpg", "wb") as f:
                f.write(uploaded_file_multi.getbuffer())
            image_path_multi = "temp_image_multi.jpg"
        elif use_sample_multi and sample_path_multi:
            image_path_multi = sample_path_multi
        else:
            st.error("Please upload or select sample image")

        if image_path_multi and segmentation:
            try:
                # Check if image exists
                if not os.path.exists(image_path_multi):
                    st.error(f"Image file not found: {image_path_multi}")
                else:
                    with st.spinner("Extracting and recognizing characters..."):
                        # Update confidence threshold and model choice
                        segmentation.confidence_threshold = multi_confidence_threshold
                        segmentation.model_choice = multi_model_choice

                        # Suppress all output during model prediction
                        with suppress_output():
                            result = segmentation.process_image(image_path_multi)

                        st.session_state.segmentation_results = result

                        if result['status'] == 'success':
                            if result['num_characters'] > 0:
                                st.success(f"✅ Found {result['num_characters']} characters with confidence > {multi_confidence_threshold*100:.0f}%!")
                            else:
                                st.warning(f"⚠️ No characters found with confidence > {multi_confidence_threshold*100:.0f}%. Try lowering the threshold.")
                        else:
                            st.error(result['message'])
            except Exception as e:
                st.error(f"Error: {str(e)}")

# TAB 3: Results
with tab3:
    st.header("Recognition Results")
    
    # Single character results
    if st.session_state.results:
        st.subheader("Single Character Result")
        r = st.session_state.results

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Model**: {r['model']}")
            st.write(f"**Character (Transliteration)**: {r['character']}")
            st.write(f"**Character (Tamil)**: {r.get('tamil_character', r['character'])}")
            st.write(f"**Confidence**: {r['confidence']*100:.2f}%")
            st.write(f"**Time**: {r['timestamp']}")
        
        with col2:
            try:
                img = Image.open(r['image_path'])
                st.image(img, use_column_width=True)
            except:
                st.info("Image preview unavailable")
        
        # Tamil Text Output
        st.subheader("Tamil Text Output")
        if sentence_generator:
            char_list = [r['character']]
            confidence_list = [r['confidence']]
            result = sentence_generator.generate_sentence(char_list, confidence_list)
            st.markdown(f"### Tamil Text: **{result['tamil_text']}**")
            st.write(f"**Character Sequence**: {' → '.join(result['character_sequence'])}")
            if result['found_words']:
                st.write(f"**Found Words**: {', '.join(result['found_words'])}")
    
    # Multi-character results
    if st.session_state.segmentation_results:
        st.subheader("Multi-Character Results")
        seg_result = st.session_state.segmentation_results
        
        if seg_result['status'] == 'success':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters Found", seg_result['num_characters'])
            with col2:
                st.metric("Average Confidence", f"{seg_result['average_confidence']*100:.2f}%")
            with col3:
                # Show confidence range
                if seg_result['num_characters'] > 0:
                    confidences = [r['confidence'] for r in st.session_state.segmentation_results.get('details', [])]
                    if confidences:
                        min_conf = min(confidences)
                        max_conf = max(confidences)
                        st.metric("Confidence Range", f"{min_conf*100:.1f}% - {max_conf*100:.1f}%")

            # Character sequence
            char_seq = ' → '.join(seg_result['character_sequence'])
            st.write(f"**Character Sequence**: {char_seq}")
            
            # Tamil text
            if sentence_generator:
                tamil_result = sentence_generator.generate_sentence(
                    seg_result['character_sequence'],
                    seg_result['confidences']
                )
                st.markdown(f"### Tamil Text: **{tamil_result['tamil_text']}**")
                if tamil_result['found_words']:
                    st.write(f"**Found Words**: {', '.join(tamil_result['found_words'])}")
            
            # Detailed character breakdown
            with st.expander("Detailed Character Breakdown"):
                for i, r in enumerate(seg_result['results'][:20]):  # Show first 20
                    tamil_char = character_mapping.get(r['character'], r['character'])
                    st.write(f"[{i+1}] {tamil_char} ({r['character']}): {r['confidence']*100:.2f}%")
                if len(seg_result['results']) > 20:
                    st.write(f"... and {len(seg_result['results']) - 20} more characters")

# TAB 4: Translation
with tab4:
    st.header("Tamil to English Translation")
    st.info("Translate Tamil text to English without using external APIs")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Tamil Text")
        tamil_input = st.text_area("Enter Tamil text to translate:", height=100, key="tamil_input")

    with col2:
        st.subheader("Translation Options")
        translation_mode = st.radio(
            "Select translation mode:",
            ["Direct Transliteration", "Dictionary + Transliteration"],
            help="Direct: Tamil → Roman script\nDictionary: Uses word dictionary when available"
        )
        show_word_mapping = st.checkbox("Show word-by-word mapping", value=True)
        show_confidence = st.checkbox("Show translation confidence", value=True)

    if st.button("Translate Tamil to English"):
        if tamil_input and translator:
            try:
                with st.spinner("Translating..."):
                    if translation_mode == "Direct Transliteration":
                        translation_result = translator.translate_text(tamil_input, use_direct_transliteration=True)
                    else:
                        translation_result = translator.translate_text(tamil_input)

                    st.success("Translation Complete!")

                    # Display translation
                    st.subheader("Roman/English Translation")
                    st.markdown(f"### {translation_result['english_text']}")

                    # Display statistics
                    if show_confidence and 'total_words' in translation_result:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Words", translation_result['total_words'])
                        with col2:
                            st.metric("Dictionary Matches", translation_result['dictionary_matches'])
                        with col3:
                            st.metric("Transliterated", translation_result['transliterated_words'])
                        with col4:
                            st.metric("Confidence", f"{translation_result['confidence']:.1f}%")
                    elif show_confidence:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Characters", translation_result['total_characters'])
                        with col2:
                            st.metric("Confidence", f"{translation_result['confidence']:.1f}%")

                    # Display word mappings
                    if show_word_mapping and 'word_mappings' in translation_result and translation_result['word_mappings']:
                        st.subheader("Word-by-Word Mapping")
                        for i, mapping in enumerate(translation_result['word_mappings'], 1):
                            match_type = "✓ Dictionary" if mapping['found_in_dict'] else "~ Transliterated"
                            st.write(f"[{i}] {mapping['tamil']} → **{mapping['english']}** ({match_type})")
            except Exception as e:
                st.error(f"Translation error: {str(e)}")
        else:
            st.error("Please enter Tamil text to translate")

    # Translation from results
    st.divider()
    st.subheader("Translate from Recognition Results")

    if st.session_state.segmentation_results:
        seg_result = st.session_state.segmentation_results
        if seg_result['status'] == 'success':
            # Get Tamil text from character sequence
            tamil_text = ''.join(seg_result['character_sequence'])

            st.write(f"**Recognized Tamil Text:** {tamil_text}")
            st.write(f"**Total Characters:** {len(seg_result['character_sequence'])}")

            col1, col2 = st.columns(2)
            with col1:
                result_translation_mode = st.radio(
                    "Translation mode for results:",
                    ["Direct Transliteration", "Character Meanings"],
                    key="result_translation_mode",
                    help="Direct: Tamil → Roman script\nCharacter Meanings: Shows phonetic meanings"
                )

            if st.button("Translate Recognized Text"):
                try:
                    with st.spinner("Translating recognized text..."):
                        if result_translation_mode == "Direct Transliteration":
                            # Direct transliteration
                            translation_result = translator.translate_text(tamil_text, use_direct_transliteration=True)
                            st.success("Translation Complete!")
                            st.markdown(f"### {translation_result['roman_text']}")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Characters", translation_result['total_characters'])
                            with col2:
                                st.metric("Confidence", f"{translation_result['confidence']:.1f}%")
                        else:
                            # Character meanings
                            translation_result = translator.translate_text(tamil_text, split_by_character=True)
                            st.success("Translation Complete!")
                            st.markdown(f"### **{translation_result['english_text']}**")

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Characters", translation_result['total_words'])
                            with col2:
                                st.metric("Dictionary Matches", translation_result['dictionary_matches'])
                            with col3:
                                st.metric("Transliterated", translation_result['transliterated_words'])
                            with col4:
                                st.metric("Confidence", f"{translation_result['confidence']:.1f}%")

                            # Show word-by-word breakdown
                            with st.expander("Word-by-Word Breakdown (First 50)"):
                                for i, mapping in enumerate(translation_result['word_mappings'][:50], 1):
                                    match_type = "✓ Dictionary" if mapping['found_in_dict'] else "~ Transliterated"
                                    st.write(f"[{i}] **{mapping['tamil']}** → *{mapping['english']}* ({match_type})")
                                if len(translation_result['word_mappings']) > 50:
                                    st.write(f"... and {len(translation_result['word_mappings']) - 50} more words")
                except Exception as e:
                    st.error(f"Translation error: {str(e)}")
    else:
        st.info("No recognized text available. Process an image first in the 'Full Text' tab.")

# TAB 5: Literal Meaning Translation
with tab5:
    st.header("Literal Meaning Translation")
    st.info("Get literal meanings and explanations of Tamil words and sentences")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Tamil Text")
        tamil_literal_input = st.text_area("Enter Tamil text to get literal meaning:", height=100, key="tamil_literal_input")

    with col2:
        st.subheader("Translation Options")
        show_word_details = st.checkbox("Show detailed word information", value=True)
        show_categories = st.checkbox("Show word categories", value=True)
        show_explanations = st.checkbox("Show explanations", value=True)

    if st.button("Get Literal Meaning"):
        if tamil_literal_input and translator:
            try:
                with st.spinner("Translating with literal meanings..."):
                    literal_result = translator.translate_with_literal_meaning(tamil_literal_input)

                    st.success("Literal Meaning Translation Complete!")

                    # Display main translation
                    st.subheader("English Translation")
                    st.markdown(f"### {literal_result['english_text']}")

                    # Display statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Words", literal_result['total_words'])
                    with col2:
                        st.metric("Found Words", literal_result['found_words'])
                    with col3:
                        st.metric("Transliterated", literal_result['transliterated_words'])
                    with col4:
                        st.metric("Confidence", f"{literal_result['confidence']:.1f}%")

                    # Display word-by-word meanings
                    if show_word_details and literal_result['word_meanings']:
                        st.subheader("Word-by-Word Literal Meanings")

                        for i, word_info in enumerate(literal_result['word_meanings'], 1):
                            with st.expander(f"[{i}] {word_info['tamil']} → {word_info['english']}", expanded=(i <= 3)):
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.write(f"**Tamil:** {word_info['tamil']}")
                                    st.write(f"**English:** {word_info['english']}")
                                    if show_categories:
                                        st.write(f"**Category:** {word_info['category']}")
                                    st.write(f"**Type:** {word_info['type']}")

                                with col2:
                                    st.write(f"**Meaning:** {word_info['meaning']}")
                                    if show_explanations:
                                        st.write(f"**Explanation:** {word_info['explanation']}")
                                    st.write(f"**Usage:** {word_info['usage']}")
                                    st.write(f"**Found:** {'✓ Yes' if word_info['found'] else '✗ No'}")

            except Exception as e:
                st.error(f"Literal meaning translation error: {str(e)}")
        else:
            st.error("Please enter Tamil text to translate")

    # Literal meaning from results
    st.divider()
    st.subheader("Get Literal Meaning from Recognition Results")

    if st.session_state.segmentation_results:
        seg_result = st.session_state.segmentation_results
        if seg_result['status'] == 'success':
            tamil_text = ''.join(seg_result['character_sequence'])
            st.write(f"**Recognized Tamil Text:** {tamil_text}")

            if st.button("Get Literal Meaning of Recognized Text"):
                try:
                    with st.spinner("Getting literal meanings..."):
                        literal_result = translator.translate_with_literal_meaning(tamil_text)
                        st.success("Literal Meaning Complete!")
                        st.markdown(f"### {literal_result['english_text']}")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Words", literal_result['total_words'])
                        with col2:
                            st.metric("Found Words", literal_result['found_words'])
                        with col3:
                            st.metric("Transliterated", literal_result['transliterated_words'])
                        with col4:
                            st.metric("Confidence", f"{literal_result['confidence']:.1f}%")

                        # Show word meanings
                        with st.expander("Word-by-Word Meanings"):
                            for i, word_info in enumerate(literal_result['word_meanings'], 1):
                                st.write(f"**[{i}] {word_info['tamil']}** → *{word_info['english']}*")
                                st.write(f"  - Meaning: {word_info['meaning']}")
                                st.write(f"  - Explanation: {word_info['explanation']}")
                                st.write("---")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("No recognized text available. Process an image first in the 'Full Text' tab.")

# TAB 6: Model Info
with tab6:
    st.header("🤖 Model Information")

    st.subheader("Available Models")
    st.info("You can select any of these models from the sidebar to use for character recognition")

    # Display all available models
    for model_key, info in model_info.items():
        with st.expander(f"📊 {info['name']}", expanded=(model_key == 'cnn')):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Type:** {info['type']}")
                st.write(f"**Accuracy:** {info['accuracy']}")
            with col2:
                st.write(f"**Description:** {info['description']}")

    st.markdown("---")
    st.subheader("General Information")
    st.write("**Classes**: 26 Tamil character classes")
    st.write("**Input Size**: 50x50 pixels")
    st.write("**Training Data**: 95 samples across 26 classes")
    st.write("**Preprocessing**: Grayscale conversion and normalization")

    st.markdown("---")
    st.subheader("Model Comparison")

    comparison_data = {
        'Model': [model_info[k]['name'] for k in model_info.keys()],
        'Type': [model_info[k]['type'] for k in model_info.keys()],
        'Accuracy': [model_info[k]['accuracy'] for k in model_info.keys()],
    }

    df = pd.DataFrame(comparison_data)
    st.table(df)

    st.markdown("---")
    st.subheader("Recommendations")

    st.success("""
    **✅ BEST CHOICE: Ensemble Voting Model**

    The Ensemble Voting model combines CNN, ResNet50, and Improved CNN using a voting mechanism.

    **Accuracy: 42.11%** (+14.3% improvement over baseline CNN)

    This is now the recommended model for best accuracy!
    """)

    st.info("""
    **Other Models:**

    - **CNN (Baseline)**: 36.84% accuracy, simple and reliable, good fallback
    - **ResNet50**: 10.53% accuracy, transfer learning, experimental, needs more data
    - **Improved CNN**: 10.53% accuracy, enhanced architecture, experimental, needs more data
    - **Improved Model (Best)**: 10.53% accuracy, optimized, experimental, needs more data

    Note: Individual improved models have lower accuracy due to limited training data (95 samples).
    They would perform better with 500+ samples.
    """)

# TAB 7: Help
with tab7:
    st.header("Help & Documentation")
    st.write("**Single Character Mode**: Upload or select an image to recognize a single character")
    st.write("**Full Text Mode**: Upload or select an image to extract and recognize all characters")
    st.write("**Results**: View detailed results including Tamil text output and confidence scores")
    st.write("**Translation**: Translate Tamil text to English without external APIs")
    st.write("**Literal Meaning**: Get detailed meanings and explanations of Tamil words with grammar information")

