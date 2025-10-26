#!/usr/bin/env python3
"""
Streamlit Web Application for Ancient Tamil Script Recognition
Full end-to-end pipeline with single and multi-character recognition
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import pickle
import json
from datetime import datetime
import os
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')
from tamil_sentence_generator import TamilSentenceGenerator
from character_segmentation import CharacterSegmentation

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
    try:
        models['cnn'] = tf.keras.models.load_model('Model-Creation/CNN.model.keras')
    except:
        pass
    try:
        models['resnet50'] = tf.keras.models.load_model('Model-Creation/ResNet50_Tamil.keras')
    except:
        pass
    return models

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
        return CharacterSegmentation('Model-Creation/CNN.model.keras', character_names)
    except Exception as e:
        st.warning(f"Could not load character segmentation: {e}")
        return None

# Load resources
models = load_models()
character_classes = load_character_classes()
sentence_generator = load_sentence_generator()
segmentation = load_character_segmentation()

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
model_choice = st.sidebar.radio("Select Model:", ["CNN (Current)", "ResNet50"])
confidence_threshold = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.3)

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Single Character", "Full Text (Multi-Char)", "Results", "Model Info", "Help"])

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
                with st.spinner("Processing..."):
                    model = models.get('cnn' if model_choice == "CNN (Current)" else 'resnet50')
                    if not model:
                        st.error(f"{model_choice} not loaded")
                    else:
                        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        img_resized = cv2.resize(img, (50, 50))
                        img_norm = img_resized / 255.0
                        X = np.array([img_norm]).reshape(1, 50, 50, 1)
                        
                        predictions = model.predict(X, verbose=0)
                        confidence = np.max(predictions[0])
                        pred_idx = np.argmax(predictions[0])
                        pred_class = character_classes[pred_idx]
                        
                        st.session_state.results = {
                            'character': pred_class,
                            'confidence': confidence,
                            'predictions': predictions[0],
                            'model': model_choice,
                            'image_path': image_path,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        st.success(f"Recognized: **{pred_class}** ({confidence*100:.2f}%)")
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
                with st.spinner("Extracting and recognizing characters..."):
                    result = segmentation.process_image(image_path_multi)
                    st.session_state.segmentation_results = result
                    
                    if result['status'] == 'success':
                        st.success(f"Found {result['num_characters']} characters!")
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
            st.write(f"**Character**: {r['character']}")
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
            st.write(f"**Characters Found**: {seg_result['num_characters']}")
            st.write(f"**Average Confidence**: {seg_result['average_confidence']*100:.2f}%")
            
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
                    st.write(f"[{i+1}] {r['character']}: {r['confidence']*100:.2f}%")
                if len(seg_result['results']) > 20:
                    st.write(f"... and {len(seg_result['results']) - 20} more characters")

# TAB 4: Model Info
with tab4:
    st.header("Model Information")
    st.write("**CNN Model**: Convolutional Neural Network trained on 95 samples")
    st.write("**Classes**: 26 Tamil character classes")
    st.write("**Accuracy**: ~36.84% on test set")
    st.write("**Input Size**: 50x50 pixels")

# TAB 5: Help
with tab5:
    st.header("Help & Documentation")
    st.write("**Single Character Mode**: Upload or select an image to recognize a single character")
    st.write("**Full Text Mode**: Upload or select an image to extract and recognize all characters")
    st.write("**Results**: View detailed results including Tamil text output and confidence scores")

