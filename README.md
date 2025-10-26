# Ancient Tamil Script Recognition

A deep learning-based application for recognizing and translating ancient Tamil script from inscriptions. This project uses CNN and ensemble models to extract and recognize individual characters from Tamil inscription images.

## 🎯 Features

### 1. **Single Character Recognition**
- Upload an image of a single Tamil character
- Get instant character recognition with confidence score
- View transliteration and Tamil Unicode output
- See detailed prediction probabilities for all 26 character classes

### 2. **Multi-Character Recognition**
- Upload full inscription images
- Automatic character segmentation and extraction
- Batch character recognition with confidence filtering
- View character sequence with confidence scores
- Display results in both transliteration and Tamil Unicode

### 3. **Full Text Recognition**
- Process complete inscription images
- Extract and recognize all characters automatically
- Get full text output in Tamil Unicode
- View average confidence and character count

### 4. **Tamil Translation**
- Automatic translation of recognized text to English
- Dictionary-based translation with 128+ word entries
- Phonetic transliteration fallback
- Support for common Tamil words and phrases

### 5. **Literal Meaning Translation**
- Word-by-word breakdown of recognized text
- Detailed meanings and explanations
- Character categories and linguistic information
- Educational insights into Tamil script

### 6. **Model Selection**
- Choose between CNN (faster) or Ensemble (more accurate) models
- Adjustable confidence threshold for predictions
- Real-time model switching without restart

## 📊 Model Performance

- **CNN Model**: 36.84% accuracy, ~50ms per character
- **ResNet50 Model**: 10.53% accuracy
- **Ensemble Model**: 42.11% accuracy (voting-based combination)
- **Training Data**: 95 samples across 26 Tamil character classes

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 2GB RAM minimum
- 500MB disk space for models

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Ancient-Tamil-Script-Recognition.git
cd Ancient-Tamil-Script-Recognition
```

2. **Create a virtual environment** (recommended)
```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

1. **Start the Streamlit app**
```bash
streamlit run streamlit_app.py
```

2. **Open in browser**
```
http://localhost:8501
```

3. **Select a tab and upload an image**
- Single Character: Upload a single character image
- Multi-Character: Upload an inscription with multiple characters
- Full Text: Upload a complete inscription
- Translation: Get English translation of recognized text
- Literal Meaning: Get word-by-word breakdown

## 📁 Project Structure

```
Ancient-Tamil-Script-Recognition/
├── streamlit_app.py              # Main web application
├── character_segmentation.py     # Character extraction & recognition
├── ensemble_predictor.py         # Ensemble voting model
├── tamil_english_translator.py   # Translation module
├── tamil_sentence_generator.py   # Sentence generation
├── Model-Creation/               # Pre-trained models
│   ├── CNN.model.keras
│   ├── ResNet50_Tamil.keras
│   ├── improved_cnn_model.keras
│   └── improved_model_best.keras
├── Labels/                       # Training dataset
│   └── Labelled Dataset - Fig 51/
├── Input Images/                 # Sample images for testing
└── requirements.txt              # Python dependencies
```

## 🔧 Configuration

### Confidence Threshold
- **Default**: 0.15 (15%)
- **Range**: 0.0 - 1.0
- **Lower**: More characters detected but less reliable
- **Higher**: Fewer characters but more reliable

### Character Size Filtering
- **Minimum**: 15x15 pixels
- **Maximum**: 80x80 pixels
- Filters out noise and artifacts

### Sanity Checks
- **Max characters per image**: 100
- **Min average confidence**: 20%
- **Min high-confidence predictions**: 50%

## 📝 Supported Characters

The model recognizes 26 Tamil character classes:

```
a, ai, c, e, i, k, l, l5, l5u, l5u4, n, n1, n1u4, n2, n2u4, 
n3, n5, o, p, pu4, r, r5, r5i, ru, t, y
```

## ⚠️ Limitations

1. **Limited Training Data**: Only 95 samples across 26 classes
2. **Model Accuracy**: 36-42% (CNN/Ensemble)
3. **Image Quality**: Works best with clear, high-contrast inscriptions
4. **Character Isolation**: Performs better on well-segmented characters
5. **False Positives**: May detect noise in non-inscription images

## 🔍 Troubleshooting

### Issue: "Low confidence predictions. Image may not be a Tamil inscription."
- **Cause**: Model confidence is too low (< 20%)
- **Solution**: 
  - Ensure image is a clear Tamil inscription
  - Try adjusting confidence threshold lower
  - Check image quality and contrast

### Issue: No characters detected
- **Cause**: Image preprocessing failed or no valid characters found
- **Solution**:
  - Ensure image is grayscale or color
  - Check image resolution (minimum 100x100 pixels)
  - Try different confidence threshold

### Issue: Incorrect character recognition
- **Cause**: Model has limited training data
- **Solution**:
  - Try ensemble model instead of CNN
  - Adjust confidence threshold
  - Ensure character is clear and isolated

## 🎓 How It Works

### Character Segmentation Pipeline

1. **Image Preprocessing**
   - Convert to grayscale
   - Denoise using fastNlMeansDenoising
   - Apply morphological operations (CLOSE, OPEN)

2. **Thresholding**
   - Use Otsu's automatic thresholding
   - Create binary image for contour detection

3. **Character Extraction**
   - Find contours in binary image
   - Filter by size, solidity, aspect ratio, area
   - Extract individual character patches

4. **Character Recognition**
   - Normalize character image to 60x60 pixels
   - Pass through CNN or Ensemble model
   - Get confidence scores for all 26 classes

5. **Post-Processing**
   - Apply confidence threshold filtering
   - Validate average confidence
   - Return recognized characters

## 📦 Dependencies

- tensorflow >= 2.10.0
- opencv-python >= 4.6.0
- streamlit >= 1.20.0
- numpy >= 1.21.0
- pillow >= 9.0.0

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👨‍💻 Author

Abhipray Chavan

## 🙏 Acknowledgments

- Tamil character dataset from academic research
- Deep learning frameworks: TensorFlow, Keras
- Web framework: Streamlit
- Computer vision: OpenCV

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Last Updated**: October 2025

