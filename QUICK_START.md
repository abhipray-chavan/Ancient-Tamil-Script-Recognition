# Quick Start Guide

Get the app running in 5 minutes!

## ⚡ TL;DR - Quick Setup

```bash
# 1. Clone and navigate
git clone https://github.com/yourusername/Ancient-Tamil-Script-Recognition.git
cd Ancient-Tamil-Script-Recognition

# 2. Create virtual environment
python3 -m venv env
source env/bin/activate  # macOS/Linux
# OR
env\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run streamlit_app.py

# 5. Open browser
# Go to: http://localhost:8501
```

## 🎯 What You Can Do

### 1. Single Character Recognition
- Upload a single Tamil character image
- Get instant recognition with confidence score
- See all 26 character class probabilities

### 2. Multi-Character Recognition
- Upload an inscription with multiple characters
- Automatic character extraction and recognition
- View character sequence with confidence scores

### 3. Full Text Recognition
- Upload complete inscription images
- Get full text output in Tamil Unicode
- View average confidence and character count

### 4. Translation
- Automatic English translation of recognized text
- Dictionary-based translation with 128+ words
- Phonetic transliteration fallback

### 5. Literal Meaning
- Word-by-word breakdown
- Detailed meanings and explanations
- Character categories and linguistic info

## 📊 Model Information

| Model | Accuracy | Speed | Best For |
|-------|----------|-------|----------|
| CNN | 36.84% | Fast (~50ms) | Quick recognition |
| Ensemble | 42.11% | Slower (~200ms) | Better accuracy |

## 🎛️ Key Settings

- **Confidence Threshold**: 0.15 (15%) - adjust for more/fewer characters
- **Model Selection**: CNN or Ensemble - choose in Multi-Character tab
- **Character Size**: 15x15 to 80x80 pixels - filters noise

## 📁 Project Structure

```
├── streamlit_app.py              # Main web app
├── character_segmentation.py     # Character extraction
├── ensemble_predictor.py         # Ensemble model
├── tamil_english_translator.py   # Translation
├── Model-Creation/               # Pre-trained models
├── Labels/                       # Training dataset
├── Input Images/                 # Sample images
└── requirements.txt              # Dependencies
```

## 🖼️ Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

**Recommended**: Clear, high-contrast images of Tamil inscriptions

## ⚙️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 8501 in use | `streamlit run streamlit_app.py --server.port 8502` |
| Module not found | `pip install -r requirements.txt` |
| Slow performance | Use CNN model, reduce image size |
| No characters found | Check image quality, lower confidence threshold |
| Low confidence | Image may not be Tamil inscription |

## 📚 Full Documentation

- **Setup Guide**: See `SETUP_GUIDE.md`
- **GitHub Push**: See `GITHUB_PUSH_GUIDE.md`
- **Full README**: See `README.md`

## 🚀 Next Steps

1. ✅ Install and run the app
2. ✅ Try with sample images in `Input Images/`
3. ✅ Upload your own inscription images
4. ✅ Experiment with different settings
5. ✅ Push to your GitHub repository

## 💡 Tips

- **First run**: Takes 30-60 seconds to load models
- **Best results**: Use clear, high-contrast inscription images
- **Faster processing**: Use CNN model instead of Ensemble
- **More accuracy**: Use Ensemble model (slower but better)
- **Adjust threshold**: Lower for more characters, higher for reliability

## 🔗 Useful Links

- [GitHub Repository](https://github.com/yourusername/Ancient-Tamil-Script-Recognition)
- [Streamlit Docs](https://docs.streamlit.io/)
- [TensorFlow Docs](https://www.tensorflow.org/docs)

## ❓ FAQ

**Q: Can I use this on my phone?**
A: Not directly, but you can access the web app from your phone if running on a network.

**Q: How accurate is the model?**
A: 36-42% depending on model. Limited by training data (95 samples).

**Q: Can I improve accuracy?**
A: Yes! Train with more data (1000+ samples per class).

**Q: Does it work offline?**
A: Yes, after initial setup. No internet needed to run.

**Q: Can I use GPU?**
A: Yes! Install `tensorflow[and-cuda]` for NVIDIA GPU support.

---

**Ready to recognize Tamil script? Let's go! 🎉**

