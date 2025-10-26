# Setup Guide - Ancient Tamil Script Recognition

Complete step-by-step guide to get the application running on your machine.

## 🖥️ System Requirements

- **OS**: macOS, Linux, or Windows
- **Python**: 3.8 or higher
- **RAM**: 2GB minimum (4GB recommended)
- **Disk Space**: 500MB for models and dependencies
- **Internet**: Required for initial setup

## 📋 Prerequisites

### Check Python Installation

```bash
python3 --version
# Should output: Python 3.8.x or higher
```

If Python is not installed, download from [python.org](https://www.python.org/downloads/)

### Check pip Installation

```bash
pip3 --version
# Should output: pip x.x.x from ...
```

## 🚀 Installation Steps

### Step 1: Clone the Repository

```bash
# Navigate to your desired directory
cd ~/Desktop  # or any directory you prefer

# Clone the repository
git clone https://github.com/yourusername/Ancient-Tamil-Script-Recognition.git

# Navigate into the project
cd Ancient-Tamil-Script-Recognition
```

### Step 2: Create Virtual Environment

Creating a virtual environment isolates project dependencies from your system Python.

**On macOS/Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

**On Windows:**
```bash
python -m venv env
env\Scripts\activate
```

You should see `(env)` prefix in your terminal after activation.

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow (deep learning framework)
- OpenCV (computer vision)
- Streamlit (web framework)
- NumPy (numerical computing)
- Pillow (image processing)
- And other required packages

**Installation time**: 5-15 minutes depending on internet speed

### Step 5: Verify Installation

```bash
python3 << 'EOF'
import tensorflow as tf
import cv2
import streamlit as st
import numpy as np
print("✅ All dependencies installed successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"OpenCV version: {cv2.__version__}")
EOF
```

## ▶️ Running the Application

### Start the Streamlit App

```bash
streamlit run streamlit_app.py
```

### Expected Output

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Open in Browser

- **Local**: http://localhost:8501
- **Network**: http://192.168.x.x:8501 (for other devices on same network)

## 🎯 First Run

1. **Wait for app to load** (first run takes 30-60 seconds as models load)
2. **Select a tab** from the sidebar
3. **Upload an image** using the file uploader
4. **View results** with character recognition and confidence scores

## 🛑 Stopping the Application

Press `Ctrl+C` in the terminal where Streamlit is running.

## 🔄 Deactivating Virtual Environment

When done, deactivate the virtual environment:

```bash
deactivate
```

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution:**
```bash
# Make sure virtual environment is activated
source env/bin/activate  # macOS/Linux
env\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Port 8501 is already in use"

**Solution:**
```bash
# Use a different port
streamlit run streamlit_app.py --server.port 8502
```

### Issue: "No module named 'streamlit'"

**Solution:**
```bash
pip install streamlit --upgrade
```

### Issue: Models not loading / "FileNotFoundError"

**Solution:**
- Ensure you're in the correct directory: `Ancient-Tamil-Script-Recognition/`
- Check that `Model-Creation/` folder exists with `.keras` files
- Verify file paths in `streamlit_app.py`

### Issue: Slow performance / High CPU usage

**Solution:**
- Close other applications
- Use CNN model instead of Ensemble (faster)
- Reduce image resolution before uploading
- Increase confidence threshold to process fewer characters

### Issue: "CUDA not available" warning

**Solution:**
- This is normal if you don't have NVIDIA GPU
- CPU mode will work fine, just slower
- To use GPU, install: `pip install tensorflow[and-cuda]`

## 📱 Using on Different Machines

### Transfer to Another Computer

1. **Copy the entire folder** to the new machine
2. **Follow installation steps** (Steps 2-4 above)
3. **Run the app** (Step 5)

### Using on Network

To access from another device on the same network:

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

Then access from other device using: `http://your-machine-ip:8501`

## 🔐 Security Notes

- Don't share your GitHub credentials
- Keep API keys and sensitive data in `.env` file (not in code)
- Use `.gitignore` to prevent uploading sensitive files

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/docs)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed successfully
- [ ] All imports verified
- [ ] Streamlit app starts without errors
- [ ] Can upload and process images
- [ ] Results display correctly

## 🆘 Still Having Issues?

1. Check the error message carefully
2. Search for the error in [GitHub Issues](https://github.com/yourusername/Ancient-Tamil-Script-Recognition/issues)
3. Create a new issue with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce
   - Screenshots if applicable

---

**Happy Recognition! 🎉**

