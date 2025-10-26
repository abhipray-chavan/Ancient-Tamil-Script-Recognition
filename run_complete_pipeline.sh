#!/bin/bash
# Complete Ancient Tamil Script Recognition Pipeline
# This script runs the entire end-to-end process

echo "=========================================="
echo "Ancient Tamil Script Recognition Pipeline"
echo "=========================================="
echo ""

# Check if Original.jpg exists
if [ ! -f "Original.jpg" ]; then
    echo "⚠️  Original.jpg not found!"
    echo "Copying sample image..."
    cp "Input Images/Inscriptions - Wiki1/1.jpg" Original.jpg
    echo "✓ Sample image copied"
fi

echo ""
echo "Step 1/3: Image Preprocessing"
echo "------------------------------"
python3 run_preprocessing.py
if [ $? -ne 0 ]; then
    echo "❌ Preprocessing failed!"
    exit 1
fi

echo ""
echo "Step 2/3: Character Segmentation"
echo "---------------------------------"
python3 run_segmentation.py
if [ $? -ne 0 ]; then
    echo "❌ Segmentation failed!"
    exit 1
fi

echo ""
echo "Step 3/3: Character Recognition"
echo "--------------------------------"
python3 run_recognition.py
if [ $? -ne 0 ]; then
    echo "❌ Recognition failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ PIPELINE COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - ImagePreProcessingFinal.jpg (preprocessed image)"
echo "  - box.jpg (segmentation visualization)"
echo "  - Images/ (segmented characters)"
echo "  - recognition_results.txt (recognition results)"
echo ""
echo "View the results:"
echo "  cat recognition_results.txt"
echo ""

