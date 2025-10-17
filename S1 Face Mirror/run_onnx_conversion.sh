#!/bin/bash
#
# Run ONNX conversion for STAR landmark model
# This converts the PyTorch STAR model to ONNX format for Apple Silicon acceleration
#

set -e  # Exit on error

cd "$(dirname "$0")"

echo "========================================"
echo "STAR Model ONNX Conversion"
echo "========================================"
echo ""

# Check if model exists
if [ ! -f "weights/Landmark_98.pkl" ]; then
    echo "Error: weights/Landmark_98.pkl not found!"
    echo "Please ensure the STAR model is downloaded."
    exit 1
fi

# Find Python with PyTorch installed
PYTHON_CMD="/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
if ! $PYTHON_CMD -c "import torch" 2>/dev/null; then
    PYTHON_CMD="/usr/local/bin/python3"
fi
if ! $PYTHON_CMD -c "import torch" 2>/dev/null; then
    echo "Error: Could not find Python with PyTorch installed!"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}')"
echo ""

# Run conversion
echo "Converting STAR model to ONNX..."
$PYTHON_CMD convert_star_to_onnx.py --model weights/Landmark_98.pkl --output weights/star_landmark_98_coreml.onnx

# Check if conversion succeeded
if [ -f "weights/star_landmark_98_coreml.onnx" ]; then
    echo ""
    echo "✓ Conversion successful!"
    echo ""
    echo "ONNX model created: weights/star_landmark_98_coreml.onnx"
    ls -lh weights/star_landmark_98_coreml.onnx
    echo ""
    echo "Next time you run Face Mirror, it will automatically use the accelerated model."
    echo "Expected speedup: 10-20x (from ~1800ms to 90-180ms per frame)"
else
    echo ""
    echo "✗ Conversion failed!"
    echo "Please check the error messages above."
    exit 1
fi
