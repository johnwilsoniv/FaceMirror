#!/bin/bash
#
# Run ONNX conversion for RetinaFace detection model
# This converts the PyTorch RetinaFace model to ONNX format for Apple Silicon acceleration
#

set -e  # Exit on error

cd "$(dirname "$0")"

echo "========================================"
echo "RetinaFace Model ONNX Conversion"
echo "========================================"
echo ""

# Check if model exists
if [ ! -f "weights/Alignment_RetinaFace.pth" ]; then
    echo "Error: weights/Alignment_RetinaFace.pth not found!"
    echo "Please ensure the RetinaFace model is downloaded."
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
echo "Converting RetinaFace model to ONNX..."
$PYTHON_CMD convert_retinaface_to_onnx.py --model weights/Alignment_RetinaFace.pth --output weights/retinaface_mobilenet025_coreml.onnx --verify

# Check if conversion succeeded
if [ -f "weights/retinaface_mobilenet025_coreml.onnx" ]; then
    echo ""
    echo "✓ Conversion successful!"
    echo ""
    echo "ONNX model created: weights/retinaface_mobilenet025_coreml.onnx"
    ls -lh weights/retinaface_mobilenet025_coreml.onnx
    echo ""
    echo "Next time you run Face Mirror, it will automatically use the accelerated model."
    echo "Expected speedup: 5-10x (from ~191ms to ~20-40ms per detection)"
else
    echo ""
    echo "✗ Conversion failed!"
    echo "Please check the error messages above."
    exit 1
fi
