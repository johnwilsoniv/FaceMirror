#!/bin/bash
#
# Run ONNX conversion for MTL (Multi-Task Learning) model
# This converts the PyTorch MTL model (EfficientNet-B0) to ONNX format for Apple Silicon acceleration
#

set -e  # Exit on error

cd "$(dirname "$0")"

echo "========================================"
echo "MTL Model ONNX Conversion"
echo "========================================"
echo ""

# Check if model exists (it's a symlink to HuggingFace cache)
if [ ! -e "weights/MTL_backbone.pth" ]; then
    echo "Error: weights/MTL_backbone.pth not found!"
    echo "Please ensure the MTL model is downloaded."
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
echo "Converting MTL model to ONNX..."
$PYTHON_CMD convert_mtl_to_onnx.py --model weights/MTL_backbone.pth --output weights/mtl_efficientnet_b0_coreml.onnx --verify

# Check if conversion succeeded
if [ -f "weights/mtl_efficientnet_b0_coreml.onnx" ]; then
    echo ""
    echo "✓ Conversion successful!"
    echo ""
    echo "ONNX model created: weights/mtl_efficientnet_b0_coreml.onnx"
    ls -lh weights/mtl_efficientnet_b0_coreml.onnx
    echo ""
    echo "Next time you run Face Mirror AU extraction, it will automatically use the accelerated model."
    echo "Expected speedup: 3-5x (from ~50-100ms to ~15-30ms per face)"
else
    echo ""
    echo "✗ Conversion failed!"
    echo "Please check the error messages above."
    exit 1
fi
