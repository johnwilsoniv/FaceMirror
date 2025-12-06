#!/bin/bash
# GPU Environment Setup for Big Red 200
# Run once to create GPU-enabled environment for neural network training
# Usage: bash setup_environment_gpu.sh

set -e

echo "=============================================="
echo "Setting up GPU PyFaceAU environment on Big Red 200"
echo "=============================================="

# Load GPU-enabled Python module
# Big Red 200 has pre-built PyTorch modules - check with: module spider pytorch
module purge
module load python/gpu/3.10.10

echo "Modules loaded:"
module list

# Create GPU virtual environment (separate from CPU environment)
GPU_VENV_DIR="$HOME/pyfaceau_gpu_env"

if [ ! -d "$GPU_VENV_DIR" ]; then
    echo "Creating GPU virtual environment at $GPU_VENV_DIR..."
    python -m venv $GPU_VENV_DIR
fi

source $GPU_VENV_DIR/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip

# Core dependencies
pip install numpy scipy pandas opencv-python-headless
pip install tqdm h5py numba scikit-learn

# PyTorch with CUDA support
# Check Big Red 200 CUDA version and adjust if needed
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ONNX for model export
pip install onnx onnxruntime-gpu

# TensorBoard for training monitoring
pip install tensorboard

# Install our packages in development mode
cd $HOME/pyfaceau
pip install -e pymtcnn/
pip install -e pyclnf/
pip install -e pyfaceau/
pip install -e pyfhog/

# Verify GPU is accessible
echo ""
echo "Verifying GPU access..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "=============================================="
echo "GPU Environment Setup Complete!"
echo "=============================================="
echo ""
echo "To use this environment in SLURM GPU jobs, add:"
echo "  module load python/gpu/3.10.10"
echo "  source $GPU_VENV_DIR/bin/activate"
echo ""
echo "For TensorBoard monitoring, create SSH tunnel:"
echo "  ssh -L 6006:localhost:6006 $USER@bigred200.uits.iu.edu"
echo "  Then open http://localhost:6006 in your browser"
echo ""
