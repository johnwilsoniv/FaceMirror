#!/bin/bash
# Setup script for Big Red 200 - run once to create environment
# Usage: bash setup_environment.sh

set -e

echo "=============================================="
echo "Setting up PyFaceAU environment on Big Red 200"
echo "=============================================="

# Load modules (CPU-only - CLNF doesn't use GPU)
module purge
module load python/3.12.11

echo "Modules loaded:"
module list

# Create conda environment (if using conda)
# Alternatively, use a virtual environment
VENV_DIR="$HOME/pyfaceau_env"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip

# Core dependencies
pip install numpy scipy pandas opencv-python-headless
pip install onnxruntime    # CPU version (CLNF is CPU-bound anyway)
pip install tqdm h5py numba

# Install our packages in development mode
cd $HOME/pyfaceau
pip install -e pymtcnn/
pip install -e pyclnf/
pip install -e pyfaceau/
pip install -e pyfhog/

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To use this environment in SLURM jobs, add:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Or use: module load python/gpu/3.10.10"
