#!/bin/bash
# ============================================================================
# Build OpenFace 2.2.0 on Big Red 200
#
# This script compiles OpenFace and its dependencies (dlib, OpenCV) on Big Red.
# Run this ONCE before running generate_cpp_reference.slurm
#
# Usage:
#   chmod +x build_openface.sh
#   ./build_openface.sh
#
# Estimated time: ~30 minutes
# ============================================================================

set -e  # Exit on error

echo "============================================"
echo "Building OpenFace on Big Red 200"
echo "Start time: $(date)"
echo "============================================"

# Configuration
INSTALL_DIR="$HOME/software"
OPENFACE_DIR="$INSTALL_DIR/OpenFace"
DLIB_DIR="$INSTALL_DIR/dlib"
BUILD_JOBS=16

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# ============================================
# Step 1: Load required modules
# ============================================
echo ""
echo "[1/5] Loading modules..."
module purge
module load gcc/11.2.0
module load cmake/3.20.2
module load opencv/4.5.5
module load boost/1.78.0

# Check versions
echo "  GCC: $(gcc --version | head -1)"
echo "  CMake: $(cmake --version | head -1)"

# ============================================
# Step 2: Build dlib (OpenFace dependency)
# ============================================
echo ""
echo "[2/5] Building dlib..."
if [ ! -d "$DLIB_DIR" ]; then
    git clone https://github.com/davisking/dlib.git "$DLIB_DIR"
fi

cd "$DLIB_DIR"
git checkout v19.24  # Stable version

mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/dlib_install" \
    -DDLIB_USE_BLAS=ON \
    -DDLIB_USE_LAPACK=ON \
    -DDLIB_NO_GUI_SUPPORT=ON

make -j$BUILD_JOBS
make install

export DLIB_ROOT="$INSTALL_DIR/dlib_install"
echo "  dlib installed to: $DLIB_ROOT"

# ============================================
# Step 3: Clone OpenFace
# ============================================
echo ""
echo "[3/5] Cloning OpenFace..."
cd "$INSTALL_DIR"
if [ ! -d "$OPENFACE_DIR" ]; then
    git clone https://github.com/TadasBaltrusaitis/OpenFace.git "$OPENFACE_DIR"
fi

cd "$OPENFACE_DIR"
git checkout OpenFace_2.2.0

# ============================================
# Step 4: Build OpenFace
# ============================================
echo ""
echo "[4/5] Building OpenFace..."
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -Ddlib_DIR="$DLIB_ROOT/lib/cmake/dlib" \
    -DCMAKE_PREFIX_PATH="$DLIB_ROOT"

make -j$BUILD_JOBS

# Check if build succeeded
if [ -f "bin/FeatureExtraction" ]; then
    echo "  Build SUCCESS!"
    echo "  Binary: $OPENFACE_DIR/build/bin/FeatureExtraction"
else
    echo "  Build FAILED - FeatureExtraction not found"
    exit 1
fi

# ============================================
# Step 5: Download models
# ============================================
echo ""
echo "[5/5] Downloading models..."
cd "$OPENFACE_DIR"

# Download model files if not present
if [ ! -f "lib/local/LandmarkDetector/model/patch_experts/cen_patches_0.25_of.dat" ]; then
    echo "  Downloading CEN patch experts..."
    # Use the official download script
    cd lib/local/LandmarkDetector/model
    ./download_cen_patch_experts.sh || {
        echo "  WARNING: Could not download models automatically"
        echo "  Please download manually from:"
        echo "  https://github.com/TadasBaltrusaitis/OpenFace/wiki/Model-download"
    }
    cd "$OPENFACE_DIR"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "============================================"
echo "Build Complete!"
echo "============================================"
echo ""
echo "OpenFace binary: $OPENFACE_DIR/build/bin/FeatureExtraction"
echo ""
echo "To use in SLURM jobs, add to your script:"
echo ""
echo "  module load gcc/11.2.0 opencv/4.5.5"
echo "  export OPENFACE_BIN=\"$OPENFACE_DIR/build/bin/FeatureExtraction\""
echo ""
echo "End time: $(date)"
