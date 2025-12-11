#!/bin/bash
# ============================================================================
# Build OpenFace 2.2.0 on Big Red 200
#
# This script compiles OpenFace and its dependencies (OpenCV, dlib) on Big Red.
# Run this ONCE before running generate_cpp_reference.slurm
#
# Usage:
#   chmod +x build_openface.sh
#   ./build_openface.sh
#
# Estimated time: ~60 minutes (includes OpenCV build)
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
OPENCV_DIR="$INSTALL_DIR/opencv"
BUILD_JOBS=16

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# ============================================
# Step 1: Load required modules
# ============================================
echo ""
echo "[1/6] Loading modules..."
module purge
module load gcc/11.2.0
module load cmake/4.1.0
module load boost/1.86.0
module load openblas/0.3.26
module load ffmpeg/7.7.1

# Set OpenBLAS paths explicitly (module doesn't set env vars correctly)
export OPENBLAS_ROOT="/N/soft/sles15sp6/openblas/gnu/0.3.26"
export LD_LIBRARY_PATH="$OPENBLAS_ROOT/lib:$LD_LIBRARY_PATH"

# Set ffmpeg paths for OpenCV build
export FFMPEG_ROOT="/N/soft/sles15sp6/ffmpeg/7.7.1"
export PKG_CONFIG_PATH="$FFMPEG_ROOT/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:$LD_LIBRARY_PATH"

# Check versions
echo "  GCC: $(gcc --version | head -1)"
echo "  CMake: $(cmake --version | head -1)"
echo "  OpenBLAS: $OPENBLAS_ROOT"
echo "  ffmpeg: $(ffmpeg -version | head -1)"

# ============================================
# Step 2: Build OpenCV (not available as module)
# ============================================
echo ""
echo "[2/6] Building OpenCV 4.10.0 (ffmpeg 7.x compatible)..."
if [ ! -d "$OPENCV_DIR" ]; then
    git clone https://github.com/opencv/opencv.git "$OPENCV_DIR"
fi

cd "$OPENCV_DIR"
git fetch --tags
git checkout 4.10.0  # Use 4.10.0 for ffmpeg 7.x compatibility

mkdir -p build && cd build
if [ ! -f "lib/libopencv_core.so" ]; then
    # Note: Exclude calib3d (LAPACK API mismatch with OpenBLAS 0.3.26)
    # OpenFace only needs core, imgproc, imgcodecs, videoio, highgui, objdetect
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/opencv_install" \
        -DBUILD_LIST=core,imgproc,imgcodecs,videoio,highgui,objdetect,features2d,flann \
        -DWITH_FFMPEG=ON \
        -DWITH_V4L=OFF \
        -DWITH_GTK=OFF \
        -DWITH_QT=OFF \
        -DWITH_LAPACK=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_apps=OFF
    make -j$BUILD_JOBS
    make install
fi

export OpenCV_DIR="$INSTALL_DIR/opencv_install/lib64/cmake/opencv4"
export LD_LIBRARY_PATH="$INSTALL_DIR/opencv_install/lib64:$LD_LIBRARY_PATH"
echo "  OpenCV installed to: $INSTALL_DIR/opencv_install"

# ============================================
# Step 3: Build dlib (OpenFace dependency)
# ============================================
echo ""
echo "[3/6] Building dlib..."
if [ ! -d "$DLIB_DIR" ]; then
    git clone https://github.com/davisking/dlib.git "$DLIB_DIR"
fi

cd "$DLIB_DIR"
git checkout v19.24  # Stable version

mkdir -p build && cd build
if [ ! -f "$INSTALL_DIR/dlib_install/lib/libdlib.a" ]; then
    # Disable PNG/GIF/JPEG to avoid nested cmake compatibility issues
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/dlib_install" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DDLIB_USE_BLAS=ON \
        -DDLIB_USE_LAPACK=ON \
        -DDLIB_NO_GUI_SUPPORT=ON \
        -DDLIB_PNG_SUPPORT=OFF \
        -DDLIB_GIF_SUPPORT=OFF \
        -DDLIB_JPEG_SUPPORT=OFF
    make -j$BUILD_JOBS
    make install
fi

export DLIB_ROOT="$INSTALL_DIR/dlib_install"
echo "  dlib installed to: $DLIB_ROOT"

# ============================================
# Step 4: Clone OpenFace
# ============================================
echo ""
echo "[4/6] Cloning OpenFace..."
cd "$INSTALL_DIR"
if [ ! -d "$OPENFACE_DIR" ]; then
    git clone https://github.com/TadasBaltrusaitis/OpenFace.git "$OPENFACE_DIR"
fi

cd "$OPENFACE_DIR"
git checkout OpenFace_2.2.0

# ============================================
# Step 5: Build OpenFace
# ============================================
echo ""
echo "[5/6] Building OpenFace..."
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -Ddlib_DIR="$DLIB_ROOT/lib/cmake/dlib" \
    -DOpenCV_DIR="$OpenCV_DIR" \
    -DCMAKE_PREFIX_PATH="$DLIB_ROOT;$INSTALL_DIR/opencv_install;$OPENBLAS_ROOT" \
    -DOpenBLAS_INCLUDE_DIR="$OPENBLAS_ROOT/include" \
    -DOpenBLAS_LIB="$OPENBLAS_ROOT/lib/libopenblas.so"

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
# Step 6: Download models
# ============================================
echo ""
echo "[6/6] Downloading models..."
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
echo "  module load gcc/11.2.0"
echo "  export LD_LIBRARY_PATH=\"$INSTALL_DIR/opencv_install/lib64:\$LD_LIBRARY_PATH\""
echo "  export OPENFACE_BIN=\"$OPENFACE_DIR/build/bin/FeatureExtraction\""
echo ""
echo "End time: $(date)"
