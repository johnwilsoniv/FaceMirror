#!/bin/bash
set -e

echo "========================================"
echo "Building CalcParams Tool"
echo "========================================"

# Create build directory
mkdir -p calc_params_build

# Copy CMake file
cp CMakeLists_calc_params.txt calc_params_build/CMakeLists.txt
cp calc_params_tool.cpp calc_params_build/

cd calc_params_build

# Configure with CMake
echo "Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE=Release .

# Build
echo "Building..."
make -j4

# Check if binary was created
if [ -f "./calc_params_tool" ]; then
    echo "✓ Build successful!"
    echo "Binary location: $(pwd)/calc_params_tool"

    # Copy to parent directory for easy access
    cp calc_params_tool ../
    echo "Copied to: ../calc_params_tool"
else
    echo "✗ Build failed - binary not found"
    exit 1
fi

cd ..
echo "========================================"
echo "Done! Run with: ./calc_params_tool <pdm_file> <landmarks_file>"
echo "========================================"
