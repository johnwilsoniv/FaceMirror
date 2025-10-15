#!/bin/bash
# Build script for SplitFace applications on macOS
# Creates standalone .app bundles for S1, S2, and S3

set -e  # Exit on error

echo "=========================================="
echo "SplitFace v2.0.0 - macOS Build Script"
echo "=========================================="
echo ""

# Check if PyInstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "ERROR: PyInstaller not found. Please install it:"
    echo "  pip install pyinstaller"
    exit 1
fi

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"
echo ""

# Function to build an application
build_app() {
    local app_name="$1"
    local app_dir="$2"
    local spec_file="$3"

    echo "=========================================="
    echo "Building: $app_name"
    echo "=========================================="

    cd "$app_dir"

    # Clean previous builds
    if [ -d "build" ]; then
        echo "Cleaning previous build artifacts..."
        rm -rf build
    fi

    if [ -d "dist" ]; then
        echo "Cleaning previous distribution..."
        rm -rf dist
    fi

    # Run PyInstaller
    echo "Running PyInstaller..."
    pyinstaller "$spec_file"

    if [ $? -eq 0 ]; then
        echo "✓ $app_name built successfully!"
        echo "  Location: $app_dir/dist/$app_name.app"
    else
        echo "✗ $app_name build failed!"
        exit 1
    fi

    echo ""
    cd - > /dev/null
}

# Build S1 Face Mirror
build_app "Face Mirror" "S1 Face Mirror" "Face_Mirror.spec"

# Build S2 Action Coder
build_app "Action Coder" "S2 Action Coder" "Action_Coder.spec"

# Build S3 Data Analysis
build_app "Data Analysis" "S3 Data Analysis" "Data_Analysis.spec"

echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "Applications built:"
echo "  • S1 Face Mirror/dist/Face Mirror.app"
echo "  • S2 Action Coder/dist/Action Coder.app"
echo "  • S3 Data Analysis/dist/Data Analysis.app"
echo ""
echo "To run the applications:"
echo "  open 'S1 Face Mirror/dist/Face Mirror.app'"
echo "  open 'S2 Action Coder/dist/Action Coder.app'"
echo "  open 'S3 Data Analysis/dist/Data Analysis.app'"
echo ""
