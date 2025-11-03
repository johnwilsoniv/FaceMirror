#!/bin/bash
# SplitFace Build and Package Script for macOS
# Builds standalone .app bundles and creates DMG installers
# Single command for complete distribution-ready builds

set -e  # Exit on error

echo "=========================================="
echo "SplitFace v2.0.0 - Build & Package"
echo "=========================================="
echo ""

VERSION="2.0.0"
OUTPUT_DIR="DMG_Installers"

# Check if PyInstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "ERROR: PyInstaller not found. Please install it:"
    echo "  conda install -c conda-forge pyinstaller"
    echo "  # or"
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

# Create output directory for DMGs
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "STEP 1: Building Applications"
echo "=========================================="
echo ""

# Build all three applications
build_app "Face Mirror" "S1 Face Mirror" "Face_Mirror.spec"
build_app "Action Coder" "S2 Action Coder" "Action_Coder.spec"
build_app "Data Analysis" "S3 Data Analysis" "Data_Analysis.spec"

echo "=========================================="
echo "STEP 2: Creating Combined DMG Installer"
echo "=========================================="
echo ""

# Create single DMG with all three apps in a SplitFace folder
DMG_NAME="SplitFace-v$VERSION.dmg"

echo "Creating combined installer: $DMG_NAME"
echo ""

# Check if all apps exist
MISSING_APPS=0
if [ ! -d "S1 Face Mirror/dist/Face Mirror.app" ]; then
    echo "ERROR: Face Mirror.app not found"
    MISSING_APPS=1
fi
if [ ! -d "S2 Action Coder/dist/Action Coder.app" ]; then
    echo "ERROR: Action Coder.app not found"
    MISSING_APPS=1
fi
if [ ! -d "S3 Data Analysis/dist/Data Analysis.app" ]; then
    echo "ERROR: Data Analysis.app not found"
    MISSING_APPS=1
fi

if [ $MISSING_APPS -eq 1 ]; then
    echo "Build must have failed - check output above"
    exit 1
fi

# Remove old DMG if exists
if [ -f "$OUTPUT_DIR/$DMG_NAME" ]; then
    echo "Removing old DMG..."
    rm "$OUTPUT_DIR/$DMG_NAME"
fi

# Create temporary directory for DMG contents
TEMP_DIR=$(mktemp -d)
echo "Temporary directory: $TEMP_DIR"

# Create SplitFace folder in temp directory
SPLITFACE_FOLDER="$TEMP_DIR/SplitFace"
mkdir -p "$SPLITFACE_FOLDER"

# Copy all three apps to SplitFace folder
echo "Copying applications to SplitFace folder..."
cp -R "S1 Face Mirror/dist/Face Mirror.app" "$SPLITFACE_FOLDER/"
cp -R "S2 Action Coder/dist/Action Coder.app" "$SPLITFACE_FOLDER/"
cp -R "S3 Data Analysis/dist/Data Analysis.app" "$SPLITFACE_FOLDER/"

# Create symbolic link to Applications folder
echo "Creating Applications link..."
ln -s /Applications "$TEMP_DIR/Applications"

# Create DMG using hdiutil
echo "Creating disk image..."
hdiutil create \
    -volname "SplitFace v$VERSION" \
    -srcfolder "$TEMP_DIR" \
    -ov \
    -format UDZO \
    "$OUTPUT_DIR/$DMG_NAME" > /dev/null

# Clean up temp directory
rm -rf "$TEMP_DIR"

if [ $? -eq 0 ]; then
    echo "✓ Combined DMG created successfully!"
    echo "  Location: $OUTPUT_DIR/$DMG_NAME"
    echo "  Size: $(du -h "$OUTPUT_DIR/$DMG_NAME" | cut -f1)"
else
    echo "✗ DMG creation failed!"
    exit 1
fi

echo ""

echo "=========================================="
echo "✓ Build Complete!"
echo "=========================================="
echo ""
echo "Applications built:"
echo "  • S1 Face Mirror/dist/Face Mirror.app"
echo "  • S2 Action Coder/dist/Action Coder.app"
echo "  • S3 Data Analysis/dist/Data Analysis.app"
echo ""
echo "Distribution-ready installer:"
echo "  • $OUTPUT_DIR/$DMG_NAME ($(du -h "$OUTPUT_DIR/$DMG_NAME" 2>/dev/null | cut -f1))"
echo ""
echo "DMG Contents:"
echo "  SplitFace/"
echo "    ├── Face Mirror.app"
echo "    ├── Action Coder.app"
echo "    └── Data Analysis.app"
echo ""
echo "Ready for distribution:"
echo "  • Upload to GitHub Releases"
echo "  • Share via file hosting service"
echo "  • Distribute directly to users"
echo ""
echo "User installation (macOS):"
echo "  1. Download SplitFace-v$VERSION.dmg"
echo "  2. Double-click to mount"
echo "  3. Drag 'SplitFace' folder to Applications"
echo "  4. Apps will be in /Applications/SplitFace/"
echo "  5. Launch any app from /Applications/SplitFace/"
echo ""
echo "Target architecture: Apple Silicon (ARM64)"
echo ""
