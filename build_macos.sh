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

# Function to create DMG installer
create_dmg() {
    local app_name="$1"
    local app_path="$2"
    local dmg_name="$3"

    echo "=========================================="
    echo "Creating DMG: $app_name"
    echo "=========================================="

    # Check if app exists
    if [ ! -d "$app_path" ]; then
        echo "ERROR: Application not found at $app_path"
        echo "Build must have failed - check output above"
        return 1
    fi

    # Remove old DMG if exists
    if [ -f "$OUTPUT_DIR/$dmg_name" ]; then
        echo "Removing old DMG..."
        rm "$OUTPUT_DIR/$dmg_name"
    fi

    # Create temporary directory for DMG contents
    TEMP_DIR=$(mktemp -d)

    # Copy app to temp directory
    echo "Copying application..."
    cp -R "$app_path" "$TEMP_DIR/"

    # Create symbolic link to Applications folder
    echo "Creating Applications link..."
    ln -s /Applications "$TEMP_DIR/Applications"

    # Create DMG using hdiutil
    echo "Creating disk image..."
    hdiutil create \
        -volname "$app_name v$VERSION" \
        -srcfolder "$TEMP_DIR" \
        -ov \
        -format UDZO \
        "$OUTPUT_DIR/$dmg_name" > /dev/null

    # Clean up temp directory
    rm -rf "$TEMP_DIR"

    if [ $? -eq 0 ]; then
        echo "✓ DMG created successfully!"
        echo "  Location: $OUTPUT_DIR/$dmg_name"
        echo "  Size: $(du -h "$OUTPUT_DIR/$dmg_name" | cut -f1)"
    else
        echo "✗ DMG creation failed!"
        return 1
    fi

    echo ""
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
echo "STEP 2: Creating DMG Installers"
echo "=========================================="
echo ""

# Create DMGs for all three applications
create_dmg \
    "Face Mirror" \
    "S1 Face Mirror/dist/Face Mirror.app" \
    "SplitFace-FaceMirror-v$VERSION.dmg"

create_dmg \
    "Action Coder" \
    "S2 Action Coder/dist/Action Coder.app" \
    "SplitFace-ActionCoder-v$VERSION.dmg"

create_dmg \
    "Data Analysis" \
    "S3 Data Analysis/dist/Data Analysis.app" \
    "SplitFace-DataAnalysis-v$VERSION.dmg"

echo "=========================================="
echo "✓ Build Complete!"
echo "=========================================="
echo ""
echo "Applications built:"
echo "  • S1 Face Mirror/dist/Face Mirror.app"
echo "  • S2 Action Coder/dist/Action Coder.app"
echo "  • S3 Data Analysis/dist/Data Analysis.app"
echo ""
echo "Distribution-ready DMG installers:"
ls -lh "$OUTPUT_DIR"/*.dmg 2>/dev/null | awk '{print "  • " $9 " (" $5 ")"}'
echo ""
echo "Ready for distribution:"
echo "  • Upload to GitHub Releases"
echo "  • Share via file hosting service"
echo "  • Distribute directly to users"
echo ""
echo "User installation (macOS):"
echo "  1. Download DMG file"
echo "  2. Double-click to mount"
echo "  3. Drag app to Applications"
echo "  4. Launch from Applications folder"
echo ""
echo "Target architecture: Apple Silicon (ARM64)"
echo ""
