#!/bin/bash
#
# Build script for FaceMirror applications
# Creates a DMG containing all three apps in a FaceMirror folder
#
# Usage:
#   ./build_dmg.sh
#
# Requirements:
#   - Python 3.10+ with PyInstaller installed
#   - All dependencies installed (pyfaceau, pyclnf, pymtcnn, etc.)
#   - FFmpeg binary in S2 Action Coder/bin/ (optional)
#

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
DIST_DIR="$SCRIPT_DIR/dist"
DMG_NAME="FaceMirror"
DMG_VERSION="1.0.0"
DMG_FILENAME="${DMG_NAME}-${DMG_VERSION}.dmg"
VOLUME_NAME="FaceMirror ${DMG_VERSION}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_status() {
    echo -e "${GREEN}[OK]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo_step() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
}

# Check for PyInstaller
if ! command -v pyinstaller &> /dev/null; then
    echo_error "PyInstaller not found. Install with: pip install pyinstaller"
    exit 1
fi

echo_step "Building FaceMirror Applications"
echo "Version: $DMG_VERSION"
echo "Output: $DMG_FILENAME"

# Clean previous builds
echo_step "Cleaning previous builds"
rm -rf "$DIST_DIR"/*.app 2>/dev/null || true
rm -rf "$BUILD_DIR" 2>/dev/null || true
mkdir -p "$DIST_DIR"

# Build S1 Face Mirror
echo_step "Building S1 Face Mirror"
cd "$SCRIPT_DIR/S1_FaceMirror"
pyinstaller --clean --noconfirm Face_Mirror.spec
if [ -d "dist/S1 Face Mirror.app" ]; then
    cp -R "dist/S1 Face Mirror.app" "$DIST_DIR/"
    echo_status "S1 Face Mirror built successfully"
else
    echo_error "S1 Face Mirror build failed"
    exit 1
fi

# Build S2 Action Coder
echo_step "Building S2 Action Coder"
cd "$SCRIPT_DIR/S2 Action Coder"

# Check for FFmpeg
if [ ! -f "bin/ffmpeg" ]; then
    echo_warning "FFmpeg not found in bin/. App will require system FFmpeg."
fi

pyinstaller --clean --noconfirm Action_Coder.spec
if [ -d "dist/S2 Action Coder.app" ]; then
    cp -R "dist/S2 Action Coder.app" "$DIST_DIR/"
    echo_status "S2 Action Coder built successfully"
else
    echo_error "S2 Action Coder build failed"
    exit 1
fi

# Build S3 Data Analysis
echo_step "Building S3 Data Analysis"
cd "$SCRIPT_DIR/S3 Data Analysis"
pyinstaller --clean --noconfirm Paralysis_Analyzer.spec
if [ -d "dist/S3 Data Analysis.app" ]; then
    cp -R "dist/S3 Data Analysis.app" "$DIST_DIR/"
    echo_status "S3 Data Analysis built successfully"
else
    echo_error "S3 Data Analysis build failed"
    exit 1
fi

# Create DMG
echo_step "Creating DMG"
cd "$SCRIPT_DIR"

# Remove old DMG if exists
rm -f "$DMG_FILENAME" 2>/dev/null || true

# Create temporary DMG directory
DMG_TEMP="$BUILD_DIR/dmg_temp"
rm -rf "$DMG_TEMP"
mkdir -p "$DMG_TEMP"

# Create FaceMirror folder that users will drag to Applications
mkdir -p "$DMG_TEMP/FaceMirror"

# Copy apps to FaceMirror folder (strip extended attributes during copy)
cp -R "$DIST_DIR/S1 Face Mirror.app" "$DMG_TEMP/FaceMirror/"
cp -R "$DIST_DIR/S2 Action Coder.app" "$DMG_TEMP/FaceMirror/"
cp -R "$DIST_DIR/S3 Data Analysis.app" "$DMG_TEMP/FaceMirror/"

# Clean extended attributes from apps BEFORE creating symlink
find "$DMG_TEMP" -exec xattr -c {} \; 2>/dev/null || true

# Create Applications symlink (must be done fresh, no extended attrs)
/bin/ln -s /Applications "$DMG_TEMP/Applications"

# Create README at root level (visible when DMG opens)
cat > "$DMG_TEMP/README.txt" << 'EOF'
FaceMirror - Facial Analysis Suite
===================================

INSTALLATION:
   Drag the "FaceMirror" folder to "Applications"

This folder contains three applications for facial analysis:

1. S1 Face Mirror
   - Processes facial videos with face mirroring
   - Extracts Action Unit (AU) data using pyfaceau
   - Outputs CSV files compatible with OpenFace format

2. S2 Action Coder
   - Video action coding with Whisper transcription
   - Frame-by-frame action annotation
   - Exports coded data for analysis

3. S3 Data Analysis
   - Detects facial paralysis from AU data
   - Analyzes upper, mid, and lower face zones
   - Machine learning-based severity classification

Requirements:
   - macOS 10.15 (Catalina) or later
   - Apple Silicon (M1/M2/M3) or Intel Mac

First Launch:
   Right-click each app and select "Open" the first time
   (required for unsigned applications)

Citation:
   Wilson IV, J., Rosenberg, J., Gray, M. L., & Razavi, C. R. (2025).
   A split-face computer vision/machine learning assessment of facial
   paralysis using facial action units. Facial Plastic Surgery &
   Aesthetic Medicine. https://doi.org/10.1177/26893614251394382

License:
   CC BY-NC 4.0 - Free for non-commercial use with attribution.

EOF

# Create DMG using hdiutil
echo "Creating DMG image..."
hdiutil create -volname "$VOLUME_NAME" \
    -srcfolder "$DMG_TEMP" \
    -ov -format UDZO \
    "$DMG_FILENAME"

# Cleanup
rm -rf "$DMG_TEMP"

# Verify DMG
if [ -f "$DMG_FILENAME" ]; then
    DMG_SIZE=$(du -h "$DMG_FILENAME" | cut -f1)
    echo_status "DMG created successfully: $DMG_FILENAME ($DMG_SIZE)"
else
    echo_error "DMG creation failed"
    exit 1
fi

echo_step "Build Complete"
echo "Applications built:"
echo "  - S1 Face Mirror.app"
echo "  - S2 Action Coder.app"
echo "  - S3 Data Analysis.app"
echo ""
echo "DMG: $DMG_FILENAME"
echo ""
echo "To install, open the DMG and drag the FaceMirror folder to Applications."
