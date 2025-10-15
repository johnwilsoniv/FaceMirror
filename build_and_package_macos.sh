#!/bin/bash
# Complete build and packaging script for macOS
# Builds all applications and creates DMG installers in one step

set -e  # Exit on error

echo "=========================================="
echo "SplitFace v2.0.0 - Complete Build & Package"
echo "=========================================="
echo ""

# Step 1: Build all applications
echo "STEP 1: Building applications..."
echo "=========================================="
./build_macos.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed. Aborting packaging."
    exit 1
fi

echo ""
echo "=========================================="
echo "STEP 2: Creating DMG installers..."
echo "=========================================="
./create_dmg.sh

if [ $? -ne 0 ]; then
    echo "WARNING: DMG creation had errors, but apps are built."
    exit 1
fi

echo "=========================================="
echo "✓ Complete Build & Package Successful!"
echo "=========================================="
echo ""
echo "Ready for distribution:"
echo "  DMG_Installers/"
echo "    ├── SplitFace-FaceMirror-v2.0.0.dmg"
echo "    ├── SplitFace-ActionCoder-v2.0.0.dmg"
echo "    └── SplitFace-DataAnalysis-v2.0.0.dmg"
echo ""
echo "Users can download these DMG files and simply:"
echo "  1. Double-click the DMG"
echo "  2. Drag the app to Applications folder"
echo "  3. Done!"
echo ""
