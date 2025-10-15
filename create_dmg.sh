#!/bin/bash
# Create DMG installers for SplitFace applications
# Creates user-friendly disk images with drag-to-Applications interface

set -e  # Exit on error

echo "=========================================="
echo "SplitFace v2.0.0 - DMG Creator"
echo "=========================================="
echo ""

# Check if create-dmg is installed
if ! command -v create-dmg &> /dev/null; then
    echo "Installing create-dmg tool..."
    if command -v brew &> /dev/null; then
        brew install create-dmg
    else
        echo "ERROR: Homebrew not found. Please install create-dmg manually:"
        echo "  brew install create-dmg"
        echo "Or install Homebrew first: https://brew.sh"
        exit 1
    fi
fi

VERSION="2.0.0"
OUTPUT_DIR="DMG_Installers"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to create DMG
create_app_dmg() {
    local app_name="$1"
    local app_path="$2"
    local dmg_name="$3"
    local volume_name="$4"

    echo "=========================================="
    echo "Creating DMG for: $app_name"
    echo "=========================================="

    # Check if app exists
    if [ ! -d "$app_path" ]; then
        echo "ERROR: Application not found at $app_path"
        echo "Please build the application first using ./build_macos.sh"
        return 1
    fi

    # Remove old DMG if exists
    if [ -f "$OUTPUT_DIR/$dmg_name" ]; then
        echo "Removing old DMG..."
        rm "$OUTPUT_DIR/$dmg_name"
    fi

    # Create DMG with create-dmg
    echo "Creating disk image..."
    create-dmg \
        --volname "$volume_name" \
        --volicon "$app_path/Contents/Resources/AppIcon.icns" 2>/dev/null || true \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "$app_name.app" 175 190 \
        --hide-extension "$app_name.app" \
        --app-drop-link 425 185 \
        --eula "LICENSE.txt" 2>/dev/null || true \
        --background "dmg_background.png" 2>/dev/null || true \
        "$OUTPUT_DIR/$dmg_name" \
        "$app_path" \
        --skip-jenkins

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

# Create DMGs for all three applications
create_app_dmg \
    "Face Mirror" \
    "S1 Face Mirror/dist/Face Mirror.app" \
    "SplitFace-FaceMirror-v$VERSION.dmg" \
    "Face Mirror $VERSION"

create_app_dmg \
    "Action Coder" \
    "S2 Action Coder/dist/Action Coder.app" \
    "SplitFace-ActionCoder-v$VERSION.dmg" \
    "Action Coder $VERSION"

create_app_dmg \
    "Data Analysis" \
    "S3 Data Analysis/dist/Data Analysis.app" \
    "SplitFace-DataAnalysis-v$VERSION.dmg" \
    "Data Analysis $VERSION"

echo "=========================================="
echo "DMG Creation Complete!"
echo "=========================================="
echo ""
echo "Installers created in: $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR"/*.dmg 2>/dev/null || echo "No DMG files found"
echo ""
echo "Distribution files:"
echo "  • $OUTPUT_DIR/SplitFace-FaceMirror-v$VERSION.dmg"
echo "  • $OUTPUT_DIR/SplitFace-ActionCoder-v$VERSION.dmg"
echo "  • $OUTPUT_DIR/SplitFace-DataAnalysis-v$VERSION.dmg"
echo ""
echo "Users can now double-click the DMG, then drag the app to Applications!"
echo ""
