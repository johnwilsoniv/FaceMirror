#!/bin/bash
# Create simple DMG installers using built-in macOS tools
# No external dependencies required (uses hdiutil)

set -e  # Exit on error

echo "=========================================="
echo "SplitFace v2.0.0 - Simple DMG Creator"
echo "=========================================="
echo ""

VERSION="2.0.0"
OUTPUT_DIR="DMG_Installers"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to create simple DMG
create_simple_dmg() {
    local app_name="$1"
    local app_path="$2"
    local dmg_name="$3"

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

    # Create temporary directory for DMG contents
    TEMP_DIR=$(mktemp -d)
    echo "Temporary directory: $TEMP_DIR"

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
        "$OUTPUT_DIR/$dmg_name"

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

# Create DMGs for all three applications
create_simple_dmg \
    "Face Mirror" \
    "S1 Face Mirror/dist/Face Mirror.app" \
    "SplitFace-FaceMirror-v$VERSION.dmg"

create_simple_dmg \
    "Action Coder" \
    "S2 Action Coder/dist/Action Coder.app" \
    "SplitFace-ActionCoder-v$VERSION.dmg"

create_simple_dmg \
    "Data Analysis" \
    "S3 Data Analysis/dist/Data Analysis.app" \
    "SplitFace-DataAnalysis-v$VERSION.dmg"

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
echo "Users can download these DMG files and:"
echo "  1. Double-click the DMG to mount it"
echo "  2. Drag the app to the Applications folder"
echo "  3. Done!"
echo ""
