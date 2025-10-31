#!/bin/bash
# Apply C++ Debug Patch and Rebuild OpenFace

set -e  # Exit on error

OPENFACE_DIR="/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace"
TARGET_FILE="$OPENFACE_DIR/lib/local/FaceAnalyser/src/Face_utils.cpp"
BACKUP_FILE="${TARGET_FILE}.backup"

echo "============================================"
echo "OpenFace C++ Debug Instrumentation"
echo "============================================"

# Backup original file
if [ ! -f "$BACKUP_FILE" ]; then
    echo "[1/5] Creating backup of Face_utils.cpp..."
    cp "$TARGET_FILE" "$BACKUP_FILE"
    echo "  ✓ Backup created: Face_utils.cpp.backup"
else
    echo "[1/5] Backup already exists, skipping..."
fi

# Check if already patched
if grep -q "DEBUG OUTPUT - Compare with Python" "$TARGET_FILE"; then
    echo "[2/5] File already patched, skipping modification..."
else
    echo "[2/5] Applying debug patch..."

    # Create the debug code
    cat > /tmp/debug_block.txt << 'EOF'

		// DEBUG OUTPUT - Compare with Python
		static int debug_frame_count = 0;
		debug_frame_count++;

		// Only print for frames we're testing (1, 493, 617, 863)
		if (debug_frame_count == 1 || debug_frame_count == 493 || debug_frame_count == 617 || debug_frame_count == 863) {
			std::cout << "\n=== DEBUG Frame " << debug_frame_count << " ===" << std::endl;

			// Print first 3 source landmarks (after rigid extraction)
			std::cout << "Source landmarks (first 3):" << std::endl;
			for (int i = 0; i < std::min(3, source_landmarks.cols); i++) {
				std::cout << "  [" << i << "]: ("
						  << source_landmarks(0, i) << ", "
						  << source_landmarks(1, i) << ")" << std::endl;
			}

			// Print first 3 destination landmarks
			std::cout << "Dest landmarks (first 3):" << std::endl;
			for (int i = 0; i < std::min(3, destination_landmarks.cols); i++) {
				std::cout << "  [" << i << "]: ("
						  << destination_landmarks(0, i) << ", "
						  << destination_landmarks(1, i) << ")" << std::endl;
			}

			// Print scale_rot_matrix
			std::cout << "Scale-rot matrix:" << std::endl;
			std::cout << "  [" << scale_rot_matrix(0,0) << ", " << scale_rot_matrix(0,1) << "]" << std::endl;
			std::cout << "  [" << scale_rot_matrix(1,0) << ", " << scale_rot_matrix(1,1) << "]" << std::endl;

			// Compute and print rotation angle
			float angle_rad = std::atan2(scale_rot_matrix(1,0), scale_rot_matrix(0,0));
			float angle_deg = angle_rad * 180.0f / 3.14159265359f;
			std::cout << "Rotation angle: " << angle_deg << "°" << std::endl;

			// Print params_global
			std::cout << "params_global:" << std::endl;
			std::cout << "  scale=" << params_global[0]
					  << " rx=" << params_global[1]
					  << " ry=" << params_global[2]
					  << " rz=" << params_global[3]
					  << " tx=" << params_global[4]
					  << " ty=" << params_global[5] << std::endl;

			std::cout << "==========================\n" << std::endl;
		}
EOF

    # Find line number of scale_rot_matrix = ...
    LINE_NUM=$(grep -n "scale_rot_matrix = Utilities::AlignShapesWithScale" "$TARGET_FILE" | head -1 | cut -d: -f1)

    if [ -z "$LINE_NUM" ]; then
        echo "  ✗ ERROR: Could not find insertion point"
        exit 1
    fi

    echo "  Found insertion point at line $LINE_NUM"

    # Insert debug code after that line
    head -n $LINE_NUM "$TARGET_FILE" > /tmp/face_utils_new.cpp
    cat /tmp/debug_block.txt >> /tmp/face_utils_new.cpp
    tail -n +$((LINE_NUM + 1)) "$TARGET_FILE" >> /tmp/face_utils_new.cpp

    # Replace original
    mv /tmp/face_utils_new.cpp "$TARGET_FILE"

    echo "  ✓ Debug patch applied"
fi

# Rebuild OpenFace
echo "[3/5] Rebuilding OpenFace..."
cd "$OPENFACE_DIR/build"

# Clean and rebuild
make clean > /dev/null 2>&1 || true
echo "  Compiling... (this may take a few minutes)"
make -j$(sysctl -n hw.ncpu) 2>&1 | grep -i "error\|warning\|%\|built" || true

if [ $? -eq 0 ]; then
    echo "  ✓ Build successful"
else
    echo "  ✗ Build failed - check output above"
    exit 1
fi

echo "[4/5] Verifying binary..."
if [ -f "$OPENFACE_DIR/build/bin/FeatureExtraction" ]; then
    echo "  ✓ FeatureExtraction binary exists"
else
    echo "  ✗ Binary not found!"
    exit 1
fi

echo "[5/5] Setup complete!"
echo ""
echo "============================================"
echo "Next Steps:"
echo "============================================"
echo "1. Run instrumented FeatureExtraction:"
echo "   cd \"$OPENFACE_DIR/build/bin\""
echo "   ./FeatureExtraction -f /path/to/IMG_0942_left_mirrored.mp4 2>&1 | tee debug_output.txt"
echo ""
echo "2. Look for '=== DEBUG Frame N ===' in output"
echo ""
echo "3. Compare to Python values using compare_cpp_python_debug.py"
echo ""
echo "To restore original:"
echo "   cp \"$BACKUP_FILE\" \"$TARGET_FILE\""
echo "   cd \"$OPENFACE_DIR/build\" && make"
echo "============================================"
