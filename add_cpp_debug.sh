#!/bin/bash
# Add debug output to C++ OpenFace to match Python's debug info

cat << 'EOF'
We need to add debug output to C++ OpenFace at these locations:

1. Print initialization parameters:
   File: LandmarkDetectorModel.cpp, after params initialization
   Print: params_local, scale, rotation, translation, initial landmarks

2. Print patch scale being used:
   File: LandmarkDetectorModel.cpp, in optimization loop
   Print: which patch_scaling is being used for each window size

3. Print response map for landmark 36:
   File: CEN_patch_expert.cpp or CCNF_patch_expert.cpp
   Print/Save: response map for landmark 36 at iteration 0

Let's start by checking which scale C++ uses by default:
EOF

# Check OpenFace source for default window sizes and scale mapping
echo ""
echo "Checking OpenFace C++ source for window sizes and scales..."
echo ""

grep -n "window_size" /Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/LandmarkDetectorModel.cpp | head -20

echo ""
echo "Checking for scale selection logic..."
echo ""

grep -n "scale.*patch\|patch.*scale" /Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/LandmarkDetectorModel.cpp | head -20
