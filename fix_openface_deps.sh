#!/bin/bash
# Fix OpenFace library dependencies by creating version symlinks

echo "Creating library version symlinks for OpenFace..."

# OpenEXR libraries
cd /opt/homebrew/opt/openexr/lib
ln -sf "libOpenEXR-3_4.33.dylib" "libOpenEXR-3_3.32.dylib"
ln -sf "libOpenEXRCore-3_4.33.dylib" "libOpenEXRCore-3_3.32.dylib"
ln -sf "libOpenEXRUtil-3_4.33.dylib" "libOpenEXRUtil-3_3.32.dylib"
ln -sf "libIlmThread-3_4.33.dylib" "libIlmThread-3_3.32.dylib"
ln -sf "libIex-3_4.33.dylib" "libIex-3_3.32.dylib"

# Imath libraries
cd /opt/homebrew/opt/imath/lib
ln -sf "libImath-3_2.30.dylib" "libImath-3_1.29.dylib"

# OpenVINO libraries
cd /opt/homebrew/Cellar/openvino/2025.3.0_3/lib
ln -sf "libopenvino.2530.dylib" "libopenvino.2500.dylib"

# FFmpeg libraries
cd /opt/homebrew/opt/ffmpeg/lib
ln -sf "libavcodec.62.dylib" "libavcodec.61.dylib"
ln -sf "libavformat.62.dylib" "libavformat.61.dylib"
ln -sf "libavutil.60.dylib" "libavutil.59.dylib"
ln -sf "libswscale.9.dylib" "libswscale.8.dylib"
ln -sf "libswresample.5.dylib" "libswresample.5.dylib"

echo "✓ Symlinks created"
echo "Testing OpenFace binary..."

/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction -help
