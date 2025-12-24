#!/bin/bash
#SBATCH --job-name=test_cpp_gt
#SBATCH --output=test_cpp_gt_%j.out
#SBATCH --error=test_cpp_gt_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --account=r01984
#SBATCH --partition=debug

# DEBUG TEST: Run C++ OpenFace on ONE video to verify setup
#
# This version includes -pdmparams for training data generation
#
# Usage:
#   sbatch test_cpp_groundtruth_debug.slurm

set -e

# Load modules
module purge
module load boost/1.86.0
module load ffmpeg/7.7.1

# Set library paths (from working generate_cpp_reference.slurm)
export LD_LIBRARY_PATH="/N/soft/sles15sp6/boost/gnu/1.86.0/lib:$HOME/software/opencv_install/lib64:/N/soft/sles15sp6/ffmpeg/7.7.1/lib:/N/soft/sles15sp6/libjpeg/3.1.2/lib64:$LD_LIBRARY_PATH"

# OpenFace paths
OPENFACE_BIN="$HOME/software/OpenFace/build/bin/FeatureExtraction"
CECLM_MODEL="$HOME/software/OpenFace/lib/local/LandmarkDetector/model/main_ceclm_general.txt"

# Configuration
VIDEO_DIR="$HOME/pyfaceau/S Data/Paralysis Cohort"
OUTPUT_DIR="$HOME/pyfaceau/cpp_groundtruth_test"

echo "============================================"
echo "DEBUG TEST: C++ OpenFace Ground Truth"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "============================================"

# Check OpenFace binary exists
if [ ! -f "$OPENFACE_BIN" ]; then
    echo "ERROR: OpenFace binary not found at $OPENFACE_BIN"
    exit 1
fi
echo "OpenFace binary: $OPENFACE_BIN"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get first video (handle both .MOV and .mov)
VIDEO=$(ls "$VIDEO_DIR"/*.MOV "$VIDEO_DIR"/*.mov 2>/dev/null | head -1)

if [ -z "$VIDEO" ]; then
    echo "ERROR: No videos found in $VIDEO_DIR"
    exit 1
fi

# Strip extension (handle both .MOV and .mov)
VIDEO_NAME=$(basename "$VIDEO")
VIDEO_NAME="${VIDEO_NAME%.[Mm][Oo][Vv]}"
OUTPUT_CSV="$OUTPUT_DIR/${VIDEO_NAME}.csv"

echo "Processing:"
echo "  Video: $VIDEO"
echo "  Output: $OUTPUT_CSV"
echo "  Model: $CECLM_MODEL"
echo ""

# Run OpenFace (first 100 frames for quick test)
# Include -pdmparams for PDM parameters needed for NNCLNF training
echo "Running OpenFace FeatureExtraction (first 100 frames)..."
timeout 300 "$OPENFACE_BIN" \
    -f "$VIDEO" \
    -out_dir "$OUTPUT_DIR" \
    -mloc "$CECLM_MODEL" \
    -n_frames 100 \
    -2Dfp \
    -3Dfp \
    -pdmparams \
    -pose \
    -aus \
    -of "$VIDEO_NAME"

# Check output
if [ -f "$OUTPUT_CSV" ]; then
    FRAME_COUNT=$(wc -l < "$OUTPUT_CSV")
    echo ""
    echo "SUCCESS! Generated $((FRAME_COUNT - 1)) frames"
    echo ""

    # Verify pdmparams columns exist
    echo "Checking for PDM params columns..."
    if head -1 "$OUTPUT_CSV" | grep -q "p_scale"; then
        echo "PDM params found!"
        echo ""
        echo "Sample columns:"
        head -1 "$OUTPUT_CSV" | tr ',' '\n' | grep -E "^p_|^p_scale" | head -10
    else
        echo "WARNING: PDM params NOT found in output!"
        echo "Columns in CSV:"
        head -1 "$OUTPUT_CSV" | tr ',' '\n' | head -20
    fi
else
    echo "ERROR: Output file not created"
    exit 1
fi

echo ""
echo "============================================"
echo "DEBUG TEST PASSED!"
echo "End time: $(date)"
echo "============================================"
