#!/bin/bash
#SBATCH --job-name=cpp_reference
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --array=0-12                     # 13 videos (indices 0-12)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                # C++ OpenFace is single-threaded
#SBATCH --mem=16G                        # Plenty for video processing
#SBATCH --time=02:00:00                  # 2 hours per video (full length)
#SBATCH --output=cpp_ref_%A_%a.out
#SBATCH --error=cpp_ref_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jw411@iu.edu

# ============================================================================
# Generate C++ OpenFace Reference Data
#
# This script runs C++ OpenFace on all 12 test videos to generate
# ground truth landmarks and AUs for comparison with Python pipeline.
#
# Prerequisites:
#   - Run build_openface.sh first to compile OpenFace
#   - Videos must be accessible on Big Red
#
# Output:
#   - Per-video CSV with landmarks (x_0..x_67, y_0..y_67) and AUs (AU01_r..AU45_r)
#   - Stored in $OUTPUT_DIR/video_N/
#
# Usage:
#   sbatch generate_cpp_reference.slurm
# ============================================================================

set -e  # Exit on error

echo "============================================"
echo "C++ OpenFace Reference Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================"

# ============================================
# Configuration
# ============================================

# Video list (same order as array indices)
# NOTE: IMG_0422 added - important test video for accuracy verification
VIDEOS=(
    "S Data/Normal Cohort/IMG_0422.MOV"
    "S Data/Normal Cohort/IMG_0428.MOV"
    "S Data/Normal Cohort/IMG_0433.MOV"
    "S Data/Normal Cohort/IMG_0434.MOV"
    "S Data/Normal Cohort/IMG_0435.MOV"
    "S Data/Normal Cohort/IMG_0438.MOV"
    "S Data/Normal Cohort/IMG_0452.MOV"
    "S Data/Normal Cohort/IMG_0453.MOV"
    "S Data/Normal Cohort/IMG_0579.MOV"
    "S Data/Normal Cohort/IMG_0942.MOV"
    "S Data/Paralysis Cohort/IMG_0592.MOV"
    "S Data/Paralysis Cohort/IMG_0861.MOV"
    "S Data/Paralysis Cohort/IMG_1366.MOV"
)

# Paths
PROJECT_DIR="$HOME/pyfaceau"
VIDEO_BASE="$PROJECT_DIR"
OUTPUT_DIR="$PROJECT_DIR/cpp_reference"
OPENFACE_BIN="$HOME/software/OpenFace/build/bin/FeatureExtraction"

# Get video for this task
VIDEO_REL="${VIDEOS[$SLURM_ARRAY_TASK_ID]}"
VIDEO_PATH="$VIDEO_BASE/$VIDEO_REL"
VIDEO_NAME=$(basename "$VIDEO_REL" .MOV)
TASK_OUTPUT="$OUTPUT_DIR/video_$SLURM_ARRAY_TASK_ID"

echo "Processing video: $VIDEO_REL"
echo "Video name: $VIDEO_NAME"
echo "Output dir: $TASK_OUTPUT"

# ============================================
# Load modules
# ============================================
module purge
module load gcc/11.2.0
module load opencv/4.5.5

# ============================================
# Check prerequisites
# ============================================
if [ ! -f "$OPENFACE_BIN" ]; then
    echo "ERROR: OpenFace not found at $OPENFACE_BIN"
    echo "Please run build_openface.sh first"
    exit 1
fi

if [ ! -f "$VIDEO_PATH" ]; then
    echo "ERROR: Video not found at $VIDEO_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$TASK_OUTPUT"

# ============================================
# Run OpenFace
# ============================================
echo ""
echo "Running OpenFace..."
echo "Command: $OPENFACE_BIN -f \"$VIDEO_PATH\" -out_dir \"$TASK_OUTPUT\" -2Dfp -aus -pose -of \"$VIDEO_NAME\""

"$OPENFACE_BIN" \
    -f "$VIDEO_PATH" \
    -out_dir "$TASK_OUTPUT" \
    -2Dfp \
    -aus \
    -pose \
    -of "$VIDEO_NAME"

# ============================================
# Verify output
# ============================================
OUTPUT_CSV="$TASK_OUTPUT/${VIDEO_NAME}.csv"
if [ -f "$OUTPUT_CSV" ]; then
    FRAME_COUNT=$(wc -l < "$OUTPUT_CSV")
    echo ""
    echo "SUCCESS: Generated $((FRAME_COUNT - 1)) frames"
    echo "Output: $OUTPUT_CSV"

    # Show first few columns to verify
    echo ""
    echo "First 5 rows (selected columns):"
    head -5 "$OUTPUT_CSV" | cut -d',' -f1-5
else
    echo "ERROR: Output CSV not found at $OUTPUT_CSV"
    exit 1
fi

# ============================================
# Summary
# ============================================
echo ""
echo "============================================"
echo "Task $SLURM_ARRAY_TASK_ID Complete"
echo "Video: $VIDEO_NAME"
echo "Frames: $((FRAME_COUNT - 1))"
echo "Output: $OUTPUT_CSV"
echo "End time: $(date)"
echo "============================================"
