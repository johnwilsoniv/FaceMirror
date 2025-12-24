#!/bin/bash
#SBATCH --job-name=cpp_gt
#SBATCH --output=cpp_gt_%A_%a.out
#SBATCH --error=cpp_gt_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --array=0-110

# Generate C++ OpenFace ground truth for all videos (both cohorts)
# Includes -pdmparams for NNCLNF training data
#
# Usage:
#   sbatch generate_cpp_groundtruth.slurm
#
# Output: CSV files in ~/pyfaceau/cpp_groundtruth/

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

# Configuration - both cohorts
VIDEO_DIRS=("$HOME/pyfaceau/S Data/Normal Cohort" "$HOME/pyfaceau/S Data/Paralysis Cohort")
OUTPUT_DIR="$HOME/pyfaceau/cpp_groundtruth"

echo "============================================"
echo "C++ OpenFace Ground Truth Generation"
echo "============================================"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "============================================"

# Check OpenFace binary exists
if [ ! -f "$OPENFACE_BIN" ]; then
    echo "ERROR: OpenFace binary not found at $OPENFACE_BIN"
    echo "Run build_openface.sh first!"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get video list from both cohorts
VIDEOS=()
for DIR in "${VIDEO_DIRS[@]}"; do
    while IFS= read -r -d '' file; do
        VIDEOS+=("$file")
    done < <(find "$DIR" -maxdepth 1 \( -name "*.MOV" -o -name "*.mov" \) -print0 2>/dev/null | sort -z)
done

if [ ${#VIDEOS[@]} -eq 0 ]; then
    echo "ERROR: No videos found"
    exit 1
fi

echo "Found ${#VIDEOS[@]} videos"

# Get this task's video
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#VIDEOS[@]}" ]; then
    echo "Task ID exceeds video count, nothing to do"
    exit 0
fi

VIDEO="${VIDEOS[$SLURM_ARRAY_TASK_ID]}"
# Strip extension (handle both .MOV and .mov)
VIDEO_NAME=$(basename "$VIDEO")
VIDEO_NAME="${VIDEO_NAME%.[Mm][Oo][Vv]}"
OUTPUT_CSV="$OUTPUT_DIR/${VIDEO_NAME}.csv"

echo "Processing:"
echo "  Video: $VIDEO"
echo "  Output: $OUTPUT_CSV"
echo "  Model: $CECLM_MODEL"

# Skip if already processed
if [ -f "$OUTPUT_CSV" ]; then
    # Check if it has pdmparams
    if head -1 "$OUTPUT_CSV" | grep -q "p_scale"; then
        echo "Already processed with pdmparams, skipping"
        exit 0
    else
        echo "Existing file lacks pdmparams, reprocessing..."
        rm -f "$OUTPUT_CSV"
    fi
fi

# Run OpenFace
echo ""
echo "Running OpenFace FeatureExtraction..."
"$OPENFACE_BIN" \
    -f "$VIDEO" \
    -out_dir "$OUTPUT_DIR" \
    -mloc "$CECLM_MODEL" \
    -2Dfp \
    -3Dfp \
    -pdmparams \
    -pose \
    -aus \
    -of "$VIDEO_NAME"

# Check output
if [ -f "$OUTPUT_CSV" ]; then
    FRAME_COUNT=$(wc -l < "$OUTPUT_CSV")
    echo "Success! Generated $((FRAME_COUNT - 1)) frames"

    # Verify pdmparams
    if head -1 "$OUTPUT_CSV" | grep -q "p_scale"; then
        echo "PDM params verified!"
    else
        echo "WARNING: PDM params not found!"
    fi
else
    echo "ERROR: Output file not created"
    exit 1
fi

echo ""
echo "============================================"
echo "Complete!"
echo "End time: $(date)"
echo "============================================"
