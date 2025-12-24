#!/bin/bash
#SBATCH --job-name=gen_nnclnf
#SBATCH --output=gen_nnclnf_%A_%a.out
#SBATCH --error=gen_nnclnf_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --array=0-110

# Generate NNCLNF training data from C++ ground truth
# 111 array tasks, one per video (both cohorts: 13 normal + 98 paralysis)
#
# Prerequisites:
#   - C++ ground truth must exist in cpp_groundtruth/
#   - Run generate_cpp_groundtruth.slurm first
#
# Output: HDF5 files in nnclnf_training_data/

set -e

module purge
module load python/3.12.11

cd "$HOME/pyfaceau"

export PYTHONPATH="$PWD:$PYTHONPATH"

# Configuration - both cohorts
CPP_GT_DIR="cpp_groundtruth"
VIDEO_DIRS=("S Data/Normal Cohort" "S Data/Paralysis Cohort")
OUTPUT_DIR="nnclnf_training_data"
PDM_FILE="weights/In-the-wild_aligned_PDM_68.txt"

echo "============================================"
echo "NNCLNF Training Data Generation"
echo "============================================"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "============================================"

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
CSV_FILE="$CPP_GT_DIR/${VIDEO_NAME}.csv"
OUTPUT_FILE="$OUTPUT_DIR/training_${VIDEO_NAME}.h5"

echo "Processing:"
echo "  Video: $VIDEO"
echo "  CSV: $CSV_FILE"
echo "  Output: $OUTPUT_FILE"

# Skip if already processed
if [ -f "$OUTPUT_FILE" ]; then
    echo "Already processed, skipping"
    exit 0
fi

# Check CSV exists (skip gracefully if not ready yet - allows parallel execution)
if [ ! -f "$CSV_FILE" ]; then
    echo "SKIP: C++ ground truth not ready yet: $CSV_FILE"
    echo "Will be processed when CSV is available"
    exit 0
fi

# Check CSV has pdmparams
if ! head -1 "$CSV_FILE" | grep -q "p_scale"; then
    echo "ERROR: C++ ground truth lacks PDM params: $CSV_FILE"
    echo "Regenerate with -pdmparams flag!"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run data generator
python3 -m nnclnf.data_generator \
    --video "$VIDEO" \
    --csv "$CSV_FILE" \
    --pdm "$PDM_FILE" \
    --output "$OUTPUT_FILE" \
    --no-mtcnn

# Verify output
if [ -f "$OUTPUT_FILE" ]; then
    python3 -c "
import h5py
with h5py.File('$OUTPUT_FILE', 'r') as f:
    n = f['images'].shape[0]
    print(f'Generated {n} samples')
"
else
    echo "ERROR: Output file not created"
    exit 1
fi

echo ""
echo "============================================"
echo "Complete!"
echo "End time: $(date)"
echo "============================================"
