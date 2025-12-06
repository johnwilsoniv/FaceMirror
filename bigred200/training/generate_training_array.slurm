#!/bin/bash
#SBATCH --job-name=gen_training
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --array=0-110%80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=bigred200/output/logs/train_%A_%a.out
#SBATCH --error=bigred200/output/logs/train_%A_%a.err
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=jw411@iu.edu

# ============================================
# Training Data Generation - SLURM Array Job
# Each task processes one video to HDF5
# ============================================

echo "=========================================="
echo "Training Data Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Load modules
module purge
module load python/3.12.11

# Activate virtual environment
source $HOME/pyfaceau_env/bin/activate

# Set working directory
cd $HOME/pyfaceau

# Set environment
export PYTHONPATH="$PWD/pyclnf:$PWD/pymtcnn:$PWD/pyfaceau:$PWD/pyfhog:$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Paths
MANIFEST="bigred200/config/video_manifest.txt"
OUTPUT_DIR="bigred200/output/per_video"
CHECKPOINT_DIR="bigred200/output/checkpoints"

# Get video path from manifest
VIDEO_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")

if [ -z "$VIDEO_PATH" ]; then
    echo "ERROR: No video found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Processing: $VIDEO_PATH"

# Output file
OUTPUT_FILE="${OUTPUT_DIR}/video_$(printf '%05d' $SLURM_ARRAY_TASK_ID).h5"
CHECKPOINT_FILE="${CHECKPOINT_DIR}/video_$(printf '%05d' $SLURM_ARRAY_TASK_ID).done"

# Check if already completed (resume support)
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Task already completed. Skipping."
    echo "Checkpoint: $CHECKPOINT_FILE"
    exit 0
fi

# Create directories
mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

# Run processing
python bigred200/training/process_single_video_hdf5.py \
    "$VIDEO_PATH" \
    --output "$OUTPUT_FILE" \
    --skip-frames 0 \
    --min-quality 0.5

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    # Mark as completed
    echo "$(date)" > "$CHECKPOINT_FILE"
    echo ""
    echo "SUCCESS: $VIDEO_PATH"
    echo "Output: $OUTPUT_FILE"
else
    echo ""
    echo "FAILED: $VIDEO_PATH (exit code $EXIT_CODE)"
fi

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE
