#!/bin/bash
#SBATCH --job-name=gen_bbox
#SBATCH --output=gen_bbox_%A_%a.out
#SBATCH --error=gen_bbox_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=r01984
#SBATCH --partition=debug
#SBATCH --array=0-3

# DEBUG: Generate training data with BBOX alignment
# 4 array tasks, each processing 3 videos = 12 total videos
# This uses the SAME bbox-based alignment as NNCLNF inference

set -e

module purge
module load python/3.12.11

cd "$HOME/pyfaceau"

# Set PYTHONPATH for all local packages
# Note: $PWD must be first so "from pyfaceau import ..." works
export PYTHONPATH="$PWD:$PWD/pyclnf:$PWD/pymtcnn:$PWD/pyfhog:$PYTHONPATH"
export PYFACEAU_BASE="$PWD"

# Limit threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Each task processes 3 videos
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
VIDEOS_PER_TASK=3
START_INDEX=$((TASK_ID * VIDEOS_PER_TASK))

echo "============================================"
echo "BBOX Training Data Generation (DEBUG)"
echo "============================================"
echo "Task ID: $TASK_ID"
echo "Processing videos: $START_INDEX to $((START_INDEX + VIDEOS_PER_TASK - 1))"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "============================================"

for i in $(seq 0 $((VIDEOS_PER_TASK - 1))); do
    VIDEO_INDEX=$((START_INDEX + i))
    echo ""
    echo "Processing video index: $VIDEO_INDEX"

    python3 bigred200/generate_training_data.py \
        --video-index $VIDEO_INDEX \
        --video-dir "S Data" \
        --output-dir training_data_bbox \
        --stagger-delay 1 || echo "Video $VIDEO_INDEX failed, continuing..."
done

echo ""
echo "============================================"
echo "Finished task $TASK_ID (videos $START_INDEX-$((START_INDEX + VIDEOS_PER_TASK - 1)))"
echo "End time: $(date)"
echo "============================================"
