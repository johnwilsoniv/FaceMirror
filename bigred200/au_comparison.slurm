#!/bin/bash
#SBATCH --job-name=au_compare
#SBATCH --output=au_compare_%A_%a.out
#SBATCH --error=au_compare_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --array=0-12

# AU Comparison - Python FullPythonAUPipeline vs C++ OpenFace
# One video per array task, ALL frames
# Uses staggered initialization (5 seconds per task) to avoid memory spikes

set -e

module purge
module load python/3.12.11

cd "$HOME/pyfaceau"

# Set PYTHONPATH for all local packages
export PYTHONPATH="$PWD/pyclnf:$PWD/pymtcnn:$PWD/pyfaceau:$PWD/pyfhog:$PYTHONPATH"
export PYFACEAU_BASE="$PWD"

# Limit threads to avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Get video index from SLURM array task ID
VIDEO_INDEX=${SLURM_ARRAY_TASK_ID:-0}

# Stagger delay: 5 seconds per task to avoid simultaneous model loading
STAGGER_DELAY=5

echo "============================================"
echo "AU Comparison - Full Pipeline"
echo "============================================"
echo "Video index: $VIDEO_INDEX"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Stagger delay: $((VIDEO_INDEX * STAGGER_DELAY)) seconds"
echo "Start time: $(date)"
echo "PYTHONPATH: $PYTHONPATH"
echo "PYFACEAU_BASE: $PYFACEAU_BASE"
echo "============================================"

# Run AU comparison for this video
# Run AU comparison for all frames (no --n-frames limit)
python3 bigred200/au_comparison.py \
    --video-index $VIDEO_INDEX \
    --cpp-ref cpp_reference \
    --video-dir "S Data" \
    --stagger-delay $STAGGER_DELAY \
    --output-dir au_comparison_results

echo ""
echo "============================================"
echo "Finished AU comparison for video index: $VIDEO_INDEX"
echo "End time: $(date)"
echo "============================================"
