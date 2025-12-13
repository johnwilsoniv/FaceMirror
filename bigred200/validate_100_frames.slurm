#!/bin/bash
#SBATCH --job-name=validate_100
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=validate_%j.out
#SBATCH --error=validate_%j.err

# Quick 100-frame validation test
# Run this BEFORE full video processing to verify pipeline works

set -e

echo "============================================"
echo "100-Frame Validation Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================"

# Load modules
module purge
module load python/3.12.11

# Setup
cd "$HOME/pyfaceau"
export PYTHONPATH="$PWD/pyclnf:$PWD/pymtcnn:$PWD/pyfaceau:$PWD/pyfhog:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Run validation - use SLURM_ARRAY_TASK_ID for array jobs, default 0 for single job
VIDEO_INDEX=${SLURM_ARRAY_TASK_ID:-0}
echo "Processing video index: $VIDEO_INDEX"
python -u bigred200/validate_100_frames.py --video-index $VIDEO_INDEX --n-frames 100

echo ""
echo "============================================"
echo "Validation Complete"
echo "End time: $(date)"
echo "============================================"
