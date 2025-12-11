#!/bin/bash
#SBATCH --job-name=au_zone_val
#SBATCH --output=au_zone_%A_%a.out
#SBATCH --error=au_zone_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --array=0-12

# AU Zone Validation - Full pipeline with AU correlations
# One video per array task, 100 frames each

set -e

module purge
module load python/3.12.11

cd "$HOME/pyfaceau"

# Match the working validate_100_frames.slurm PYTHONPATH
export PYTHONPATH="$PWD/pyclnf:$PWD/pymtcnn:$PWD/pyfaceau:$PWD/pyfhog:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Get video index from SLURM array task ID
VIDEO_INDEX=${SLURM_ARRAY_TASK_ID:-0}

echo "============================================"
echo "AU Zone Validation - Full Pipeline"
echo "============================================"
echo "Video index: $VIDEO_INDEX"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"
echo "PYTHONPATH: $PYTHONPATH"
echo "============================================"

# Run AU zone validation for this video
python3 bigred200/au_zone_validation.py \
    --video-index $VIDEO_INDEX \
    --n-frames 100 \
    --cpp-ref cpp_reference \
    --video-dir "S Data"

echo ""
echo "============================================"
echo "Finished AU zone validation for video index: $VIDEO_INDEX"
echo "End time: $(date)"
echo "============================================"
