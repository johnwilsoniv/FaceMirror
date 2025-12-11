#!/bin/bash
#SBATCH --job-name=zone_analysis
#SBATCH --output=zone_analysis_%A_%a.out
#SBATCH --error=zone_analysis_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --array=0-12

# Zone analysis - one video per array task

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

echo "Starting zone analysis for video index: $VIDEO_INDEX"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
date
echo "PYTHONPATH: $PYTHONPATH"

# Run zone analysis for this video
python3 bigred200/zone_analysis.py \
    --video-index $VIDEO_INDEX \
    --n-frames 100 \
    --cpp-ref cpp_reference \
    --video-dir "S Data"

echo "Finished zone analysis for video index: $VIDEO_INDEX"
date
