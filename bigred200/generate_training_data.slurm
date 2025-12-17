#!/bin/bash
#SBATCH --job-name=gen_train
#SBATCH --output=gen_train_%A_%a.out
#SBATCH --error=gen_train_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --array=0-110

# Training Data Generation - All 111 videos (98 Paralysis + 13 Normal)
# Each array task processes one video
# Staggered initialization to avoid memory spikes

set -e

module purge
module load python/3.12.11

cd "$HOME/pyfaceau"

# Set PYTHONPATH for all local packages
export PYTHONPATH="$PWD/pyclnf:$PWD/pymtcnn:$PWD/pyfaceau:$PWD/pyfhog:$PYTHONPATH"
export PYFACEAU_BASE="$PWD"

# Limit threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

VIDEO_INDEX=${SLURM_ARRAY_TASK_ID:-0}
STAGGER_DELAY=3  # 3 seconds between jobs in same group

echo "============================================"
echo "Training Data Generation"
echo "============================================"
echo "Video index: $VIDEO_INDEX"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"
echo "============================================"

python3 bigred200/generate_training_data.py \
    --video-index $VIDEO_INDEX \
    --video-dir "S Data" \
    --output-dir training_data_new \
    --stagger-delay $STAGGER_DELAY

echo ""
echo "============================================"
echo "Finished video index: $VIDEO_INDEX"
echo "End time: $(date)"
echo "============================================"
