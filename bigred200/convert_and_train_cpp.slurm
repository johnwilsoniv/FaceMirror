#!/bin/bash
#SBATCH --job-name=convert_cpp
#SBATCH --output=convert_cpp_%j.out
#SBATCH --error=convert_cpp_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=r01984
#SBATCH --partition=general

# Convert C++ training data to expected format

set -e

module purge
module load python/3.11.13

cd "$HOME/pyfaceau"

export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Converting C++ Training Data"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "============================================"

python convert_cpp_to_training_format.py \
    --input cpp_training_merged.h5 \
    --output cpp_training_converted.h5

echo ""
echo "============================================"
echo "Conversion complete!"
echo "End time: $(date)"
echo "============================================"
