#!/bin/bash
#SBATCH --job-name=cpp_training_data
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-110
#SBATCH --output=logs/cpp_training_%a.out
#SBATCH --error=logs/cpp_training_%a.err

# Generate training data using C++ OpenFace landmarks (ground truth)
# This creates training data without "baking in" pyCLNF landmark errors

echo "=============================================="
echo "C++ Training Data Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================="

# Setup environment
module purge
module load python/3.11.13
module load boost/1.86.0
module load ffmpeg/7.7.1
module load libjpeg/3.1.2

cd /N/u/jw411/BigRed200/pyfaceau

# Set paths
export PYFACEAU_BASE=/N/u/jw411/BigRed200/pyfaceau
export PYTHONPATH="$PYFACEAU_BASE/pyfaceau:$PYFACEAU_BASE/pymtcnn:$PYFACEAU_BASE/pyclnf:$PYTHONPATH"
export OPENFACE_BIN="/N/u/jw411/BigRed200/software/OpenFace/build/bin/FeatureExtraction"

# Limit threads to avoid contention
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Output directory
OUTPUT_DIR="cpp_training_data"
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Run the generator for this video index
python generate_cpp_training_data_bigred.py \
    --video-index $SLURM_ARRAY_TASK_ID \
    --video-dir "S Data" \
    --output-dir $OUTPUT_DIR \
    --openface $OPENFACE_BIN \
    --no-visualize

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
