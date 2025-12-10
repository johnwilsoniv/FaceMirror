#!/bin/bash
#SBATCH --job-name=py_vs_cpp
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --array=0-12                     # 13 videos (indices 0-12)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                # Sequential processing (CLNF bottleneck)
#SBATCH --mem=32G
#SBATCH --time=02:00:00                  # 2 hours per video (full length)
#SBATCH --output=compare_%A_%a.out
#SBATCH --error=compare_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jw411@iu.edu

# ============================================================================
# Python vs C++ Accuracy Comparison
#
# This script:
# 1. Processes each video with the Python pyfaceau pipeline
# 2. Compares against C++ OpenFace reference
# 3. Reports landmark accuracy and AU correlation metrics
#
# Prerequisites:
#   - Run generate_cpp_reference.slurm first to create C++ ground truth
#
# Output:
#   - Per-video comparison JSON in comparison_results/
#   - Python pipeline CSV in comparison_results/
#
# Usage:
#   sbatch compare_accuracy.slurm
# ============================================================================

set -e  # Exit on error

echo "============================================"
echo "Python vs C++ Accuracy Comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================"

# ============================================
# Configuration
# ============================================

# Paths
PROJECT_DIR="$HOME/pyfaceau"
OUTPUT_DIR="$PROJECT_DIR/comparison_results/video_$SLURM_ARRAY_TASK_ID"

# ============================================
# Load modules
# ============================================
module purge
module load python/3.12.11

# ============================================
# Set up Python environment
# ============================================
cd "$PROJECT_DIR"

export PYTHONPATH="$PWD/pyclnf:$PWD/pymtcnn:$PWD/pyfaceau:$PWD/pyfhog:$PYTHONPATH"

# Limit threads (critical for HPC - avoids thread affinity conflicts)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export ORT_NUM_THREADS=1

echo "PYTHONPATH set"
echo "Project dir: $PROJECT_DIR"
echo "Output dir: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ============================================
# Run comparison
# ============================================
echo ""
echo "Running Python vs C++ comparison..."

python bigred200/compare_accuracy.py \
    --video-index "$SLURM_ARRAY_TASK_ID" \
    --output "$OUTPUT_DIR"

# ============================================
# Summary
# ============================================
echo ""
echo "============================================"
echo "Task $SLURM_ARRAY_TASK_ID Complete"
echo "Results: $OUTPUT_DIR"
echo "End time: $(date)"
echo "============================================"
