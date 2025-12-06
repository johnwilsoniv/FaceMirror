#!/bin/bash
#SBATCH --job-name=au_batch
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/au_batch_%A_%a.out
#SBATCH --error=logs/au_batch_%A_%a.err

# HPC Batch AU Pipeline - SLURM Array Job Script
#
# Usage:
#   sbatch --array=0-99 submit_batch.sh videos.txt results/
#
# This processes videos in parallel using SLURM array jobs.
# Each array task processes one video sequentially (most efficient approach).
#
# Arguments:
#   $1 = video list file (one video path per line)
#   $2 = output directory for CSV files

VIDEO_LIST="$1"
OUTPUT_DIR="$2"

if [ -z "$VIDEO_LIST" ]; then
    echo "ERROR: Video list file required"
    echo "Usage: sbatch --array=0-N submit_batch.sh videos.txt results/"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="results"
fi

# Load modules
module load python/3.11.13

# Set up environment
cd ~/pyfaceau
export PYTHONPATH=$PWD:$PWD/pyclnf:$PWD/pyfaceau:$PWD/pymtcnn:$PYTHONPATH

# Create directories
mkdir -p logs "$OUTPUT_DIR"

# Print job info
echo "=========================================="
echo "HPC Batch AU Pipeline - Array Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Video list: $VIDEO_LIST"
echo "Output dir: $OUTPUT_DIR"
echo "Start time: $(date)"
echo ""

# Run pipeline for this array task
python bigred200/hpc_batch_pipeline.py \
    --video-list "$VIDEO_LIST" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="
