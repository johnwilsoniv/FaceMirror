#!/bin/bash
#SBATCH --job-name=au_pipeline
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/au_pipeline_%j.out
#SBATCH --error=logs/au_pipeline_%j.err

# HPC AU Pipeline SLURM Job Script
# Usage: sbatch submit_au_job.sh --video input.mp4 [options]
#        sbatch submit_au_job.sh --video-list videos.txt --output-dir results/

# Load modules
module load python/3.11.13

# Set up environment
cd ~/pyfaceau
export PYTHONPATH=$PWD:$PWD/pyclnf:$PWD/pyfaceau:$PWD/pymtcnn:$PYTHONPATH

# Create logs directory
mkdir -p logs

# Print job info
echo "=========================================="
echo "HPC AU Pipeline Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo ""

# Run pipeline with auto-config
# --auto-config automatically sets optimal workers based on SLURM resources
python bigred200/hpc_optimized_pipeline.py --auto-config "$@"

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="
