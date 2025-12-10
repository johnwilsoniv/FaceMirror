#!/bin/bash
#SBATCH --job-name=pyfaceau_batch
#SBATCH --account=r01984
#SBATCH --partition=general              # Use 'gpu' partition for GPU
#SBATCH --array=0-12                     # Process 13 videos (indices 0-12)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                # Sequential (CLNF is bottleneck)
#SBATCH --mem=32G
#SBATCH --time=02:00:00                  # 2 hours per video (full length)
#SBATCH --output=video_%A_%a.out
#SBATCH --error=video_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jw411@iu.edu

# ============================================
# Parallel Video Processing on Big Red 200
# Each array task processes one video
# ============================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Video list (same order as array indices)
# NOTE: IMG_0422 added - important test video for accuracy verification
VIDEOS=(
    "S Data/Normal Cohort/IMG_0422.MOV"
    "S Data/Normal Cohort/IMG_0428.MOV"
    "S Data/Normal Cohort/IMG_0433.MOV"
    "S Data/Normal Cohort/IMG_0434.MOV"
    "S Data/Normal Cohort/IMG_0435.MOV"
    "S Data/Normal Cohort/IMG_0438.MOV"
    "S Data/Normal Cohort/IMG_0452.MOV"
    "S Data/Normal Cohort/IMG_0453.MOV"
    "S Data/Normal Cohort/IMG_0579.MOV"
    "S Data/Normal Cohort/IMG_0942.MOV"
    "S Data/Paralysis Cohort/IMG_0592.MOV"
    "S Data/Paralysis Cohort/IMG_0861.MOV"
    "S Data/Paralysis Cohort/IMG_1366.MOV"
)

# Get video for this task
VIDEO_PATH="${VIDEOS[$SLURM_ARRAY_TASK_ID]}"
echo "Processing video: $VIDEO_PATH"

# Load modules
module purge
module load python/3.12.11

# Set working directory
cd $HOME/pyfaceau

# Set PYTHONPATH
export PYTHONPATH="$PWD/pyclnf:$PWD/pymtcnn:$PWD/pyfaceau:$PWD/pyfhog:$PYTHONPATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run processing script
python bigred200/process_single_video.py "$VIDEO_PATH" --output-dir "output/video_${SLURM_ARRAY_TASK_ID}"

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
