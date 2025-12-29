#!/bin/bash
#SBATCH --job-name=s1_mirror
#SBATCH --account=r01984
#SBATCH --partition=gpu
#SBATCH --array=0-110                    # 111 videos (indices 0-110)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00                  # 1 hour per video (GPU: ~20 min actual)
#SBATCH --output=logs/s1_%A_%a.out
#SBATCH --error=logs/s1_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jw411@iu.edu

# ============================================
# S1 Face Mirror - All 111 Videos on BigRed200
# Each array task processes one video through:
#   1. Face detection + landmark extraction
#   2. Anatomical midline mirroring (L/R)
#   3. AU extraction on both mirrored versions
# ============================================

echo "=========================================="
echo "S1 FACE MIRROR - BIGRED200"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Load modules
module purge
module load python/3.12.11
module load cudatoolkit/12.6
module load cudnn/9.5.1.17_cuda12
module load ffmpeg/7.7.1

# Set working directory (BigRed200 uses ~/pyfaceau/)
cd $HOME/pyfaceau

# Use scratch for output (home quota is limited)
OUTPUT_BASE="/N/scratch/jw411/s1_output"
mkdir -p "$OUTPUT_BASE"

# Set temp directories to scratch to avoid quota issues
export TMPDIR="/N/scratch/jw411/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR"

# Set PYTHONPATH for local packages
export PYTHONPATH="$PWD/pyclnf:$PWD/pymtcnn:$PWD:$PWD/pyfhog:$PWD/S1 Face Mirror:$PYTHONPATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Video list - all 111 videos sorted alphabetically
# Generated from: find "S Data" -name "*.MOV" -o -name "*.mov" -o -name "*.mp4" | sort
VIDEOS=(
    # Normal Cohort (13 videos)
    "S Data/Normal Cohort/IMG_0422.MOV"
    "S Data/Normal Cohort/IMG_0428.MOV"
    "S Data/Normal Cohort/IMG_0433.MOV"
    "S Data/Normal Cohort/IMG_0434.MOV"
    "S Data/Normal Cohort/IMG_0435.MOV"
    "S Data/Normal Cohort/IMG_0437.MOV"
    "S Data/Normal Cohort/IMG_0438.MOV"
    "S Data/Normal Cohort/IMG_0441.MOV"
    "S Data/Normal Cohort/IMG_0443.MOV"
    "S Data/Normal Cohort/IMG_0452.MOV"
    "S Data/Normal Cohort/IMG_0453.MOV"
    "S Data/Normal Cohort/IMG_0579.MOV"
    "S Data/Normal Cohort/IMG_0942.MOV"
    # Paralysis Cohort (98 videos)
    "S Data/Paralysis Cohort/20240723_175947000_iOS.MOV"
    "S Data/Paralysis Cohort/20240723_185024000_iOS.MOV"
    "S Data/Paralysis Cohort/20240730_134811000_iOS.MOV"
    "S Data/Paralysis Cohort/20240730_153031000_iOS.MOV"
    "S Data/Paralysis Cohort/20240730_172902000_iOS.MOV"
    "S Data/Paralysis Cohort/20240731_181857000_iOS.MOV"
    "S Data/Paralysis Cohort/20240731_192357000_iOS.MOV"
    "S Data/Paralysis Cohort/20240806_140516000_iOS.MOV"
    "S Data/Paralysis Cohort/20240820_141306000_iOS.MOV"
    "S Data/Paralysis Cohort/20240820_152150000_iOS.MOV"
    "S Data/Paralysis Cohort/20240820_153346000_iOS.MOV"
    "S Data/Paralysis Cohort/20240820_173737000_iOS.MOV"
    "S Data/Paralysis Cohort/20240821_152738000_iOS.MOV"
    "S Data/Paralysis Cohort/20240821_173040000_iOS.MOV"
    "S Data/Paralysis Cohort/20240903_145723000_iOS.MOV"
    "S Data/Paralysis Cohort/20240903_160558000_iOS.MOV"
    "S Data/Paralysis Cohort/20240905_183857000_iOS.MOV"
    "S Data/Paralysis Cohort/20240918_170056000_iOS.MOV"
    "S Data/Paralysis Cohort/20241105_182638000_iOS.MOV"
    "S Data/Paralysis Cohort/20241106_144524000_iOS.MOV"
    "S Data/Paralysis Cohort/20241106_185658000_iOS.MOV"
    "S Data/Paralysis Cohort/20241112_185426000_iOS.MOV"
    "S Data/Paralysis Cohort/20241115_183047000_iOS.MOV"
    "S Data/Paralysis Cohort/20241119_160508000_iOS.MOV"
    "S Data/Paralysis Cohort/20241120_182250000_iOS.MOV"
    "S Data/Paralysis Cohort/20250107_204041000_iOS.MOV"
    "S Data/Paralysis Cohort/20250109_130218000_iOS.MOV"
    "S Data/Paralysis Cohort/20250114_185738000_iOS.MOV"
    "S Data/Paralysis Cohort/20250204_174155000_iOS.MOV"
    "S Data/Paralysis Cohort/20250204_180409000_iOS.MOV"
    "S Data/Paralysis Cohort/20250211_170817000_iOS.MOV"
    "S Data/Paralysis Cohort/20250211_174301000_iOS.MOV"
    "S Data/Paralysis Cohort/20250213_160905000_iOS.MOV"
    "S Data/Paralysis Cohort/20250213_193056000_iOS.MOV"
    "S Data/Paralysis Cohort/20250225_161739000_iOS.MOV"
    "S Data/Paralysis Cohort/20250225_180036000_iOS.MOV"
    "S Data/Paralysis Cohort/20250228_134810000_iOS.MOV"
    "S Data/Paralysis Cohort/20250304_165847000_iOS.MOV"
    "S Data/Paralysis Cohort/20250305_184617000_iOS.MOV"
    "S Data/Paralysis Cohort/20250305_184649000_iOS.MOV"
    "S Data/Paralysis Cohort/20250311_175016000_iOS.MOV"
    "S Data/Paralysis Cohort/20250312_124816000_iOS.MOV"
    "S Data/Paralysis Cohort/20250313_183242000_iOS.MOV"
    "S Data/Paralysis Cohort/20250313_195944000_iOS.MOV"
    "S Data/Paralysis Cohort/20250326_184059000_iOS.MOV"
    "S Data/Paralysis Cohort/20250326_201159000_iOS.MOV"
    "S Data/Paralysis Cohort/20250327_120839000_iOS.MOV"
    "S Data/Paralysis Cohort/20250402_133347000_iOS.MOV"
    "S Data/Paralysis Cohort/20250408_191322000_iOS.MOV"
    "S Data/Paralysis Cohort/20250409_124159000_iOS.MOV"
    "S Data/Paralysis Cohort/20250410_151806000_iOS.MOV"
    "S Data/Paralysis Cohort/IMG_0490.MOV"
    "S Data/Paralysis Cohort/IMG_0492.MOV"
    "S Data/Paralysis Cohort/IMG_0495.mov"
    "S Data/Paralysis Cohort/IMG_0504.MOV"
    "S Data/Paralysis Cohort/IMG_0592.MOV"
    "S Data/Paralysis Cohort/IMG_0642.MOV"
    "S Data/Paralysis Cohort/IMG_0804.MOV"
    "S Data/Paralysis Cohort/IMG_0861.MOV"
    "S Data/Paralysis Cohort/IMG_0935.MOV"
    "S Data/Paralysis Cohort/IMG_1339.MOV"
    "S Data/Paralysis Cohort/IMG_1366.MOV"
    "S Data/Paralysis Cohort/IMG_1368.MOV"
    "S Data/Paralysis Cohort/IMG_1602.MOV"
    "S Data/Paralysis Cohort/IMG_1837.MOV"
    "S Data/Paralysis Cohort/IMG_2068.MOV"
    "S Data/Paralysis Cohort/IMG_2122.MOV"
    "S Data/Paralysis Cohort/IMG_2259.MOV"
    "S Data/Paralysis Cohort/IMG_2380.mov"
    "S Data/Paralysis Cohort/IMG_2501.MOV"
    "S Data/Paralysis Cohort/IMG_2737.MOV"
    "S Data/Paralysis Cohort/IMG_2814.MOV"
    "S Data/Paralysis Cohort/IMG_2817.MOV"
    "S Data/Paralysis Cohort/IMG_3102.MOV"
    "S Data/Paralysis Cohort/IMG_3137.MOV"
    "S Data/Paralysis Cohort/IMG_3148.MOV"
    "S Data/Paralysis Cohort/IMG_3170.MOV"
    "S Data/Paralysis Cohort/IMG_3324.MOV"
    "S Data/Paralysis Cohort/IMG_3812.MOV"
    "S Data/Paralysis Cohort/IMG_3847.MOV"
    "S Data/Paralysis Cohort/IMG_4036.MOV"
    "S Data/Paralysis Cohort/IMG_4157.MOV"
    "S Data/Paralysis Cohort/IMG_4210.MOV"
    "S Data/Paralysis Cohort/IMG_4923.MOV"
    "S Data/Paralysis Cohort/IMG_5198.MOV"
    "S Data/Paralysis Cohort/IMG_5694.MOV"
    "S Data/Paralysis Cohort/IMG_6177.MOV"
    "S Data/Paralysis Cohort/IMG_7251.MOV"
    "S Data/Paralysis Cohort/IMG_7365.mov"
    "S Data/Paralysis Cohort/IMG_7540.MOV"
    "S Data/Paralysis Cohort/IMG_8270.MOV"
    "S Data/Paralysis Cohort/IMG_8401.MOV"
    "S Data/Paralysis Cohort/IMG_8514.MOV"
    "S Data/Paralysis Cohort/IMG_8537.mov"
    "S Data/Paralysis Cohort/IMG_9105.MOV"
    "S Data/Paralysis Cohort/IMG_9330.MOV"
    "S Data/Paralysis Cohort/IMG_9374.MOV"
    "S Data/Paralysis Cohort/IMG_9640.MOV"
)

# Get video for this task
VIDEO_PATH="${VIDEOS[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$VIDEO_PATH" ]; then
    echo "ERROR: No video found for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo ""
echo "Processing video $((SLURM_ARRAY_TASK_ID + 1))/111: $VIDEO_PATH"
echo ""

# Run S1 batch processor (output to scratch, use libx264 since NVENC has issues)
python bigred200/s1_batch_process.py "$VIDEO_PATH" \
    --output-dir "$OUTPUT_BASE/video_$(printf '%03d' $SLURM_ARRAY_TASK_ID)" \
    --weights-dir "S1 Face Mirror/weights" \
    --no-nvenc

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
