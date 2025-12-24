#!/bin/bash
#SBATCH --job-name=nnclnf_pipe
#SBATCH --output=nnclnf_pipe_%A_%a.out
#SBATCH --error=nnclnf_pipe_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --array=0-110

# Combined pipeline: C++ ground truth → NNCLNF training data
# 111 tasks, one per video (both cohorts: 13 normal + 98 paralysis)

set -e

echo "============================================"
echo "NNCLNF Pipeline (C++ GT + Training Data)"
echo "============================================"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "============================================"

# ============================================
# STEP 1: C++ OpenFace Ground Truth
# ============================================

module purge
module load boost/1.86.0
module load ffmpeg/7.7.1

export LD_LIBRARY_PATH="/N/soft/sles15sp6/boost/gnu/1.86.0/lib:$HOME/software/opencv_install/lib64:/N/soft/sles15sp6/ffmpeg/7.7.1/lib:/N/soft/sles15sp6/libjpeg/3.1.2/lib64:$LD_LIBRARY_PATH"

OPENFACE_BIN="$HOME/software/OpenFace/build/bin/FeatureExtraction"
CECLM_MODEL="$HOME/software/OpenFace/lib/local/LandmarkDetector/model/main_ceclm_general.txt"

VIDEO_DIRS=("$HOME/pyfaceau/S Data/Normal Cohort" "$HOME/pyfaceau/S Data/Paralysis Cohort")
CPP_GT_DIR="$HOME/pyfaceau/cpp_groundtruth"
NNCLNF_DIR="$HOME/pyfaceau/nnclnf_training_data"
PDM_FILE="$HOME/pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"

mkdir -p "$CPP_GT_DIR" "$NNCLNF_DIR"

# Get video list from both cohorts
VIDEOS=()
for DIR in "${VIDEO_DIRS[@]}"; do
    while IFS= read -r -d '' file; do
        VIDEOS+=("$file")
    done < <(find "$DIR" -maxdepth 1 \( -name "*.MOV" -o -name "*.mov" \) -print0 2>/dev/null | sort -z)
done

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#VIDEOS[@]}" ]; then
    echo "Task ID exceeds video count, nothing to do"
    exit 0
fi

VIDEO="${VIDEOS[$SLURM_ARRAY_TASK_ID]}"
# Strip extension (handle both .MOV and .mov)
VIDEO_NAME=$(basename "$VIDEO")
VIDEO_NAME="${VIDEO_NAME%.[Mm][Oo][Vv]}"
CSV_FILE="$CPP_GT_DIR/${VIDEO_NAME}.csv"
H5_FILE="$NNCLNF_DIR/training_${VIDEO_NAME}.h5"

echo "Video: $VIDEO"
echo "CSV: $CSV_FILE"
echo "H5: $H5_FILE"

# Skip if H5 already exists (fully complete)
if [ -f "$H5_FILE" ]; then
    echo "Already complete, skipping"
    exit 0
fi

# Generate C++ ground truth if needed
if [ -f "$CSV_FILE" ] && head -1 "$CSV_FILE" | grep -q "p_scale"; then
    echo "C++ ground truth exists with pdmparams"
else
    echo ""
    echo "=== Running C++ OpenFace ==="
    "$OPENFACE_BIN" \
        -f "$VIDEO" \
        -out_dir "$CPP_GT_DIR" \
        -mloc "$CECLM_MODEL" \
        -2Dfp -3Dfp -pdmparams -pose -aus \
        -of "$VIDEO_NAME"

    if [ ! -f "$CSV_FILE" ]; then
        echo "ERROR: C++ OpenFace failed"
        exit 1
    fi
    echo "C++ ground truth generated: $(wc -l < "$CSV_FILE") lines"
fi

# ============================================
# STEP 2: NNCLNF Training Data
# ============================================

echo ""
echo "=== Generating NNCLNF Training Data ==="

module purge
module load python/3.12.11

cd "$HOME/pyfaceau"
export PYTHONPATH="$PWD:$PYTHONPATH"

python3 -m nnclnf.data_generator \
    --video "$VIDEO" \
    --csv "$CSV_FILE" \
    --pdm "$PDM_FILE" \
    --output "$H5_FILE" \
    --no-mtcnn

if [ -f "$H5_FILE" ]; then
    python3 -c "
import h5py
with h5py.File('$H5_FILE', 'r') as f:
    print(f'Generated {f[\"images\"].shape[0]} samples')
"
else
    echo "ERROR: Training data generation failed"
    exit 1
fi

echo ""
echo "============================================"
echo "Complete!"
echo "End time: $(date)"
echo "============================================"
