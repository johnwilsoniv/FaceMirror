#!/bin/bash
#SBATCH --job-name=gen_heatmap_data
#SBATCH --output=gen_heatmap_data_%j.out
#SBATCH --error=gen_heatmap_data_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=r01984
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

# Generate 256x256 training data with video-stratified splits
#
# Creates:
#   - data/heatmap/train.h5  (80% of videos)
#   - data/heatmap/val.h5    (10% of videos)
#   - data/heatmap/test.h5   (10% of videos)
#   - data/heatmap/split_info.json

set -e

module purge
module load python/3.12.11 cudatoolkit/12.2

cd "$HOME/pyfaceau"

# Add pymtcnn to path for MTCNN face detection
export PYTHONPATH="$PWD:$HOME/pyfaceau/pymtcnn:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Heatmap Training Data Generation"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================"

# Paths - include both cohorts
VIDEO_DIR_PARALYSIS="$HOME/pyfaceau/S/Paralysis Cohort"
VIDEO_DIR_NORMAL="$HOME/pyfaceau/S/Normal Cohort"
CSV_DIR="$HOME/pyfaceau/cpp_groundtruth"
OUTPUT_DIR="$HOME/pyfaceau/data/heatmap"

echo ""
echo "Configuration:"
echo "  Video dir 1: $VIDEO_DIR_PARALYSIS"
echo "  Video dir 2: $VIDEO_DIR_NORMAL"
echo "  CSV dir: $CSV_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Output size: 256x256"
echo "  Train/Val/Test: 80/10/10"
echo ""

# Check inputs exist
if [ ! -d "$VIDEO_DIR_PARALYSIS" ]; then
    echo "ERROR: Video directory not found: $VIDEO_DIR_PARALYSIS"
    exit 1
fi

if [ ! -d "$VIDEO_DIR_NORMAL" ]; then
    echo "ERROR: Video directory not found: $VIDEO_DIR_NORMAL"
    exit 1
fi

if [ ! -d "$CSV_DIR" ]; then
    echo "ERROR: CSV directory not found: $CSV_DIR"
    exit 1
fi

# Count videos
N_PARALYSIS=$(find "$VIDEO_DIR_PARALYSIS" \( -name "*.MOV" -o -name "*.mov" -o -name "*.mp4" \) | wc -l)
N_NORMAL=$(find "$VIDEO_DIR_NORMAL" \( -name "*.MOV" -o -name "*.mov" -o -name "*.mp4" \) | wc -l)
N_CSVS=$(find "$CSV_DIR" -name "*.csv" | wc -l)
echo "Found $N_PARALYSIS paralysis + $N_NORMAL normal = $((N_PARALYSIS + N_NORMAL)) total videos"
echo "Found $N_CSVS CSVs"

# Run data generator with both directories
python3 -m nnclnf.data_generator_heatmap \
    --video-dir "$VIDEO_DIR_PARALYSIS" "$VIDEO_DIR_NORMAL" \
    --csv-dir "$CSV_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --output-size 256 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --seed 42 \
    --min-confidence 0.5

echo ""
echo "============================================"
echo "Data Generation Complete!"
echo "End time: $(date)"
echo "============================================"

# Show results
if [ -f "$OUTPUT_DIR/split_info.json" ]; then
    echo ""
    echo "Split Info:"
    cat "$OUTPUT_DIR/split_info.json"
fi

# Show file sizes
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.h5 2>/dev/null || echo "No HDF5 files found"
