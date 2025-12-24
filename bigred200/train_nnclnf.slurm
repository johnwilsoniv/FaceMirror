#!/bin/bash
#SBATCH --job-name=train_nnclnf
#SBATCH --output=train_nnclnf_%j.out
#SBATCH --error=train_nnclnf_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=r01984
#SBATCH --partition=gpu-debug
#SBATCH --gpus-per-node=4

# Train NNCLNF (Neural Network CLNF) on BigRed200
#
# Prerequisites:
# 1. C++ ground truth generated for all videos
# 2. Training data generated using nnclnf/data_generator.py
# 3. Training data merged into single HDF5

set -e

module purge
module load python/3.12.11 cudatoolkit/12.2

cd "$HOME/pyfaceau"

export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "NNCLNF Training"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "============================================"

# Check GPU
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"

# Training configuration
DATA_FILE="nnclnf_training_merged.h5"
OUTPUT_DIR="models/nnclnf"
PDM_FILE="weights/In-the-wild_aligned_PDM_68.txt"

echo ""
echo "Training Configuration:"
echo "  Data: $DATA_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  PDM: $PDM_FILE"
echo "  Model: standard (ParamConditionedNet)"
echo "  Batch size: 256 (64/GPU x 4)"
echo "  Epochs: 100"
echo "  LR: 1e-3"
echo ""

# Check data exists
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Training data not found: $DATA_FILE"
    echo "Run data generation first!"
    exit 1
fi

# Check PDM exists
if [ ! -f "$PDM_FILE" ]; then
    echo "ERROR: PDM file not found: $PDM_FILE"
    exit 1
fi

# Run training
python3 -m nnclnf.train \
    --data "$DATA_FILE" \
    --output "$OUTPUT_DIR" \
    --pdm "$PDM_FILE" \
    --model standard \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --landmark-weight 10.0 \
    --val-split 0.1 \
    --patience 20 \
    --save-every 10 \
    --num-workers 8 \
    --amp

echo ""
echo "============================================"
echo "Training Complete!"
echo "End time: $(date)"
echo "============================================"

# Show final results
if [ -f "$OUTPUT_DIR/training_stats.json" ]; then
    echo ""
    echo "Final Results:"
    python3 -c "
import json
with open('$OUTPUT_DIR/training_stats.json') as f:
    stats = json.load(f)
print(f'  Total time: {stats[\"total_time\"]/60:.1f} minutes')
print(f'  Epochs: {stats[\"epochs\"]}')
print(f'  Best val loss: {stats[\"best_val_loss\"]:.4f}')
"
fi
