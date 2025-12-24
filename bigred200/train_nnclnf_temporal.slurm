#!/bin/bash
#SBATCH --job-name=train_temporal
#SBATCH --output=train_temporal_%j.out
#SBATCH --error=train_temporal_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=r01984
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=4

# Train Temporal NNCLNF (5-frame window) on BigRed200
#
# Uses temporal context from neighboring frames to improve landmark accuracy.
# Estimated training time: ~3-4 hours with 4 A100 GPUs

set -e

module purge
module load python/3.12.11 cudatoolkit/12.2

cd "$HOME/pyfaceau"

export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Temporal NNCLNF Training (5-frame window)"
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
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Configuration
DATA_FILE="nnclnf_training_merged.h5"
OUTPUT_DIR="models/nnclnf_temporal"
PDM_FILE="weights/In-the-wild_aligned_PDM_68.txt"
WINDOW_SIZE=5

echo ""
echo "Training Configuration:"
echo "  Data: $DATA_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  PDM: $PDM_FILE"
echo "  Window size: $WINDOW_SIZE frames"
echo "  Batch size: 128 (32/GPU x 4)"
echo "  Epochs: 100"
echo "  LR: 5e-4"
echo ""

# Check data exists
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Training data not found: $DATA_FILE"
    exit 1
fi

# Resume from checkpoint if it exists
RESUME_FLAG=""
if [ -f "$OUTPUT_DIR/best_model.pt" ]; then
    RESUME_FLAG="--resume $OUTPUT_DIR/best_model.pt"
    echo "Resuming from checkpoint: $OUTPUT_DIR/best_model.pt"
fi

# Run training
python3 -m nnclnf.train_temporal \
    --data "$DATA_FILE" \
    --output "$OUTPUT_DIR" \
    --pdm "$PDM_FILE" \
    --window-size $WINDOW_SIZE \
    --epochs 100 \
    --batch-size 128 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --landmark-weight 10.0 \
    --val-split 0.1 \
    --patience 20 \
    --save-every 10 \
    --num-workers 8 \
    --amp \
    $RESUME_FLAG

echo ""
echo "============================================"
echo "Training Complete!"
echo "End time: $(date)"
echo "============================================"

# Show results
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
print(f'  Window size: {stats[\"window_size\"]} frames')
"
fi
