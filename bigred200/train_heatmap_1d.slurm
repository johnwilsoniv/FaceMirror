#!/bin/bash
#SBATCH --job-name=train_heatmap_1d
#SBATCH --output=train_heatmap_1d_%j.out
#SBATCH --error=train_heatmap_1d_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --account=r01984
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=4

# Train Temporal1DHeatmapNet with 1D Heatmaps on BigRed200
#
# Multi-GPU DDP Training Configuration:
#   - 4 GPUs with DistributedDataParallel
#   - 16 batch per GPU x 4 GPUs x 2 accum = 128 effective batch
#   - Gradient accumulation for memory efficiency
#
# Architecture:
#   - HRNet-W18 backbone (maintains high resolution)
#   - 5-frame temporal window
#   - 1D heatmaps at 1024 resolution (X and Y separately)
#   - XY co-attention for coordinate correlation
#   - AWing loss for heatmaps + Wing loss for coordinates
#
# Target: <0.5 px accuracy in original image coordinates
# Improvement: 16× better resolution than 2D 64×64 heatmaps

set -e

module purge
module load python/3.12.11 cudatoolkit/12.2

cd "$HOME/pyfaceau"

# Add pymtcnn to path for consistency with data generation
export PYTHONPATH="$PWD:$HOME/pyfaceau/pymtcnn:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# DDP settings
export MASTER_ADDR=localhost
export MASTER_PORT=29500
NUM_GPUS=4

echo "============================================"
echo "Temporal1DHeatmapNet Training (1D Heatmaps)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $NUM_GPUS (DDP)"
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

# Check for timm (HRNet)
python3 -c "
try:
    import timm
    print(f'timm version: {timm.__version__}')
    print('HRNet-W18 available: True')
except ImportError:
    print('WARNING: timm not installed, will use simplified HRNet')
    print('Install with: pip install timm')
"

# Configuration
DATA_DIR="data/heatmap"
TRAIN_DATA="$DATA_DIR/train.h5"
VAL_DATA="$DATA_DIR/val.h5"
OUTPUT_DIR="models/heatmap_1d"
WINDOW_SIZE=5
HEATMAP_SIZE=1024
BATCH_SIZE=16      # Per GPU
ACCUM_STEPS=2      # Gradient accumulation
# Effective batch = 16 * 4 GPUs * 2 accum = 128

echo ""
echo "Training Configuration:"
echo "  Train data: $TRAIN_DATA"
echo "  Val data: $VAL_DATA"
echo "  Output: $OUTPUT_DIR"
echo "  Window size: $WINDOW_SIZE frames"
echo "  Heatmap resolution: $HEATMAP_SIZE (16× better than 2D 64×64)"
echo "  Batch size: $BATCH_SIZE per GPU x $NUM_GPUS GPUs x $ACCUM_STEPS accum = $((BATCH_SIZE * NUM_GPUS * ACCUM_STEPS)) effective"
echo "  Epochs: 200"
echo "  LR: 1e-4"
echo "  Image resolution: 256x256"
echo ""

# Check data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found: $TRAIN_DATA"
    echo "Run data generation first!"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    echo "ERROR: Validation data not found: $VAL_DATA"
    exit 1
fi

# Resume from checkpoint if it exists
RESUME_FLAG=""
if [ -f "$OUTPUT_DIR/best_model.pt" ]; then
    RESUME_FLAG="--resume $OUTPUT_DIR/best_model.pt"
    echo "Resuming from checkpoint: $OUTPUT_DIR/best_model.pt"
fi

# Run training with torchrun for DDP
torchrun --standalone --nproc_per_node=$NUM_GPUS \
    -m nnclnf.train_heatmap_1d \
    --train-data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --output "$OUTPUT_DIR" \
    --window-size $WINDOW_SIZE \
    --heatmap-size $HEATMAP_SIZE \
    --epochs 200 \
    --batch-size $BATCH_SIZE \
    --accum-steps $ACCUM_STEPS \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --coord-weight 10.0 \
    --heatmap-sigma 2.0 \
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
print(f'  Best val MAE: {stats[\"best_val_mae\"]:.3f} px')
print(f'  Heatmap resolution: {stats[\"heatmap_size\"]} (1D)')
if stats['best_val_mae'] < 0.5:
    print('  TARGET ACHIEVED: <0.5 px MAE!')
elif stats['best_val_mae'] < 1.0:
    print('  Close to target: <1.0 px MAE')
else:
    print(f'  Improvement needed: {stats[\"best_val_mae\"]:.3f} px > 0.5 px target')
"
fi
