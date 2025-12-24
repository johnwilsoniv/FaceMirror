#!/bin/bash
#SBATCH --job-name=train_heatmap
#SBATCH --output=train_heatmap_%j.out
#SBATCH --error=train_heatmap_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=r01984
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=4

# Train TemporalHeatmapNet on BigRed200
#
# Architecture:
#   - HRNet-W18 backbone (maintains high resolution)
#   - 5-frame temporal window
#   - Heatmap + offset regression
#   - AWing loss for heatmaps
#   - Wing loss for coordinates
#
# Target: <0.5 px accuracy in original image coordinates

set -e

module purge
module load python/3.12.11 cudatoolkit/12.2

cd "$HOME/pyfaceau"

# Add pymtcnn to path for consistency with data generation
export PYTHONPATH="$PWD:$HOME/pyfaceau/pymtcnn:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "TemporalHeatmapNet Training"
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
OUTPUT_DIR="models/heatmap_temporal"
WINDOW_SIZE=5

echo ""
echo "Training Configuration:"
echo "  Train data: $TRAIN_DATA"
echo "  Val data: $VAL_DATA"
echo "  Output: $OUTPUT_DIR"
echo "  Window size: $WINDOW_SIZE frames"
echo "  Batch size: 128 (32/GPU x 4)"
echo "  Epochs: 200"
echo "  LR: 1e-4"
echo "  Resolution: 256x256"
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

# Run training
python3 -m nnclnf.train_heatmap \
    --train-data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --output "$OUTPUT_DIR" \
    --window-size $WINDOW_SIZE \
    --epochs 200 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --coord-weight 10.0 \
    --heatmap-sigma 1.5 \
    --patience 20 \
    --save-every 10 \
    --num-workers 16 \
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
if 'history' in stats and len(stats['history']) > 0:
    last = stats['history'][-1]
    if 'val_mae' in last:
        print(f'  Final MAE: {last[\"val_mae\"]:.2f} px (original coords)')
"
fi
