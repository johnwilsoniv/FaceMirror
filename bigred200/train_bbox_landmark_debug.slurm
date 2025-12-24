#!/bin/bash
#SBATCH --job-name=train_bbox_lm
#SBATCH --output=train_bbox_lm_%j.out
#SBATCH --error=train_bbox_lm_%j.err
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=r01984
#SBATCH --partition=gpu-debug
#SBATCH --gpus=1

# Train NNCLNF with BBOX-aligned training data
# Optimized for gpu-debug partition (1 hour limit)

set -e

module purge
module load python/3.12.11 cudatoolkit/12.2

cd "$HOME/pyfaceau"

export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Train NNCLNF with BBOX Alignment"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "============================================"

# Check GPU
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "Training Configuration:"
echo "  - Data: bbox_training_merged.h5"
echo "  - Model: HeatmapLandmarkNet"
echo "  - Batch size: 256 (optimized for A100)"
echo "  - LR: 2e-3 with cosine annealing"
echo "  - Target MAE: 0.5 px (in 112x112 space)"
echo "  - Early stopping patience: 30"
echo ""

python3 -m pyfaceau.nn.train_landmark_pose \
    --data bbox_training_merged.h5 \
    --output models/bbox_landmark \
    --model heatmap \
    --epochs 100 \
    --batch-size 256 \
    --lr 2e-3 \
    --weight-decay 1e-4 \
    --heatmap-weight 1.0 \
    --coord-weight 10.0 \
    --consistency-weight 0.5 \
    --gp-weight 0.01 \
    --lp-weight 0.001 \
    --target-mae 0.5 \
    --min-delta 0.001 \
    --patience 30 \
    --num-workers 8 \
    --save-every 5 \
    --val-split 0.1

echo ""
echo "============================================"
echo "Training complete!"
echo "End time: $(date)"
echo "============================================"

# Check best MAE achieved
echo ""
echo "Final Results:"
if [ -f models/bbox_landmark/training_stats.json ]; then
    python3 -c "
import json
with open('models/bbox_landmark/training_stats.json') as f:
    stats = json.load(f)
print(f'Best MAE: {stats.get(\"best_mae\", \"N/A\")} px')
print(f'Final epoch: {stats.get(\"epoch\", \"N/A\")}')
"
fi
