#!/bin/bash
#SBATCH --job-name=train_heatmap_lm
#SBATCH --output=train_heatmap_%j.out
#SBATCH --error=train_heatmap_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=r01984
#SBATCH --partition=gpu
#SBATCH --gpus=1

# Train HeatmapLandmarkNet for high-accuracy landmark prediction
# Target: MAE < 0.5px (sub-pixel accuracy)
# Model: ResNet backbone + U-Net decoder + soft-argmax + refinement head

set -e

module purge
module load python/3.12.11 cudatoolkit/12.2

cd "$HOME/pyfaceau"

# Set PYTHONPATH and unbuffered output
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "HeatmapLandmarkNet Training"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Target MAE: 0.5px"
echo "============================================"

# Check GPU
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "Starting heatmap model training..."
echo "Architecture: ResNet backbone + U-Net decoder + soft-argmax + refinement"
echo "Parameters: ~14.6M"
echo ""

python3 -m pyfaceau.nn.train_landmark_pose \
    --data training_data_merged.h5 \
    --output models/heatmap_landmark \
    --model heatmap \
    --epochs 500 \
    --batch-size 128 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --heatmap-weight 1.0 \
    --coord-weight 10.0 \
    --consistency-weight 0.5 \
    --gp-weight 0.01 \
    --lp-weight 0.001 \
    --target-mae 0.5 \
    --min-delta 0.0001 \
    --patience 100 \
    --num-workers 8 \
    --save-every 25 \
    --val-split 0.1

echo ""
echo "============================================"
echo "Training complete!"
echo "End time: $(date)"
echo "============================================"
