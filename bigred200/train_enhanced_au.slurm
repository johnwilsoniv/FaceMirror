#!/bin/bash
#SBATCH --job-name=train_enhanced_au
#SBATCH --output=train_enhanced_au_%j.out
#SBATCH --error=train_enhanced_au_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=r01984
#SBATCH --partition=gpu
#SBATCH --gpus=1

# Train EnhancedAUPredictionNet for maximum accuracy AU prediction
# Target: r >= 0.99 correlation for all AUs
# Features: CBAM attention, region-specific heads, AU correlation transformer

set -e

module purge
module load python/3.12.11 cudatoolkit/12.2

cd "$HOME/pyfaceau"

export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "EnhancedAUPredictionNet Training"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Target: r >= 0.99 correlation"
echo "============================================"

# Check GPU
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "Model features:"
echo "  - CBAM attention (channel + spatial)"
echo "  - Region-specific AU heads (upper/mid/lower face, eyes)"
echo "  - AU correlation transformer (2 layers)"
echo "  - CCC loss for correlation optimization"
echo ""

python3 -m pyfaceau.nn.train_au_prediction \
    --data training_data_merged.h5 \
    --output models/enhanced_au \
    --model enhanced \
    --epochs 200 \
    --batch-size 128 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --l1-weight 1.0 \
    --ccc-weight 1.0 \
    --dropout 0.2 \
    --transformer-layers 2 \
    --num-workers 8 \
    --save-every 20 \
    --patience 50 \
    --val-split 0.1

echo ""
echo "============================================"
echo "Training complete!"
echo "End time: $(date)"
echo "============================================"
