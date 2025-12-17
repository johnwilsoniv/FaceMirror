#!/bin/bash
#SBATCH --job-name=train_cpp_au
#SBATCH --output=train_cpp_au_%j.out
#SBATCH --error=train_cpp_au_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=r01984
#SBATCH --partition=gpu
#SBATCH --gpus=1

# Train AU Prediction Network on C++ OpenFace ground truth data
# Target: Per-AU correlation >= 0.95, MAE < 0.3
# Using C++ AU values as ground truth

set -e

module purge
module load python/3.12.11 cudatoolkit/12.2

cd "$HOME/pyfaceau"

export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "C++ AU Training (Enhanced Model)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Data: cpp_training_converted.h5 (C++ OpenFace GT)"
echo "Target: r>=0.95 per-AU, MAE<0.3"
echo "============================================"

# Check GPU
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "Training AU prediction model on C++ ground truth..."
echo "Architecture: EfficientNet-lite backbone + attention + multi-scale"
echo ""

python3 -m pyfaceau.nn.train_au_prediction \
    --data cpp_training_converted.h5 \
    --output models/cpp_au_prediction \
    --model enhanced \
    --epochs 200 \
    --batch-size 256 \
    --lr 2e-3 \
    --weight-decay 1e-4 \
    --l1-weight 1.0 \
    --ccc-weight 0.5 \
    --warmup-epochs 5 \
    --patience 50 \
    --num-workers 8 \
    --save-every 20 \
    --val-split 0.1

echo ""
echo "============================================"
echo "Training complete!"
echo "End time: $(date)"
echo "============================================"
