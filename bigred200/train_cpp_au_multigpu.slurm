#!/bin/bash
#SBATCH --job-name=train_cpp_au_4gpu
#SBATCH --output=train_cpp_au_4gpu_%j.out
#SBATCH --error=train_cpp_au_4gpu_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --account=r01984
#SBATCH --partition=gpu
#SBATCH --gpus=4

# Multi-GPU Training: AU Prediction Network on C++ OpenFace ground truth
# Uses 4 GPUs with DataParallel for ~3-4x speedup
# Target: Per-AU correlation >= 0.95, MAE < 0.3

set -e

module purge
module load python/3.12.11 cudatoolkit/12.2

cd "$HOME/pyfaceau"

export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Multi-GPU C++ AU Training"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Data: cpp_training_converted.h5"
echo "Target: r>=0.95 per-AU, MAE<0.3"
echo "============================================"

# Check GPUs
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "Training with DataParallel across $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) GPUs..."
echo "Effective batch size: 256 * num_gpus"
echo ""

# Use --multi-gpu flag to enable DataParallel
python3 -m pyfaceau.nn.train_au_prediction \
    --data cpp_training_converted.h5 \
    --output models/cpp_au_prediction_4gpu \
    --model enhanced \
    --epochs 200 \
    --batch-size 256 \
    --lr 2e-3 \
    --weight-decay 1e-4 \
    --l1-weight 1.0 \
    --ccc-weight 0.5 \
    --warmup-epochs 5 \
    --patience 50 \
    --num-workers 16 \
    --save-every 20 \
    --val-split 0.1 \
    --multi-gpu

echo ""
echo "============================================"
echo "Training complete!"
echo "End time: $(date)"
echo "============================================"
