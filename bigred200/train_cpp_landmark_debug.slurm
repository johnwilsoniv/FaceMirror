#!/bin/bash
#SBATCH --job-name=debug_cpp_lm
#SBATCH --output=debug_cpp_lm_%j.out
#SBATCH --error=debug_cpp_lm_%j.err
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=r01984
#SBATCH --partition=gpu-debug
#SBATCH --gpus=1

# DEBUG: Test C++ landmark training - run 1 epoch to validate setup

set -e

module purge
module load python/3.12.11 cudatoolkit/12.2

cd "$HOME/pyfaceau"

export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "DEBUG: C++ Landmark Training Test"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Running 2 epochs to validate setup"
echo "============================================"

# Check GPU
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "Starting debug training run..."
echo ""

python3 -m pyfaceau.nn.train_landmark_pose \
    --data cpp_training_converted.h5 \
    --output models/debug_cpp_landmark \
    --model heatmap \
    --epochs 2 \
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
    --num-workers 4 \
    --save-every 1 \
    --val-split 0.1

echo ""
echo "============================================"
echo "DEBUG RUN COMPLETE!"
echo "End time: $(date)"
echo "============================================"
