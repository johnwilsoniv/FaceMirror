#!/bin/bash
#SBATCH --job-name=train_lm_tight
#SBATCH --output=train_landmark_%j.out
#SBATCH --error=train_landmark_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=r01984
#SBATCH --partition=gpu
#SBATCH --gpus=1

# Train LandmarkPoseNet with TIGHT parameters for sub-pixel accuracy
# Target: MAE < 0.5px

set -e

module purge
module load python/3.12.11 cudatoolkit/12.2

cd "$HOME/pyfaceau"

# Set PYTHONPATH and unbuffered output
export PYTHONPATH="$PWD/pyfaceau:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "LandmarkPoseNet Training (TIGHT MODE)"
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
echo "Starting training with tight convergence criteria..."
echo ""

python3 -m pyfaceau.nn.train_landmark_pose \
    --data training_data_merged.h5 \
    --output models/landmark_pose_tight \
    --epochs 500 \
    --batch-size 256 \
    --lr 1e-3 \
    --lm-weight 10.0 \
    --gp-weight 0.01 \
    --lp-weight 0.001 \
    --wing-w 2.0 \
    --wing-epsilon 0.5 \
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
