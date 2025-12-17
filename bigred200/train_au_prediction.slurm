#!/bin/bash
#SBATCH --job-name=train_au
#SBATCH --output=train_au_%j.out
#SBATCH --error=train_au_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=r01984
#SBATCH --partition=gpu
#SBATCH --gpus=1

# Train AUPredictionNet on merged training data

set -e

module purge
module load python/3.12.11 cudatoolkit/12.2

cd "$HOME/pyfaceau"

# Set PYTHONPATH and unbuffered output
export PYTHONPATH="$PWD/pyfaceau:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "AUPredictionNet Training"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "============================================"

# Check GPU
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "Starting training..."
echo ""

python3 -m pyfaceau.nn.train_au_prediction \
    --data training_data_merged.h5 \
    --output models/au_prediction \
    --epochs 100 \
    --batch-size 512 \
    --lr 2e-3 \
    --l1-weight 1.0 \
    --ccc-weight 0.5 \
    --dropout 0.3 \
    --patience 30 \
    --num-workers 8 \
    --save-every 10 \
    --val-split 0.1

echo ""
echo "============================================"
echo "Training complete!"
echo "End time: $(date)"
echo "============================================"
