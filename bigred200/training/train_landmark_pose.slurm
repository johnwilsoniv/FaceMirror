#!/bin/bash
#SBATCH --job-name=train_landmark_pose
#SBATCH --account=r01984
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=bigred200/output/logs/train_landmark_pose_%j.out
#SBATCH --error=bigred200/output/logs/train_landmark_pose_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jw411@iu.edu

# ============================================
# Landmark/Pose Network Training - GPU Job
# Trains MobileNetV2 for landmarks + pose
# ============================================

echo "=========================================="
echo "Landmark/Pose Network Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Load GPU modules
module purge
module load python/gpu/3.10.10

# Activate GPU environment
source $HOME/pyfaceau_gpu_env/bin/activate

# Set working directory
cd $HOME/pyfaceau

# Set environment
export PYTHONPATH="$PWD/pyclnf:$PWD/pymtcnn:$PWD/pyfaceau:$PWD/pyfhog:$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Enable cuDNN benchmarking
export TORCH_CUDNN_V8_API_ENABLED=1

# Verify PyTorch GPU access
echo "Verifying PyTorch GPU access..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
echo ""

# Training parameters
DATA_H5="bigred200/output/training_data_final.h5"
OUTPUT_DIR="models/landmark_pose"
EPOCHS=100
BATCH_SIZE=128
LR=4e-3

# Check data exists
if [ ! -f "$DATA_H5" ]; then
    echo "ERROR: Training data not found: $DATA_H5"
    echo "Please run the data generation pipeline first."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
echo "Starting training..."
echo "  Data: $DATA_H5"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo ""

python -m pyfaceau.nn.train_landmark_pose \
    --data "$DATA_H5" \
    --output "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --val-split 0.1 \
    --num-workers 8 \
    --save-every 10 \
    --patience 30

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Training complete"
    ls -la "$OUTPUT_DIR/"

    # Export to ONNX if training succeeded
    echo ""
    echo "Exporting to ONNX..."
    python -c "
import torch
from pyfaceau.nn.landmark_pose_net import UnifiedLandmarkPoseNet, export_to_onnx
import os

output_dir = '$OUTPUT_DIR'
best_ckpt = os.path.join(output_dir, 'checkpoint_best.pt')
if os.path.exists(best_ckpt):
    model = UnifiedLandmarkPoseNet()
    ckpt = torch.load(best_ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    onnx_path = os.path.join(output_dir, 'landmark_pose.onnx')
    export_to_onnx(model, onnx_path)
    print(f'ONNX model exported to: {onnx_path}')
else:
    print(f'Best checkpoint not found: {best_ckpt}')
"
else
    echo ""
    echo "FAILED: Training failed (exit code $EXIT_CODE)"
fi

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE
