#!/bin/bash
#SBATCH --job-name=s1_gpu_debug
#SBATCH --account=r01984
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/s1_gpu_debug_%j.out
#SBATCH --error=logs/s1_gpu_debug_%j.err

# ============================================
# S1 Face Mirror - GPU DEBUG TEST (Optimized)
# Tests NVENC + parallel AU extraction
# ============================================

echo "=========================================="
echo "S1 FACE MIRROR - OPTIMIZED GPU DEBUG TEST"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Load modules
module purge
module load python/3.12.11
module load cudatoolkit/12.6
module load cudnn/9.5.1.17_cuda12

# Set working directory
cd $HOME/pyfaceau

echo ""
echo "Working directory: $PWD"
echo ""

# Set PYTHONPATH
export PYTHONPATH="$PWD/pyclnf:$PWD/pymtcnn:$PWD:$PWD/pyfhog:$PWD/S1 Face Mirror:$PYTHONPATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Test CUDA availability
echo "Testing CUDA..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('WARNING: CUDA not available!')
"

# Check NVENC availability
echo ""
echo "Checking NVENC..."
ffmpeg -hide_banner -encoders 2>/dev/null | grep nvenc || echo "NVENC not found in ffmpeg"

echo ""
echo "Testing imports..."
python3 -c "
import sys
print('Python:', sys.version)

try:
    from pymtcnn import MTCNN
    print('  pymtcnn: OK')
except Exception as e:
    print(f'  pymtcnn: FAILED - {e}')

try:
    from pyclnf import CLNF
    print('  pyclnf: OK')
except Exception as e:
    print(f'  pyclnf: FAILED - {e}')

try:
    from pyfaceau.processor import OpenFaceProcessor
    print('  pyfaceau: OK')
except Exception as e:
    print(f'  pyfaceau: FAILED - {e}')

try:
    from multiprocessing import Process, Queue
    print('  multiprocessing: OK')
except Exception as e:
    print(f'  multiprocessing: FAILED - {e}')
"

if [ $? -ne 0 ]; then
    echo "Import test failed!"
    exit 1
fi

# Check for test video
TEST_VIDEO="S Data/Normal Cohort/IMG_0422.MOV"
if [ ! -f "$TEST_VIDEO" ]; then
    echo "ERROR: Test video not found: $TEST_VIDEO"
    exit 1
fi

echo ""
echo "Test video found: $TEST_VIDEO"
echo ""

# Run optimized S1 on single video with timing
# Use --use-cv2 first to test without ffmpeg dependency
echo "Running OPTIMIZED S1 batch processor (OpenCV writer)..."
echo "=========================================="
time python3 bigred200/s1_batch_process.py "$TEST_VIDEO" \
    --output-dir "output/s1_gpu_debug_optimized" \
    --weights-dir "S1 Face Mirror/weights" \
    --use-cv2

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "GPU DEBUG TEST PASSED"
    echo ""
    echo "Output files:"
    find output/s1_gpu_debug_optimized -type f 2>/dev/null
    echo ""
    echo "File sizes:"
    ls -lh output/s1_gpu_debug_optimized/mirrored/ 2>/dev/null
    ls -lh output/s1_gpu_debug_optimized/au_data/ 2>/dev/null
else
    echo "GPU DEBUG TEST FAILED"
fi

exit $EXIT_CODE
