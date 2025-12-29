#!/bin/bash
#SBATCH --job-name=s1_debug
#SBATCH --account=r01984
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=s1_debug_%j.out
#SBATCH --error=s1_debug_%j.err

# ============================================
# S1 Face Mirror - DEBUG TEST (single video)
# ============================================

echo "=========================================="
echo "S1 FACE MIRROR - DEBUG TEST"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
module purge
module load python/3.12.11

# Set working directory
cd $HOME/pyfaceau

echo ""
echo "Working directory: $PWD"
echo "Contents:"
ls -la

# Set PYTHONPATH
export PYTHONPATH="$PWD/pyclnf:$PWD/pymtcnn:$PWD:$PWD/pyfhog:$PWD/S1 Face Mirror:$PYTHONPATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ""
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# Test imports
echo "Testing Python imports..."
python3 -c "
import sys
print('Python:', sys.version)
print()

print('Testing imports...')
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
    import pyfhog
    print('  pyfhog: OK')
except Exception as e:
    print(f'  pyfhog: FAILED - {e}')

print()
print('All imports successful!')
"

if [ $? -ne 0 ]; then
    echo "Import test failed!"
    exit 1
fi

# Check for test video
TEST_VIDEO="S Data/Normal Cohort/IMG_0422.MOV"
if [ ! -f "$TEST_VIDEO" ]; then
    echo "ERROR: Test video not found: $TEST_VIDEO"
    echo "Available in S Data:"
    ls -la "S Data/" 2>/dev/null || echo "S Data directory not found"
    exit 1
fi

echo ""
echo "Test video found: $TEST_VIDEO"
echo ""

# Run S1 on single video
echo "Running S1 batch processor..."
python3 bigred200/s1_batch_process.py "$TEST_VIDEO" \
    --output-dir "output/s1_debug_test" \
    --weights-dir "S1 Face Mirror/weights"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "DEBUG TEST PASSED - Ready to submit full job"
    echo ""
    echo "Output files:"
    find output/s1_debug_test -type f 2>/dev/null
else
    echo "DEBUG TEST FAILED"
fi

exit $EXIT_CODE
