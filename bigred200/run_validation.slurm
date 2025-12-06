#!/bin/bash
#SBATCH --job-name=pyfaceau_validation
#SBATCH --account=r01984
#SBATCH --partition=general              # CPU partition (CLNF is CPU-bound)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128              # Use all 128 cores (2x 64-core AMD EPYC)
#SBATCH --mem=128G                       # More RAM for loading frames into memory
#SBATCH --time=02:00:00                  # 2 hour time limit
#SBATCH --output=validation_%j.out
#SBATCH --error=validation_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jw411@iu.edu

# ============================================
# PyFaceAU Pipeline Validation on Big Red 200
# ============================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load required modules
module purge
module load python/3.12.11

# Activate virtual environment if using one
# source $HOME/pyfaceau_env/bin/activate

# Set working directory
cd $HOME/pyfaceau

# Set PYTHONPATH for our custom packages (pyfhog installed from PyPI)
export PYTHONPATH="$PWD/pyclnf:$PWD/pymtcnn:$PWD/pyfaceau:$PYTHONPATH"

# Set number of threads for OpenMP/NumPy
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "PYTHONPATH: $PYTHONPATH"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

# Run validation script
echo "Starting validation..."
python bigred200/run_validation_hpc.py

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
