#!/bin/bash
#SBATCH --job-name=pyclnf_diagnostic
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128              # Use all 128 cores (2x 64-core AMD EPYC)
#SBATCH --mem=128G                        # 128GB RAM
#SBATCH --time=04:00:00
#SBATCH --output=diagnostic_%j.out
#SBATCH --error=diagnostic_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jw411@iu.edu

# ============================================
# pyCLNF Diagnostic Analysis on Big Red 200
# Collects per-iteration convergence data
# ============================================

echo "=========================================="
echo "pyCLNF Diagnostic Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "=========================================="

# Load modules
module purge
module load python/3.12.11

# Activate virtual environment
source $HOME/pyfaceau_env/bin/activate

# Set working directory
cd $HOME/pyfaceau

# Set PYTHONPATH (pyfhog installed from PyPI)
export PYTHONPATH="$PWD/pyclnf:$PWD/pymtcnn:$PWD/pyfaceau:$PWD:$PYTHONPATH"

# Set thread counts
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ""
echo "PYTHONPATH: $PYTHONPATH"
echo "Workers: $SLURM_CPUS_PER_TASK"
echo ""

# Run diagnostic analysis
# Full response maps for all landmarks
# Use 16 workers to avoid memory bandwidth saturation
python -m bigred200.diagnostic.hpc_diagnostic_runner \
    --video "S Data/Normal Cohort/IMG_0942.MOV" \
    --cpp-reference "validation_output_0942/IMG_0942.csv" \
    --output-dir "diagnostic_output_full" \
    --workers 16

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
