#!/bin/bash
#SBATCH --job-name=pyclnf_ablation
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --array=0-499%64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=ablation_%A_%a.out
#SBATCH --error=ablation_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jw411@iu.edu

# ============================================
# pyCLNF Ablation Study - SLURM Array Job
# Each task tests one parameter configuration
# ============================================

echo "=========================================="
echo "Ablation Experiment ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Load modules
module purge
module load python/3.12.11

# Activate virtual environment
source $HOME/pyfaceau_env/bin/activate

# Set working directory
cd $HOME/pyfaceau

# Set PYTHONPATH
export PYTHONPATH="$PWD/pyclnf:$PWD/pymtcnn:$PWD/pyfaceau:$PWD/pyfhog:$PWD:$PYTHONPATH"

# Run single experiment
python -m bigred200.ablation.run_single_experiment \
    --manifest bigred200/config/experiment_manifest.csv \
    --task-id $SLURM_ARRAY_TASK_ID \
    --video "S Data/Normal Cohort/IMG_0942.MOV" \
    --cpp-reference "validation_output_0942/IMG_0942.csv" \
    --max-frames 100 \
    --output-dir bigred200/ablation/results/raw

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
