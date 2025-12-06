#!/bin/bash
#SBATCH --job-name=merge_training
#SBATCH --account=r01984
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=bigred200/output/logs/merge_%j.out
#SBATCH --error=bigred200/output/logs/merge_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jw411@iu.edu

# ============================================
# Merge Per-Video HDF5 Files
# Run after array job completes
# ============================================

echo "=========================================="
echo "Merging Training Data"
echo "Job ID: $SLURM_JOB_ID"
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

export PYTHONPATH="$PWD/pyclnf:$PWD/pymtcnn:$PWD/pyfaceau:$PWD/pyfhog:$PWD:$PYTHONPATH"

# Count completed videos
COMPLETE_COUNT=$(ls -1 bigred200/output/checkpoints/*.done 2>/dev/null | wc -l)
EXPECTED_COUNT=$(wc -l < bigred200/config/video_manifest.txt)
HDF5_COUNT=$(ls -1 bigred200/output/per_video/*.h5 2>/dev/null | wc -l)

echo "Checkpoints completed: $COMPLETE_COUNT / $EXPECTED_COUNT"
echo "HDF5 files found: $HDF5_COUNT"

if [ "$COMPLETE_COUNT" -lt "$EXPECTED_COUNT" ]; then
    echo ""
    echo "WARNING: Not all videos completed!"
    echo "Missing: $(($EXPECTED_COUNT - $COMPLETE_COUNT)) videos"
    echo "Proceeding with available data..."
fi

if [ "$HDF5_COUNT" -eq 0 ]; then
    echo ""
    echo "ERROR: No HDF5 files found to merge!"
    exit 1
fi

# Run merge
echo ""
echo "Starting merge..."
python bigred200/training/merge_hdf5_files.py \
    --input-dir bigred200/output/per_video \
    --output bigred200/output/training_data_final.h5 \
    --chunk-size 1000

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    # Verify the output
    echo ""
    echo "Verifying merged dataset..."
    python -c "
import h5py
import numpy as np
h5_path = 'bigred200/output/training_data_final.h5'
with h5py.File(h5_path, 'r') as f:
    print(f'Total samples: {f[\"images\"].shape[0]}')
    print(f'Image shape: {f[\"images\"].shape[1:]}')
    print(f'HOG features: {f[\"hog_features\"].shape[1]}')
    print(f'Landmarks: {f[\"landmarks\"].shape[1:]}')
    print(f'Global params: {f[\"global_params\"].shape[1]}')
    print(f'Local params: {f[\"local_params\"].shape[1]}')
    print(f'AU intensities: {f[\"au_intensities\"].shape[1]}')
    if 'metadata' in f:
        unique_videos = len(set(f['metadata/video_names'][:]))
        print(f'Unique videos: {unique_videos}')
"

    echo ""
    echo "SUCCESS: Merge complete"
    ls -lh bigred200/output/training_data_final.h5
else
    echo ""
    echo "FAILED: Merge failed (exit code $EXIT_CODE)"
fi

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE
