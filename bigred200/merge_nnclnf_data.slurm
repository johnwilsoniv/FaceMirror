#!/bin/bash
#SBATCH --job-name=merge_nnclnf
#SBATCH --output=merge_nnclnf_%j.out
#SBATCH --error=merge_nnclnf_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=r01984
#SBATCH --partition=debug

# Merge NNCLNF training data files into single HDF5

set -e

module purge
module load python/3.12.11

cd "$HOME/pyfaceau"

echo "============================================"
echo "Merge NNCLNF Training Data"
echo "============================================"
echo "Start time: $(date)"

python3 -c "
import h5py
import numpy as np
from pathlib import Path
import os

input_dir = Path('nnclnf_training_data')
output_file = 'nnclnf_training_merged.h5'

# Find all HDF5 files
h5_files = sorted(input_dir.glob('training_*.h5'))
print(f'Found {len(h5_files)} training files')

if len(h5_files) == 0:
    print('ERROR: No training files found!')
    exit(1)

# Count total samples
total_samples = 0
for f in h5_files:
    with h5py.File(f, 'r') as h5:
        total_samples += h5['images'].shape[0]
print(f'Total samples: {total_samples}')

# Get shapes from first file
with h5py.File(h5_files[0], 'r') as h5:
    img_shape = h5['images'].shape[1:]  # (112, 112, 3)
    param_shape = h5['init_params'].shape[1:]  # (40,)
    warp_shape = h5['warp_matrices'].shape[1:]  # (2, 3)

print(f'Image shape: {img_shape}')
print(f'Param shape: {param_shape}')

# Create merged file
with h5py.File(output_file, 'w') as out_h5:
    # Pre-allocate datasets
    images = out_h5.create_dataset('images', (total_samples, *img_shape),
                                    dtype='uint8', chunks=(32, *img_shape))
    init_params = out_h5.create_dataset('init_params', (total_samples, *param_shape),
                                         dtype='float32')
    target_params = out_h5.create_dataset('target_params', (total_samples, *param_shape),
                                           dtype='float32')
    confidence = out_h5.create_dataset('confidence', (total_samples,), dtype='float32')
    warp_matrices = out_h5.create_dataset('warp_matrices', (total_samples, *warp_shape),
                                           dtype='float32')
    frame_indices = out_h5.create_dataset('frame_indices', (total_samples,), dtype='int32')
    video_indices = out_h5.create_dataset('video_indices', (total_samples,), dtype='int32')

    # Copy data
    idx = 0
    for video_idx, f in enumerate(h5_files):
        print(f'  Merging {f.name}...')
        with h5py.File(f, 'r') as h5:
            n = h5['images'].shape[0]
            images[idx:idx+n] = h5['images'][:]
            init_params[idx:idx+n] = h5['init_params'][:]
            target_params[idx:idx+n] = h5['target_params'][:]
            confidence[idx:idx+n] = h5['confidence'][:]
            warp_matrices[idx:idx+n] = h5['warp_matrices'][:]
            frame_indices[idx:idx+n] = h5['frame_indices'][:]
            video_indices[idx:idx+n] = video_idx
            idx += n

    print(f'Merged {idx} samples to {output_file}')
    print(f'File size: {os.path.getsize(output_file) / 1e9:.2f} GB')
"

echo ""
echo "============================================"
echo "Merge complete!"
echo "End time: $(date)"
echo "============================================"
