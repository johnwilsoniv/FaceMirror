#!/bin/bash
#SBATCH --job-name=merge_bbox
#SBATCH --output=merge_bbox_%j.out
#SBATCH --error=merge_bbox_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=r01984
#SBATCH --partition=debug

# Merge BBOX training data files

set -e

module purge
module load python/3.12.11

cd "$HOME/pyfaceau"

export PYTHONPATH="$PWD/pyfaceau:$PYTHONPATH"

echo "============================================"
echo "Merge BBOX Training Data"
echo "============================================"
echo "Start time: $(date)"

python3 -c "
import h5py
import numpy as np
from pathlib import Path
import os

input_dir = Path('training_data_bbox')
output_file = 'bbox_training_merged.h5'

# Find all HDF5 files
h5_files = sorted(input_dir.glob('training_video_*.h5'))
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
    img_shape = h5['images'].shape[1:]
    lm_shape = h5['landmarks'].shape[1:]
    gp_shape = h5['global_params'].shape[1:]
    lp_shape = h5['local_params'].shape[1:]
    wm_shape = h5['warp_matrices'].shape[1:]
    au_shape = h5['au_intensities'].shape[1:]

print(f'Image shape: {img_shape}')
print(f'Landmark shape: {lm_shape}')

# Create merged file
with h5py.File(output_file, 'w') as out_h5:
    # Pre-allocate datasets
    images = out_h5.create_dataset('images', (total_samples, *img_shape), dtype='uint8', chunks=(32, *img_shape))
    landmarks = out_h5.create_dataset('landmarks', (total_samples, *lm_shape), dtype='float32')
    global_params = out_h5.create_dataset('global_params', (total_samples, *gp_shape), dtype='float32')
    local_params = out_h5.create_dataset('local_params', (total_samples, *lp_shape), dtype='float32')
    warp_matrices = out_h5.create_dataset('warp_matrices', (total_samples, *wm_shape), dtype='float32')
    au_intensities = out_h5.create_dataset('au_intensities', (total_samples, *au_shape), dtype='float32')

    # Copy data
    idx = 0
    for f in h5_files:
        print(f'  Merging {f.name}...')
        with h5py.File(f, 'r') as h5:
            n = h5['images'].shape[0]
            images[idx:idx+n] = h5['images'][:]
            landmarks[idx:idx+n] = h5['landmarks'][:]
            global_params[idx:idx+n] = h5['global_params'][:]
            local_params[idx:idx+n] = h5['local_params'][:]
            warp_matrices[idx:idx+n] = h5['warp_matrices'][:]
            au_intensities[idx:idx+n] = h5['au_intensities'][:]
            idx += n

    print(f'Merged {idx} samples to {output_file}')
    print(f'File size: {os.path.getsize(output_file) / 1e9:.2f} GB')
"

echo ""
echo "============================================"
echo "Merge complete!"
echo "End time: $(date)"
echo "============================================"
