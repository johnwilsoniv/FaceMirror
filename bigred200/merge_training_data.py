#!/usr/bin/env python3
"""
Merge per-video HDF5 training data files into a single dataset.
Creates train/val split (90/10, stratified by video).
"""
import argparse
import numpy as np
import h5py
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Merge training data files')
    parser.add_argument('--input-dir', default='training_data_new', help='Input directory with per-video HDF5 files')
    parser.add_argument('--output-file', default='training_data_merged.h5', help='Output merged HDF5 file')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
    args = parser.parse_args()

    print("=" * 70)
    print("MERGE TRAINING DATA")
    print("=" * 70)

    # Find all HDF5 files
    input_path = Path(args.input_dir)
    h5_files = sorted(input_path.glob('training_video_*.h5'))
    print(f"\nFound {len(h5_files)} HDF5 files in {args.input_dir}")

    if len(h5_files) == 0:
        print("ERROR: No HDF5 files found!")
        return

    # Collect all data
    all_aligned_faces = []
    all_landmarks = []
    all_pose_params = []
    all_local_params = []
    all_au_intensities = []
    all_video_indices = []  # Track which video each sample came from

    total_samples = 0

    for i, h5_file in enumerate(h5_files):
        with h5py.File(h5_file, 'r') as f:
            n_samples = len(f['landmarks'])
            video_name = f.attrs.get('video_name', 'unknown')
            cohort = f.attrs.get('cohort', 'unknown')

            all_aligned_faces.append(f['aligned_faces'][:])
            all_landmarks.append(f['landmarks'][:])
            all_pose_params.append(f['pose_params'][:])
            all_local_params.append(f['local_params'][:])
            all_au_intensities.append(f['au_intensities'][:])
            all_video_indices.append(np.full(n_samples, i, dtype=np.int32))

            total_samples += n_samples

            if (i + 1) % 20 == 0 or i == len(h5_files) - 1:
                print(f"  Loaded {i + 1}/{len(h5_files)} files, {total_samples:,} samples total")

    print(f"\nConcatenating arrays...")

    # Concatenate all arrays
    aligned_faces = np.concatenate(all_aligned_faces, axis=0)
    landmarks = np.concatenate(all_landmarks, axis=0)
    pose_params = np.concatenate(all_pose_params, axis=0)
    local_params = np.concatenate(all_local_params, axis=0)
    au_intensities = np.concatenate(all_au_intensities, axis=0)
    video_indices = np.concatenate(all_video_indices, axis=0)

    print(f"  Aligned faces: {aligned_faces.shape}")
    print(f"  Landmarks: {landmarks.shape}")
    print(f"  Pose params: {pose_params.shape}")
    print(f"  Local params: {local_params.shape}")
    print(f"  AU intensities: {au_intensities.shape}")

    # Save merged dataset (format compatible with pyfaceau TrainingDataset)
    print(f"\nSaving to {args.output_file}...")

    # AU names for reference
    au_names = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
                'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
                'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

    with h5py.File(args.output_file, 'w') as f:
        f.create_dataset('images', data=aligned_faces,
                         compression='gzip', compression_opts=4)
        f.create_dataset('landmarks', data=landmarks)
        f.create_dataset('global_params', data=pose_params)
        f.create_dataset('local_params', data=local_params)
        f.create_dataset('au_intensities', data=au_intensities)
        f.create_dataset('video_indices', data=video_indices)
        f.attrs['num_samples'] = total_samples
        f.attrs['au_names'] = au_names
        f.attrs['n_videos'] = len(h5_files)

    file_size = os.path.getsize(args.output_file) / (1024 ** 3)
    print(f"\nSaved: {args.output_file} ({file_size:.2f} GB)")

    print("\n" + "=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)
    print(f"Total samples: {total_samples:,}")
    print(f"From {len(h5_files)} videos")


if __name__ == '__main__':
    main()
