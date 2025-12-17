#!/usr/bin/env python3
"""
Merge individual C++ training data HDF5 files into a single training dataset.

Usage:
    python merge_cpp_training_data.py --input-dir cpp_training_data --output cpp_training_merged.h5
"""
import argparse
import h5py
import numpy as np
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description='Merge C++ training data files')
    parser.add_argument('--input-dir', default='cpp_training_data', help='Input directory with HDF5 files')
    parser.add_argument('--output', default='cpp_training_merged.h5', help='Output merged file')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    h5_files = sorted(input_dir.glob('cpp_training_video_*.h5'))

    print(f"Found {len(h5_files)} HDF5 files to merge")

    if len(h5_files) == 0:
        print("ERROR: No files found!")
        sys.exit(1)

    # Collect all data
    all_aligned_faces = []
    all_landmarks = []
    all_landmarks_aligned = []
    all_pose_params = []
    all_au_intensities = []
    all_warp_matrices = []
    all_frame_indices = []
    all_confidences = []
    all_video_indices = []

    for h5_file in h5_files:
        print(f"  Loading {h5_file.name}...", end=" ", flush=True)
        try:
            with h5py.File(h5_file, 'r') as f:
                n_samples = f['aligned_faces'].shape[0]
                video_idx = f.attrs.get('video_index', 0)

                all_aligned_faces.append(f['aligned_faces'][:])
                all_landmarks.append(f['landmarks'][:])
                all_landmarks_aligned.append(f['landmarks_aligned'][:])
                all_pose_params.append(f['pose_params'][:])
                all_au_intensities.append(f['au_intensities'][:])
                all_warp_matrices.append(f['warp_matrices'][:])
                all_frame_indices.append(f['frame_indices'][:])
                all_confidences.append(f['confidences'][:])
                all_video_indices.append(np.full(n_samples, video_idx, dtype=np.int32))

                print(f"{n_samples} samples")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Concatenate all data
    print("\nConcatenating data...")
    aligned_faces = np.concatenate(all_aligned_faces, axis=0)
    landmarks = np.concatenate(all_landmarks, axis=0)
    landmarks_aligned = np.concatenate(all_landmarks_aligned, axis=0)
    pose_params = np.concatenate(all_pose_params, axis=0)
    au_intensities = np.concatenate(all_au_intensities, axis=0)
    warp_matrices = np.concatenate(all_warp_matrices, axis=0)
    frame_indices = np.concatenate(all_frame_indices, axis=0)
    confidences = np.concatenate(all_confidences, axis=0)
    video_indices = np.concatenate(all_video_indices, axis=0)

    total_samples = aligned_faces.shape[0]
    print(f"Total samples: {total_samples}")

    # Save merged file
    print(f"\nSaving to {args.output}...")
    AU_NAMES = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
                'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
                'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

    with h5py.File(args.output, 'w') as f:
        f.create_dataset('aligned_faces', data=aligned_faces, compression='gzip', compression_opts=4)
        f.create_dataset('landmarks', data=landmarks)
        f.create_dataset('landmarks_aligned', data=landmarks_aligned)
        f.create_dataset('pose_params', data=pose_params)
        f.create_dataset('au_intensities', data=au_intensities)
        f.create_dataset('warp_matrices', data=warp_matrices)
        f.create_dataset('frame_indices', data=frame_indices)
        f.create_dataset('confidences', data=confidences)
        f.create_dataset('video_indices', data=video_indices)

        f.attrs['source'] = 'C++ OpenFace (merged)'
        f.attrs['num_samples'] = total_samples
        f.attrs['num_videos'] = len(h5_files)
        f.attrs['au_names'] = AU_NAMES

    import os
    file_size = os.path.getsize(args.output) / 1024 / 1024 / 1024
    print(f"Saved: {args.output} ({file_size:.2f} GB)")
    print(f"\nSummary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Videos merged: {len(h5_files)}")
    print(f"  Unique videos: {len(np.unique(video_indices))}")


if __name__ == '__main__':
    main()
