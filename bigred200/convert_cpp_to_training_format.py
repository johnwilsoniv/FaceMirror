#!/usr/bin/env python3
"""
Convert C++ training data format to match expected training script format.

C++ format:
- aligned_faces, landmarks, landmarks_aligned, pose_params, au_intensities, warp_matrices, confidences, video_indices

Expected format:
- images, landmarks, global_params, local_params, au_intensities, warp_matrices

This creates a new HDF5 file with the expected field names.
"""
import argparse
import h5py
import numpy as np
from pathlib import Path


def convert_cpp_to_training_format(input_path: str, output_path: str):
    """Convert C++ training data to expected training format."""

    print(f"Loading {input_path}...")
    with h5py.File(input_path, 'r') as f_in:
        print(f"  Input datasets: {list(f_in.keys())}")
        print(f"  Input attrs: {dict(f_in.attrs)}")

        # Load data
        aligned_faces = f_in['aligned_faces'][:]
        landmarks_aligned = f_in['landmarks_aligned'][:]
        pose_params = f_in['pose_params'][:]
        au_intensities = f_in['au_intensities'][:]
        warp_matrices = f_in['warp_matrices'][:]
        confidences = f_in['confidences'][:]
        video_indices = f_in['video_indices'][:]

        num_samples = aligned_faces.shape[0]
        print(f"  Samples: {num_samples}")

        au_names = list(f_in.attrs['au_names'])

    print(f"\nConverting to training format...")

    # Convert pose_params to global_params format
    # C++ pose_params: [Tx, Ty, Tz, Rx, Ry, Rz] (mm and radians)
    # Expected global_params: [scale, rx, ry, rz, tx, ty] (pixels and radians)
    # Note: We'll use a nominal scale since we don't have it directly
    global_params = np.zeros((num_samples, 6), dtype=np.float32)
    global_params[:, 0] = 1.0  # scale (nominal, face is already aligned)
    global_params[:, 1] = pose_params[:, 3]  # rx
    global_params[:, 2] = pose_params[:, 4]  # ry
    global_params[:, 3] = pose_params[:, 5]  # rz
    global_params[:, 4] = pose_params[:, 0]  # tx (from Tx)
    global_params[:, 5] = pose_params[:, 1]  # ty (from Ty)

    # Create dummy local_params (34 zeros - PDM shape parameters)
    # Not used for AU prediction, but needed for compatibility
    local_params = np.zeros((num_samples, 34), dtype=np.float32)

    print(f"Saving to {output_path}...")
    with h5py.File(output_path, 'w') as f_out:
        # Store with expected names
        f_out.create_dataset('images', data=aligned_faces, compression='gzip', compression_opts=4)
        f_out.create_dataset('landmarks', data=landmarks_aligned)
        f_out.create_dataset('global_params', data=global_params)
        f_out.create_dataset('local_params', data=local_params)
        f_out.create_dataset('au_intensities', data=au_intensities)
        f_out.create_dataset('warp_matrices', data=warp_matrices)
        f_out.create_dataset('confidences', data=confidences)
        f_out.create_dataset('video_indices', data=video_indices)

        # Attributes
        f_out.attrs['num_samples'] = num_samples
        f_out.attrs['num_videos'] = len(np.unique(video_indices))
        f_out.attrs['au_names'] = au_names
        f_out.attrs['source'] = 'C++ OpenFace (converted)'

    import os
    file_size = os.path.getsize(output_path) / 1024 / 1024 / 1024
    print(f"\nDone! Output: {output_path} ({file_size:.2f} GB)")
    print(f"  Samples: {num_samples}")
    print(f"  Videos: {len(np.unique(video_indices))}")


def main():
    parser = argparse.ArgumentParser(description='Convert C++ training data format')
    parser.add_argument('--input', default='cpp_training_merged.h5', help='Input C++ HDF5 file')
    parser.add_argument('--output', default='cpp_training_converted.h5', help='Output converted HDF5 file')
    args = parser.parse_args()

    convert_cpp_to_training_format(args.input, args.output)


if __name__ == '__main__':
    main()
