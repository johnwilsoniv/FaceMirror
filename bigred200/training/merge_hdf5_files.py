#!/usr/bin/env python3
"""
Merge Per-Video HDF5 Files into Single Training Dataset

Combines multiple HDF5 files (one per video) into a unified training dataset.

Usage:
    python merge_hdf5_files.py --input-dir output/per_video --output training_data.h5
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import time
import sys


def get_h5_sample_count(h5_path: Path) -> int:
    """Get number of samples in an HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        if 'images' in f:
            return f['images'].shape[0]
        return f.attrs.get('num_samples', 0)


def get_h5_info(h5_path: Path) -> Dict:
    """Get dataset info from HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        info = {
            'path': h5_path,
            'num_samples': get_h5_sample_count(h5_path),
            'datasets': {},
        }

        for name in ['images', 'hog_features', 'landmarks', 'global_params',
                     'local_params', 'au_intensities', 'bboxes']:
            if name in f:
                info['datasets'][name] = {
                    'shape': f[name].shape,
                    'dtype': f[name].dtype,
                }

        # Check for metadata
        info['has_metadata'] = 'metadata' in f

    return info


def merge_hdf5_files(
    input_files: List[Path],
    output_path: Path,
    chunk_size: int = 1000,
    compression: str = 'gzip',
    compression_level: int = 4,
    verbose: bool = True
) -> Dict:
    """
    Merge multiple HDF5 files into one.

    Args:
        input_files: List of input HDF5 file paths
        output_path: Output HDF5 file path
        chunk_size: Chunk size for output datasets
        compression: Compression algorithm
        compression_level: Compression level
        verbose: Print progress

    Returns:
        Statistics dictionary
    """
    if verbose:
        print(f"Merging {len(input_files)} HDF5 files...")

    # Gather info from all files
    file_info = []
    total_samples = 0

    for i, h5_path in enumerate(input_files):
        try:
            info = get_h5_info(h5_path)
            if info['num_samples'] > 0:
                file_info.append(info)
                total_samples += info['num_samples']
                if verbose and i < 10:
                    print(f"  [{i+1}] {h5_path.name}: {info['num_samples']} samples")
        except Exception as e:
            print(f"  WARNING: Skipping {h5_path.name}: {e}")

    if verbose and len(input_files) > 10:
        print(f"  ... and {len(input_files) - 10} more files")

    if not file_info:
        print("ERROR: No valid HDF5 files found!")
        return None

    if verbose:
        print(f"\nTotal samples to merge: {total_samples}")

    # Get dataset shapes from first file
    first_info = file_info[0]

    # Create output file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compression options
    comp_opts = {}
    if compression:
        comp_opts['compression'] = compression
        if compression == 'gzip':
            comp_opts['compression_opts'] = compression_level

    with h5py.File(output_path, 'w') as out_f:
        # Create datasets
        datasets = {}
        for name, ds_info in first_info['datasets'].items():
            # Get shape without first dimension
            shape = ds_info['shape'][1:]
            full_shape = (total_samples,) + shape
            chunk_shape = (min(chunk_size, total_samples),) + shape

            datasets[name] = out_f.create_dataset(
                name,
                shape=full_shape,
                dtype=ds_info['dtype'],
                chunks=chunk_shape,
                **comp_opts
            )

        # Metadata lists
        all_video_names = []
        all_frame_indices = []
        all_quality_scores = []

        # Copy data from each input file
        current_idx = 0
        for i, info in enumerate(file_info):
            if verbose and (i % 10 == 0 or i == len(file_info) - 1):
                print(f"  Processing [{i+1}/{len(file_info)}] {info['path'].name}...")

            with h5py.File(info['path'], 'r') as in_f:
                count = info['num_samples']
                end_idx = current_idx + count

                # Copy datasets
                for name in datasets.keys():
                    if name in in_f:
                        datasets[name][current_idx:end_idx] = in_f[name][:]

                # Copy metadata if available
                if info['has_metadata'] and 'metadata' in in_f:
                    if 'video_names' in in_f['metadata']:
                        names = in_f['metadata/video_names'][:]
                        if isinstance(names[0], bytes):
                            names = [n.decode('utf-8') for n in names]
                        all_video_names.extend(names)

                    if 'frame_indices' in in_f['metadata']:
                        all_frame_indices.extend(in_f['metadata/frame_indices'][:].tolist())

                    if 'quality_scores' in in_f['metadata']:
                        all_quality_scores.extend(in_f['metadata/quality_scores'][:].tolist())

                current_idx = end_idx

        # Create metadata group
        if all_video_names or all_frame_indices or all_quality_scores:
            metadata = out_f.create_group('metadata')

            if all_video_names:
                dt = h5py.special_dtype(vlen=str)
                metadata.create_dataset('video_names', data=all_video_names, dtype=dt)

            if all_frame_indices:
                metadata.create_dataset('frame_indices',
                                        data=np.array(all_frame_indices, dtype=np.int32))

            if all_quality_scores:
                metadata.create_dataset('quality_scores',
                                        data=np.array(all_quality_scores, dtype=np.float32))

        # Store attributes
        out_f.attrs['num_samples'] = total_samples
        out_f.attrs['source_files'] = len(file_info)
        out_f.attrs['image_size'] = (112, 112)
        out_f.attrs['color_format'] = 'RGB'

        # Copy AU names from first file if available
        with h5py.File(file_info[0]['path'], 'r') as first_f:
            if 'au_names' in first_f.attrs:
                out_f.attrs['au_names'] = first_f.attrs['au_names']

    # Return stats
    stats = {
        'total_samples': total_samples,
        'source_files': len(file_info),
        'output_path': str(output_path),
        'output_size_mb': output_path.stat().st_size / (1024 * 1024),
    }

    if verbose:
        print(f"\nMerge complete!")
        print(f"  Output: {output_path}")
        print(f"  Total samples: {total_samples}")
        print(f"  Source files: {len(file_info)}")
        print(f"  Output size: {stats['output_size_mb']:.1f} MB")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Merge per-video HDF5 files')
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing per-video HDF5 files')
    parser.add_argument('--output', required=True,
                        help='Output merged HDF5 file path')
    parser.add_argument('--pattern', default='video_*.h5',
                        help='Glob pattern for input files')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Chunk size for output datasets')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return 1

    # Find input files
    input_files = sorted(input_dir.glob(args.pattern))

    if not input_files:
        print(f"ERROR: No files found matching {args.pattern} in {input_dir}")
        return 1

    print(f"Found {len(input_files)} input files")

    # Merge
    start_time = time.time()
    stats = merge_hdf5_files(
        input_files,
        output_path,
        chunk_size=args.chunk_size,
        verbose=not args.quiet
    )
    elapsed = time.time() - start_time

    if stats:
        print(f"\nTotal time: {elapsed:.1f}s")
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
