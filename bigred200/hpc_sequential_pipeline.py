#!/usr/bin/env python3
"""
HPC Sequential AU Pipeline - Single Video Processing

This is a simple wrapper around the local Python pipeline for HPC use.
It processes frames sequentially for maximum accuracy.

For batch processing of multiple videos, use hpc_batch_pipeline.py instead.

Usage:
    python hpc_sequential_pipeline.py --video input.mp4 --output results.csv
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "pyclnf"))
sys.path.insert(0, str(project_root / "pyfaceau"))
sys.path.insert(0, str(project_root / "pymtcnn"))

# Limit threads (important for HPC)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['ORT_NUM_THREADS'] = '1'


def main():
    parser = argparse.ArgumentParser(description="HPC Sequential AU Pipeline")
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', help='Output CSV file')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--profile', default='optimized',
                        choices=['accurate', 'optimized', 'fast'],
                        help='Convergence profile')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Import pipeline
    from pyfaceau.pipeline import FullPythonAUPipeline

    # Set default output
    if not args.output:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_aus.csv")

    # Model paths
    pdm_file = project_root / "pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
    au_models_dir = project_root / "pyfaceau/weights/AU_predictors"
    triangulation_file = project_root / "pyfaceau/weights/tris_68_full.txt"

    # Check paths
    if not pdm_file.exists():
        pdm_file = project_root / "pyclnf/pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
    if not au_models_dir.exists():
        au_models_dir = project_root / "pyclnf/pyfaceau/weights/AU_predictors"
    if not triangulation_file.exists():
        triangulation_file = project_root / "pyclnf/pyfaceau/weights/tris_68_full.txt"

    print("=" * 70)
    print("HPC SEQUENTIAL AU PIPELINE")
    print("=" * 70)
    print(f"Video: {args.video}")
    print(f"Output: {args.output}")
    print(f"Profile: {args.profile}")
    print()

    # Initialize pipeline
    max_iterations = {'accurate': 10, 'optimized': 7, 'fast': 5}[args.profile]

    pipeline = FullPythonAUPipeline(
        pdm_file=str(pdm_file),
        au_models_dir=str(au_models_dir),
        triangulation_file=str(triangulation_file),
        patch_expert_file="",
        mtcnn_backend='onnx',  # ONNX for HPC (no CoreML)
        use_calc_params=True,
        track_faces=False,
        use_batched_predictor=True,
        max_clnf_iterations=max_iterations,
        clnf_convergence_threshold=0.005,  # Gold standard for sub-pixel accuracy
        verbose=args.verbose
    )

    # Process video
    start = time.time()
    df = pipeline.process_video(
        video_path=args.video,
        output_csv=args.output,
        max_frames=args.max_frames
    )
    elapsed = time.time() - start

    success_count = df['success'].sum()
    fps = success_count / elapsed

    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Frames: {success_count}/{len(df)} successful")
    print(f"Time: {elapsed:.1f}s ({fps:.1f} FPS)")
    print(f"Output: {args.output}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
