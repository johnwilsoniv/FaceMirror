#!/usr/bin/env python3
"""
HPC Accurate AU Pipeline - Baseline for Accuracy Validation

This pipeline prioritizes ACCURACY over speed by:
1. Disabling all speed optimizations
2. Using same settings as local Python pipeline
3. Processing frames sequentially

SPEED OPTIMIZATIONS DISABLED (can be re-enabled later):
- [DISABLED] Face tracking (track_faces=False)
- [DISABLED] Reduced CLNF iterations
- [DISABLED] Early convergence exit
- [DISABLED] Batched AU predictor

Once accuracy is validated, optimizations can be re-enabled one by one.

Usage:
    python hpc_accurate_pipeline.py --video input.mp4 --output results.csv
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
    parser = argparse.ArgumentParser(description="HPC Accurate AU Pipeline (Baseline)")
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', help='Output CSV file')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    # Backend selection - use 'onnx' for HPC, 'coreml' for Mac
    parser.add_argument('--backend', default='auto',
                        choices=['auto', 'onnx', 'coreml', 'cpu'],
                        help='MTCNN backend (default: auto)')

    args = parser.parse_args()

    # Import pipeline
    from pyfaceau.pipeline import FullPythonAUPipeline

    # Set default output
    if not args.output:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_accurate_aus.csv")

    # Model paths - try multiple locations
    pdm_file = project_root / "pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
    au_models_dir = project_root / "pyfaceau/weights/AU_predictors"
    triangulation_file = project_root / "pyfaceau/weights/tris_68_full.txt"

    # Check paths and try alternates
    if not pdm_file.exists():
        pdm_file = project_root / "pyclnf/pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
    if not au_models_dir.exists():
        au_models_dir = project_root / "pyclnf/pyfaceau/weights/AU_predictors"
    if not triangulation_file.exists():
        triangulation_file = project_root / "pyclnf/pyfaceau/weights/tris_68_full.txt"

    print("=" * 70)
    print("HPC ACCURATE AU PIPELINE (BASELINE)")
    print("=" * 70)
    print(f"Video: {args.video}")
    print(f"Output: {args.output}")
    print(f"Backend: {args.backend}")
    print()
    print("ACCURACY MODE: All speed optimizations DISABLED")
    print("  - Face tracking: DISABLED (re-detect every frame)")
    print("  - CLNF iterations: 10 (maximum accuracy)")
    print("  - Convergence threshold: 0.01 (strict)")
    print("  - Batched predictor: ENABLED (doesn't affect accuracy)")
    print()

    # Determine backend
    # On HPC (Linux), use ONNX; on Mac, auto will select CoreML
    if args.backend == 'auto':
        import platform
        if platform.system() == 'Darwin':
            backend = 'coreml'  # Mac - use CoreML
        else:
            backend = 'onnx'  # Linux/HPC - use ONNX
        print(f"Auto-detected backend: {backend}")
    else:
        backend = args.backend

    # Initialize pipeline with ACCURACY settings (speed opts disabled)
    pipeline = FullPythonAUPipeline(
        pdm_file=str(pdm_file),
        au_models_dir=str(au_models_dir),
        triangulation_file=str(triangulation_file),
        patch_expert_file="",  # CLNF uses its own models

        # Backend
        mtcnn_backend=backend,

        # ACCURACY SETTINGS (speed optimizations disabled)
        use_calc_params=True,           # Full pose estimation
        track_faces=False,              # [SPEED OPT DISABLED] Re-detect every frame
        use_batched_predictor=True,     # This doesn't affect accuracy

        # CLNF settings for maximum accuracy
        max_clnf_iterations=10,         # [SPEED OPT DISABLED] Use full iterations
        clnf_convergence_threshold=0.005,  # Gold standard for sub-pixel accuracy

        verbose=args.verbose
    )

    # Process video
    print()
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
    print(f"Time: {elapsed:.1f}s ({fps:.2f} FPS)")
    print(f"Output: {args.output}")
    print()

    # Print AU summary
    au_cols = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]
    if au_cols:
        print("AU Summary (first 5):")
        for au_col in sorted(au_cols)[:5]:
            mean_val = df[df['success']][au_col].mean()
            max_val = df[df['success']][au_col].max()
            print(f"  {au_col}: mean={mean_val:.3f}, max={max_val:.3f}")
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
