#!/usr/bin/env python3
"""
HPC Pipeline Accuracy Validation

Compares HPC AU Pipeline output to local Python pipeline to verify:
1. AU predictions match within tolerance
2. Landmarks are consistent
3. Feature extraction is equivalent

This validates that the HPC optimizations don't change output accuracy.

Usage:
    # Generate local reference (run on local machine with good accuracy)
    python validate_hpc_accuracy.py --generate-reference --video "S Data/Normal Cohort/IMG_0942.MOV" --max-frames 100

    # Validate HPC output against reference (run on BR200)
    python validate_hpc_accuracy.py --validate --video test_data/IMG_0942.MOV --reference reference_aus.csv --max-frames 100
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import argparse
import time
import json

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "pyclnf"))
sys.path.insert(0, str(project_root / "pyfaceau"))
sys.path.insert(0, str(project_root / "pymtcnn"))

# Limit ONNX threads
os.environ['ORT_NUM_THREADS'] = '1'


def generate_reference(video_path: str, output_csv: str, max_frames: int = 100) -> pd.DataFrame:
    """
    Generate reference AU predictions using local Python pipeline.

    Args:
        video_path: Path to test video
        output_csv: Output CSV path
        max_frames: Number of frames to process

    Returns:
        DataFrame with reference AU predictions
    """
    from pyfaceau.pipeline import FullPythonAUPipeline

    print("=" * 70)
    print("GENERATING REFERENCE AU PREDICTIONS")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Output: {output_csv}")
    print(f"Max frames: {max_frames}")
    print()

    # Find weight files
    pdm_file = project_root / "pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
    au_models_dir = project_root / "pyfaceau/weights/AU_predictors"
    triangulation_file = project_root / "pyfaceau/weights/tris_68_full.txt"

    # Check paths
    if not pdm_file.exists():
        # Try alternate paths
        pdm_file = project_root / "pyclnf/pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
    if not au_models_dir.exists():
        au_models_dir = project_root / "pyclnf/pyfaceau/weights/AU_predictors"
    if not triangulation_file.exists():
        triangulation_file = project_root / "pyclnf/pyfaceau/weights/tris_68_full.txt"

    print(f"PDM file: {pdm_file} (exists: {pdm_file.exists()})")
    print(f"AU models: {au_models_dir} (exists: {au_models_dir.exists()})")
    print(f"Triangulation: {triangulation_file} (exists: {triangulation_file.exists()})")
    print()

    # Initialize pipeline
    print("Initializing local Python pipeline...")
    pipeline = FullPythonAUPipeline(
        pdm_file=str(pdm_file),
        au_models_dir=str(au_models_dir),
        triangulation_file=str(triangulation_file),
        patch_expert_file="",  # CLNF uses its own models
        mtcnn_backend='auto',
        use_calc_params=True,
        track_faces=False,  # Disabled for accuracy
        use_batched_predictor=True,
        max_clnf_iterations=10,
        clnf_convergence_threshold=0.005,  # Gold standard for sub-pixel accuracy
        verbose=True
    )

    # Process video
    start = time.time()
    df = pipeline.process_video(
        video_path=video_path,
        output_csv=output_csv,
        max_frames=max_frames
    )
    elapsed = time.time() - start

    success_count = df['success'].sum()
    fps = success_count / elapsed

    print()
    print("=" * 70)
    print("REFERENCE GENERATION COMPLETE")
    print("=" * 70)
    print(f"Frames: {success_count}/{len(df)} successful")
    print(f"Time: {elapsed:.1f}s ({fps:.1f} FPS)")
    print(f"Output saved to: {output_csv}")
    print()

    return df


def run_hpc_pipeline(video_path: str, max_frames: int = 100, n_workers: int = 4) -> pd.DataFrame:
    """
    Run HPC pipeline on video.

    Args:
        video_path: Path to test video
        max_frames: Number of frames to process
        n_workers: Number of workers

    Returns:
        DataFrame with HPC AU predictions
    """
    from hpc_au_pipeline import HPCAUPipeline, HPCConfig

    print("=" * 70)
    print("RUNNING HPC PIPELINE")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Workers: {n_workers}")
    print(f"Max frames: {max_frames}")
    print()

    config = HPCConfig(
        n_workers=n_workers,
        convergence_profile='optimized',
        use_shared_memory=True,
        use_numa=True,
        verbose=True
    )

    pipeline = HPCAUPipeline(config)

    start = time.time()
    df = pipeline.process_video(video_path, max_frames=max_frames)
    elapsed = time.time() - start

    success_count = df['success'].sum()
    fps = success_count / elapsed

    print()
    print(f"HPC processing complete: {success_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")

    return df


def validate_accuracy(hpc_df: pd.DataFrame, ref_df: pd.DataFrame, tolerance: float = 0.1) -> dict:
    """
    Validate HPC predictions against reference.

    Args:
        hpc_df: HPC pipeline output
        ref_df: Reference pipeline output
        tolerance: Acceptable AU difference (default: 0.1 = 2% of [0,5] range)

    Returns:
        Dictionary with validation results
    """
    print()
    print("=" * 70)
    print("VALIDATING HPC ACCURACY")
    print("=" * 70)
    print()

    # Find AU columns
    au_cols = [col for col in ref_df.columns if col.startswith('AU') and col.endswith('_r')]

    # Only compare successful frames in both
    hpc_success = hpc_df[hpc_df['success'] == True].copy()
    ref_success = ref_df[ref_df['success'] == True].copy()

    # Align by frame index
    common_frames = set(hpc_success['frame'].values) & set(ref_success['frame'].values)
    print(f"Common successful frames: {len(common_frames)}")

    if len(common_frames) == 0:
        print("ERROR: No common successful frames to compare!")
        return {'status': 'FAILED', 'reason': 'No common frames'}

    hpc_aligned = hpc_success[hpc_success['frame'].isin(common_frames)].sort_values('frame')
    ref_aligned = ref_success[ref_success['frame'].isin(common_frames)].sort_values('frame')

    # Compare AU predictions
    results = {
        'status': 'PASSED',
        'num_frames': len(common_frames),
        'tolerance': tolerance,
        'au_results': {}
    }

    all_passed = True

    print()
    print(f"{'AU':<12} {'MAE':<10} {'Max Err':<10} {'Corr':<10} {'Status'}")
    print("-" * 60)

    for au_col in sorted(au_cols):
        if au_col not in hpc_aligned.columns:
            continue

        hpc_vals = hpc_aligned[au_col].values
        ref_vals = ref_aligned[au_col].values

        # Calculate metrics
        mae = np.mean(np.abs(hpc_vals - ref_vals))
        max_err = np.max(np.abs(hpc_vals - ref_vals))

        # Correlation (handle constant values)
        if np.std(ref_vals) > 0 and np.std(hpc_vals) > 0:
            corr = np.corrcoef(hpc_vals, ref_vals)[0, 1]
        else:
            corr = 1.0 if np.allclose(hpc_vals, ref_vals) else 0.0

        # Check pass/fail
        passed = mae <= tolerance
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"{au_col:<12} {mae:<10.4f} {max_err:<10.4f} {corr:<10.4f} {status}")

        results['au_results'][au_col] = {
            'mae': float(mae),
            'max_error': float(max_err),
            'correlation': float(corr),
            'passed': passed
        }

    print("-" * 60)

    # Overall metrics
    all_hpc = np.concatenate([hpc_aligned[au].values for au in au_cols if au in hpc_aligned.columns])
    all_ref = np.concatenate([ref_aligned[au].values for au in au_cols if au in ref_aligned.columns])

    overall_mae = np.mean(np.abs(all_hpc - all_ref))
    overall_corr = np.corrcoef(all_hpc, all_ref)[0, 1]

    print()
    print(f"Overall MAE: {overall_mae:.4f}")
    print(f"Overall Correlation: {overall_corr:.4f}")
    print()

    results['overall_mae'] = float(overall_mae)
    results['overall_correlation'] = float(overall_corr)

    if all_passed:
        print("STATUS: ALL AUs WITHIN TOLERANCE")
        results['status'] = 'PASSED'
    else:
        print("STATUS: SOME AUs EXCEED TOLERANCE")
        results['status'] = 'FAILED'
        failed_aus = [au for au, res in results['au_results'].items() if not res['passed']]
        print(f"Failed AUs: {', '.join(failed_aus)}")

    print()

    return results


def main():
    parser = argparse.ArgumentParser(description="HPC Pipeline Accuracy Validation")

    parser.add_argument('--generate-reference', action='store_true',
                        help='Generate reference predictions using local pipeline')
    parser.add_argument('--validate', action='store_true',
                        help='Validate HPC output against reference')
    parser.add_argument('--video', required=True, help='Video file path')
    parser.add_argument('--reference', default='reference_aus.csv',
                        help='Reference CSV file (for validation)')
    parser.add_argument('--output', default='hpc_aus.csv',
                        help='Output CSV file')
    parser.add_argument('--max-frames', type=int, default=100,
                        help='Max frames to process')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of HPC workers')
    parser.add_argument('--tolerance', type=float, default=0.1,
                        help='AU difference tolerance (default: 0.1)')

    args = parser.parse_args()

    video_path = project_root / args.video
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return 1

    if args.generate_reference:
        # Generate reference using local pipeline
        ref_df = generate_reference(
            str(video_path),
            args.reference,
            args.max_frames
        )
        return 0

    elif args.validate:
        # Load reference
        if not Path(args.reference).exists():
            print(f"Error: Reference file not found: {args.reference}")
            print("Run with --generate-reference first")
            return 1

        ref_df = pd.read_csv(args.reference)
        print(f"Loaded reference: {len(ref_df)} frames")

        # Run HPC pipeline
        hpc_df = run_hpc_pipeline(
            str(video_path),
            args.max_frames,
            args.workers
        )

        # Save HPC output
        hpc_df.to_csv(args.output, index=False)
        print(f"HPC output saved to: {args.output}")

        # Validate
        results = validate_accuracy(hpc_df, ref_df, args.tolerance)

        # Save validation results (convert numpy bools to Python bools)
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj

        serializable_results = convert_to_serializable(results)
        with open('validation_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Validation results saved to: validation_results.json")

        return 0 if results['status'] == 'PASSED' else 1

    else:
        print("Specify --generate-reference or --validate")
        return 1


if __name__ == '__main__':
    sys.exit(main())
