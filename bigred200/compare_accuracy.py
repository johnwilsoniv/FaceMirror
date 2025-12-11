#!/usr/bin/env python3
"""
Compare Python pyfaceau Pipeline vs C++ OpenFace

This script:
1. Processes a full video with the Python AU pipeline
2. Loads C++ reference CSV (landmarks + AUs)
3. Computes accuracy metrics (landmark error, AU correlation, AU MAE)
4. Outputs detailed comparison report

Usage:
    python compare_accuracy.py --video input.mp4 --cpp-ref cpp_output.csv --output results/

    # With SLURM array jobs:
    python compare_accuracy.py --video-index $SLURM_ARRAY_TASK_ID
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from scipy import stats

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "pyclnf"))
sys.path.insert(0, str(project_root / "pyfaceau"))
sys.path.insert(0, str(project_root / "pymtcnn"))

# Limit threads for HPC
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['ORT_NUM_THREADS'] = '1'


# AU intensity columns (17 total)
AU_INTENSITY_COLS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
    'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
    'AU25_r', 'AU26_r', 'AU45_r'
]

# Landmark region indices
REGIONS = {
    'jaw': list(range(0, 17)),
    'brows': list(range(17, 27)),
    'nose': list(range(27, 36)),
    'eyes': list(range(36, 48)),
    'mouth': list(range(48, 68))
}


def load_cpp_reference(csv_path: str) -> pd.DataFrame:
    """Load C++ OpenFace CSV with landmarks and AUs."""
    df = pd.read_csv(csv_path)

    # Strip whitespace from column names (OpenFace adds space padding)
    df.columns = df.columns.str.strip()

    # Verify required columns
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]

    missing_x = [c for c in x_cols if c not in df.columns]
    missing_y = [c for c in y_cols if c not in df.columns]

    if missing_x or missing_y:
        raise ValueError(f"Missing landmark columns: x={len(missing_x)}, y={len(missing_y)}")

    # Check for AU columns
    au_cols_present = [c for c in AU_INTENSITY_COLS if c in df.columns]
    if len(au_cols_present) < len(AU_INTENSITY_COLS):
        print(f"Warning: Only {len(au_cols_present)}/{len(AU_INTENSITY_COLS)} AU columns found")

    return df


def process_video_python(
    video_path: str,
    output_csv: str = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Process video with Python pyfaceau pipeline.

    Uses 'accurate' convergence profile for best quality.
    """
    from pyfaceau.pipeline import FullPythonAUPipeline

    # Model paths
    pdm_file = project_root / "pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
    au_models_dir = project_root / "pyfaceau/weights/AU_predictors"
    tri_file = project_root / "pyfaceau/weights/tris_68_full.txt"
    patch_file = project_root / "pyclnf/pyclnf/models/main_ceclm_general.txt"

    # Check paths
    for p, name in [(pdm_file, 'PDM'), (au_models_dir, 'AU models'), (tri_file, 'triangulation')]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    if verbose:
        print(f"Processing: {video_path}")

    # Initialize pipeline with accurate profile
    pipeline = FullPythonAUPipeline(
        pdm_file=str(pdm_file),
        au_models_dir=str(au_models_dir),
        triangulation_file=str(tri_file),
        patch_expert_file=str(patch_file) if patch_file.exists() else "",
        mtcnn_backend='onnx',  # ONNX for Linux HPC
        use_calc_params=True,
        track_faces=False,  # Disabled for best accuracy
        use_batched_predictor=True,
        max_clnf_iterations=10,  # Full iterations
        clnf_convergence_threshold=0.005,  # Gold standard
        verbose=verbose
    )

    # Process video
    start = time.time()
    df = pipeline.process_video(
        video_path=video_path,
        output_csv=output_csv
    )
    elapsed = time.time() - start

    if verbose:
        print(f"Processed {len(df)} frames in {elapsed:.1f}s ({len(df)/elapsed:.1f} FPS)")

    return df


def extract_landmarks(df: pd.DataFrame) -> np.ndarray:
    """Extract 68x2 landmarks from dataframe.

    Returns: (n_frames, 68, 2) array
    """
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]

    n_frames = len(df)
    landmarks = np.zeros((n_frames, 68, 2), dtype=np.float32)

    landmarks[:, :, 0] = df[x_cols].values
    landmarks[:, :, 1] = df[y_cols].values

    return landmarks


def compute_landmark_errors(py_lm: np.ndarray, cpp_lm: np.ndarray) -> dict:
    """Compute per-frame and per-region landmark errors.

    Args:
        py_lm: (n, 68, 2) Python landmarks
        cpp_lm: (n, 68, 2) C++ landmarks

    Returns:
        dict with error metrics
    """
    # Euclidean error per landmark per frame
    diff = py_lm - cpp_lm  # (n, 68, 2)
    per_lm_error = np.sqrt(diff[:, :, 0]**2 + diff[:, :, 1]**2)  # (n, 68)

    results = {
        'overall_mean': float(per_lm_error.mean()),
        'overall_std': float(per_lm_error.std()),
        'overall_max': float(per_lm_error.max()),
        'per_frame_mean': per_lm_error.mean(axis=1).tolist(),
    }

    # Per-region errors
    for region_name, indices in REGIONS.items():
        region_error = per_lm_error[:, indices]
        results[f'{region_name}_mean'] = float(region_error.mean())
        results[f'{region_name}_std'] = float(region_error.std())
        results[f'{region_name}_max'] = float(region_error.max())

    return results


def compute_au_metrics(py_df: pd.DataFrame, cpp_df: pd.DataFrame) -> dict:
    """Compute AU correlation and MAE between Python and C++.

    Returns:
        dict with per-AU correlation and MAE
    """
    results = {
        'correlations': {},
        'mae': {},
        'mean_diff': {},
    }

    # Find common AU columns
    au_cols = [c for c in AU_INTENSITY_COLS
               if c in py_df.columns and c in cpp_df.columns]

    if not au_cols:
        print("Warning: No AU columns found for comparison")
        return results

    correlations = []
    maes = []

    for au_col in au_cols:
        py_au = py_df[au_col].values
        cpp_au = cpp_df[au_col].values

        # Handle potential length mismatch
        min_len = min(len(py_au), len(cpp_au))
        py_au = py_au[:min_len]
        cpp_au = cpp_au[:min_len]

        # Correlation (if variance exists)
        if py_au.std() > 0 and cpp_au.std() > 0:
            corr, _ = stats.pearsonr(py_au, cpp_au)
            results['correlations'][au_col] = float(corr)
            correlations.append(corr)
        else:
            results['correlations'][au_col] = None

        # MAE
        mae = np.abs(py_au - cpp_au).mean()
        results['mae'][au_col] = float(mae)
        maes.append(mae)

        # Mean difference (bias)
        results['mean_diff'][au_col] = float(py_au.mean() - cpp_au.mean())

    # Summary statistics
    valid_corrs = [c for c in correlations if c is not None and not np.isnan(c)]
    results['avg_correlation'] = float(np.mean(valid_corrs)) if valid_corrs else None
    results['avg_mae'] = float(np.mean(maes)) if maes else None

    return results


def compare_single_video(
    video_path: str,
    cpp_ref_csv: str,
    output_dir: str = None,
    verbose: bool = True
) -> dict:
    """Run full comparison for a single video.

    Args:
        video_path: Path to video file
        cpp_ref_csv: Path to C++ reference CSV
        output_dir: Optional output directory for results
        verbose: Print progress

    Returns:
        dict with all metrics
    """
    video_name = Path(video_path).stem

    if verbose:
        print(f"\n{'='*60}")
        print(f"Comparing: {video_name}")
        print(f"{'='*60}")

    # Load C++ reference
    if verbose:
        print("\n[1/4] Loading C++ reference...")
    cpp_df = load_cpp_reference(cpp_ref_csv)
    cpp_frames = len(cpp_df)
    if verbose:
        print(f"  Loaded {cpp_frames} frames")

    # Process with Python
    if verbose:
        print("\n[2/4] Processing with Python pipeline...")
    py_csv = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        py_csv = str(Path(output_dir) / f"{video_name}_python.csv")

    py_df = process_video_python(video_path, py_csv, verbose)
    py_frames = len(py_df)

    if verbose:
        print(f"  Generated {py_frames} frames")

    # Align frame counts
    min_frames = min(py_frames, cpp_frames)
    if py_frames != cpp_frames:
        if verbose:
            print(f"  Warning: Frame count mismatch (Python={py_frames}, C++={cpp_frames})")
            print(f"  Using first {min_frames} frames")
        py_df = py_df.iloc[:min_frames]
        cpp_df = cpp_df.iloc[:min_frames]

    # Extract landmarks
    if verbose:
        print("\n[3/4] Computing landmark errors...")
    py_lm = extract_landmarks(py_df)
    cpp_lm = extract_landmarks(cpp_df)

    landmark_results = compute_landmark_errors(py_lm, cpp_lm)

    if verbose:
        print(f"  Overall: {landmark_results['overall_mean']:.3f} px")
        for region in REGIONS:
            print(f"  {region.capitalize()}: {landmark_results[f'{region}_mean']:.3f} px")

    # Compute AU metrics
    if verbose:
        print("\n[4/4] Computing AU metrics...")
    au_results = compute_au_metrics(py_df, cpp_df)

    if verbose and au_results['avg_correlation'] is not None:
        print(f"  Average correlation: {au_results['avg_correlation']:.3f}")
        print(f"  Average MAE: {au_results['avg_mae']:.3f}")

    # Compile results
    results = {
        'video': video_name,
        'video_path': str(video_path),
        'cpp_ref': str(cpp_ref_csv),
        'frames_python': py_frames,
        'frames_cpp': cpp_frames,
        'frames_compared': min_frames,
        'landmarks': landmark_results,
        'aus': au_results,
    }

    # Save results
    if output_dir:
        results_file = Path(output_dir) / f"{video_name}_comparison.json"
        with open(results_file, 'w') as f:
            # Remove per-frame data for JSON (too large)
            results_save = results.copy()
            results_save['landmarks'] = {k: v for k, v in landmark_results.items()
                                         if not k.endswith('per_frame_mean')}
            json.dump(results_save, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {results_file}")

    return results


def print_summary(results: dict):
    """Print formatted summary of comparison results."""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {results['video']}")
    print(f"{'='*60}")

    print(f"\nFrames: Python={results['frames_python']}, C++={results['frames_cpp']}")

    lm = results['landmarks']
    print(f"\nLandmark Error (px):")
    print(f"  Overall:  {lm['overall_mean']:.3f} +/- {lm['overall_std']:.3f}")
    for region in REGIONS:
        print(f"  {region.capitalize():8s}: {lm[f'{region}_mean']:.3f} +/- {lm[f'{region}_std']:.3f}")

    au = results['aus']
    if au['avg_correlation'] is not None:
        print(f"\nAU Metrics:")
        print(f"  Avg Correlation: {au['avg_correlation']:.3f}")
        print(f"  Avg MAE: {au['avg_mae']:.3f}")

        # Top 5 best/worst correlations
        corrs = au['correlations']
        valid = {k: v for k, v in corrs.items() if v is not None}
        if valid:
            sorted_corrs = sorted(valid.items(), key=lambda x: x[1], reverse=True)
            print(f"\n  Top 5 AUs:")
            for au_name, corr in sorted_corrs[:5]:
                print(f"    {au_name}: {corr:.3f}")
            print(f"  Bottom 5 AUs:")
            for au_name, corr in sorted_corrs[-5:]:
                print(f"    {au_name}: {corr:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Compare Python vs C++ AU pipeline")

    # Input options
    parser.add_argument('--video', help='Video file path')
    parser.add_argument('--cpp-ref', help='C++ reference CSV path')
    parser.add_argument('--video-index', type=int,
                        help='SLURM array task index for batch mode')

    # Output options
    parser.add_argument('--output', default='comparison_results',
                        help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()
    verbose = not args.quiet

    # Handle SLURM array mode
    if args.video_index is not None:
        # Video list (same as process_videos.slurm)
        # NOTE: IMG_0422 added - important test video for accuracy verification
        VIDEOS = [
            "S Data/Normal Cohort/IMG_0422.MOV",
            "S Data/Normal Cohort/IMG_0428.MOV",
            "S Data/Normal Cohort/IMG_0433.MOV",
            "S Data/Normal Cohort/IMG_0434.MOV",
            "S Data/Normal Cohort/IMG_0435.MOV",
            "S Data/Normal Cohort/IMG_0438.MOV",
            "S Data/Normal Cohort/IMG_0452.MOV",
            "S Data/Normal Cohort/IMG_0453.MOV",
            "S Data/Normal Cohort/IMG_0579.MOV",
            "S Data/Normal Cohort/IMG_0942.MOV",
            "S Data/Paralysis Cohort/IMG_0592.MOV",
            "S Data/Paralysis Cohort/IMG_0861.MOV",
            "S Data/Paralysis Cohort/IMG_1366.MOV",
        ]

        idx = args.video_index
        if idx >= len(VIDEOS):
            print(f"Error: Index {idx} out of range (max {len(VIDEOS)-1})")
            sys.exit(1)

        video_rel = VIDEOS[idx]
        video_path = project_root / video_rel
        video_name = Path(video_rel).stem
        cpp_ref = project_root / f"cpp_reference/video_{idx}/{video_name}.csv"

        args.video = str(video_path)
        args.cpp_ref = str(cpp_ref)

    # Validate inputs
    if not args.video or not args.cpp_ref:
        parser.error("Must provide --video and --cpp-ref (or --video-index for batch mode)")

    if not Path(args.video).exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    if not Path(args.cpp_ref).exists():
        print(f"Error: C++ reference not found: {args.cpp_ref}")
        sys.exit(1)

    # Run comparison
    results = compare_single_video(
        video_path=args.video,
        cpp_ref_csv=args.cpp_ref,
        output_dir=args.output,
        verbose=verbose
    )

    # Print summary
    if verbose:
        print_summary(results)


if __name__ == '__main__':
    main()
