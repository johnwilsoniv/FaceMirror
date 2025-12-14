#!/usr/bin/env python3
"""
AU Comparison - Python FullPythonAUPipeline vs C++ OpenFace
Compares AU intensity values across facial zones.
Designed for HPC with staggered initialization.
"""
import argparse
import numpy as np
import pandas as pd
import cv2
import os
import sys
import time

# AU zone definitions
AU_ZONES = {
    'upper_face': {
        'brows': ['AU01_r', 'AU02_r', 'AU04_r'],
        'eyes': ['AU05_r', 'AU06_r', 'AU07_r', 'AU45_r'],
    },
    'lower_face': {
        'nose': ['AU09_r'],
        'mouth': ['AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
                  'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r'],
    }
}

ALL_AUS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
           'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
           'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

# Video list
VIDEOS = [
    ('video_0', 'IMG_0422.MOV', 'Normal Cohort'),
    ('video_1', 'IMG_0428.MOV', 'Normal Cohort'),
    ('video_2', 'IMG_0433.MOV', 'Normal Cohort'),
    ('video_3', 'IMG_0434.MOV', 'Normal Cohort'),
    ('video_4', 'IMG_0435.MOV', 'Normal Cohort'),
    ('video_5', 'IMG_0438.MOV', 'Normal Cohort'),
    ('video_6', 'IMG_0452.MOV', 'Normal Cohort'),
    ('video_7', 'IMG_0453.MOV', 'Normal Cohort'),
    ('video_8', 'IMG_0579.MOV', 'Normal Cohort'),
    ('video_9', 'IMG_0942.MOV', 'Normal Cohort'),
    ('video_10', 'IMG_0592.MOV', 'Paralysis Cohort'),
    ('video_11', 'IMG_0861.MOV', 'Paralysis Cohort'),
    ('video_12', 'IMG_1366.MOV', 'Paralysis Cohort'),
]


def compute_au_metrics(py_df, cpp_df, au_cols):
    """Compute correlation and error metrics for AU columns."""
    results = {}

    for au in au_cols:
        if au not in py_df.columns or au not in cpp_df.columns:
            continue

        py_vals = py_df[au].values
        cpp_vals = cpp_df[au].values

        # Ensure same length
        min_len = min(len(py_vals), len(cpp_vals))
        py_vals = py_vals[:min_len]
        cpp_vals = cpp_vals[:min_len]

        # Correlation (only if both have variance)
        if py_vals.std() > 0.001 and cpp_vals.std() > 0.001:
            corr = np.corrcoef(py_vals, cpp_vals)[0, 1]
        else:
            corr = np.nan

        # Mean absolute error
        mae = np.mean(np.abs(py_vals - cpp_vals))

        # RMSE
        rmse = np.sqrt(np.mean((py_vals - cpp_vals)**2))

        results[au] = {
            'correlation': corr,
            'mae': mae,
            'rmse': rmse,
            'py_mean': np.mean(py_vals),
            'cpp_mean': np.mean(cpp_vals),
            'py_std': np.std(py_vals),
            'cpp_std': np.std(cpp_vals),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='AU Comparison - Python vs C++ OpenFace')
    parser.add_argument('--video-index', type=int, required=True, help='Video index (0-12)')
    parser.add_argument('--n-frames', type=int, default=None, help='Frames per video (None = all frames)')
    parser.add_argument('--cpp-ref', default='cpp_reference', help='C++ reference directory')
    parser.add_argument('--video-dir', default='S Data', help='Video directory')
    parser.add_argument('--stagger-delay', type=int, default=0, help='Stagger delay in seconds')
    parser.add_argument('--output-dir', default='au_comparison_results', help='Output directory')
    args = parser.parse_args()

    # Stagger initialization to avoid memory spikes
    if args.stagger_delay > 0:
        delay = args.video_index * args.stagger_delay
        print(f"Staggering initialization by {delay} seconds...")
        sys.stdout.flush()
        time.sleep(delay)

    print("=" * 70)
    print("AU COMPARISON - Python FullPythonAUPipeline vs C++ OpenFace")
    print("=" * 70)
    sys.stdout.flush()

    # Get video info
    video_dir_name, video_name, cohort = VIDEOS[args.video_index]
    video_idx = args.video_index

    print(f"\nVideo: {video_name} ({cohort}, index {video_idx})")
    sys.stdout.flush()

    # Load C++ reference
    cpp_csv = os.path.join(args.cpp_ref, f'video_{video_idx}', video_name.replace('.MOV', '.csv'))
    if not os.path.exists(cpp_csv):
        print(f"ERROR: C++ reference not found: {cpp_csv}")
        sys.exit(1)

    cpp_df = pd.read_csv(cpp_csv)
    cpp_df.columns = cpp_df.columns.str.strip()
    if args.n_frames is not None:
        cpp_df = cpp_df.head(args.n_frames)
    print(f"C++ reference loaded: {len(cpp_df)} frames")
    sys.stdout.flush()

    # Find video file
    video_path = os.path.join(args.video_dir, cohort, video_name)
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # Initialize Python AU pipeline with explicit paths for Big Red
    print("\nInitializing Python AU pipeline...")
    print("  Step 1/4: Importing pyfaceau...")
    sys.stdout.flush()

    from pyfaceau import FullPythonAUPipeline

    # Determine base path
    base_path = os.environ.get('PYFACEAU_BASE', os.path.expanduser('~/pyfaceau'))

    print("  Step 2/4: Loading CLNF patch experts (~5-10 seconds)...")
    sys.stdout.flush()

    pipeline = FullPythonAUPipeline(
        pdm_file=f'{base_path}/pyfaceau/pyfaceau/weights/In-the-wild_aligned_PDM_68.txt',
        au_models_dir=f'{base_path}/pyfaceau/pyfaceau/weights/AU_predictors',
        triangulation_file=f'{base_path}/pyfaceau/pyfaceau/weights/tris_68_full.txt',
        patch_expert_file=f'{base_path}/pyfaceau/pyfaceau/weights/patch_experts/cen_patches_0.25_of.dat',
        track_faces=True,
        verbose=False
    )

    print("  Step 3/4: Pipeline initialized!")
    sys.stdout.flush()

    # Process video
    frame_msg = f"{args.n_frames} frames" if args.n_frames else "all frames"
    print(f"\nProcessing video with Python pipeline ({frame_msg})...")
    sys.stdout.flush()

    try:
        py_df = pipeline.process_video(video_path=video_path, max_frames=args.n_frames)
        print(f"  Python frames processed: {len(py_df)}")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR processing video: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Compute AU metrics
    print("\nComputing AU metrics...")
    sys.stdout.flush()

    au_metrics = compute_au_metrics(py_df, cpp_df, ALL_AUS)

    # Print per-AU results
    print("\n" + "-" * 60)
    print("PER-AU RESULTS")
    print("-" * 60)
    print(f"{'AU':<10} {'Corr':>10} {'MAE':>10} {'Py Mean':>10} {'C++ Mean':>10}")
    print("-" * 60)

    for au in ALL_AUS:
        if au in au_metrics:
            m = au_metrics[au]
            corr_str = f"{m['correlation']:.3f}" if not np.isnan(m['correlation']) else "N/A"
            print(f"{au:<10} {corr_str:>10} {m['mae']:>10.3f} {m['py_mean']:>10.3f} {m['cpp_mean']:>10.3f}")

    # Zone summary
    print("\n" + "-" * 60)
    print("ZONE SUMMARY")
    print("-" * 60)

    for zone_name, subzones in AU_ZONES.items():
        zone_corrs = []
        zone_maes = []
        for subzone_name, au_list in subzones.items():
            for au in au_list:
                if au in au_metrics:
                    if not np.isnan(au_metrics[au]['correlation']):
                        zone_corrs.append(au_metrics[au]['correlation'])
                    zone_maes.append(au_metrics[au]['mae'])

        if zone_corrs:
            print(f"{zone_name}: corr={np.mean(zone_corrs):.3f} (n={len(zone_corrs)}), MAE={np.mean(zone_maes):.3f}")

    # Overall summary
    valid_corrs = [m['correlation'] for m in au_metrics.values() if not np.isnan(m['correlation'])]
    all_maes = [m['mae'] for m in au_metrics.values()]

    print("\n" + "-" * 60)
    print("OVERALL SUMMARY")
    print("-" * 60)
    if valid_corrs:
        print(f"Mean Correlation: {np.mean(valid_corrs):.3f} (std={np.std(valid_corrs):.3f}, n={len(valid_corrs)})")
    print(f"Mean MAE: {np.mean(all_maes):.3f}")

    # Save results to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, f'au_comparison_video_{video_idx}.csv')

    results_rows = []
    for au, m in au_metrics.items():
        results_rows.append({
            'video': video_name,
            'video_idx': video_idx,
            'cohort': cohort,
            'au': au,
            **m
        })

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    # Also save the raw AU values for detailed analysis
    py_au_file = os.path.join(args.output_dir, f'py_aus_video_{video_idx}.csv')
    py_df.to_csv(py_au_file, index=False)
    print(f"Python AU values saved to: {py_au_file}")

    print("\n" + "=" * 70)
    print(f"COMPLETED: Video {video_idx} ({video_name})")
    print("=" * 70)


if __name__ == '__main__':
    main()
