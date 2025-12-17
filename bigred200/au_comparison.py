#!/usr/bin/env python3
"""
AU & Landmark Comparison - Python FullPythonAUPipeline vs C++ OpenFace
Compares AU intensity values AND landmark accuracy across facial zones.
Designed for HPC with staggered initialization.
"""
import argparse
import numpy as np
import pandas as pd
import cv2
import os
import sys
import time
from pathlib import Path

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

# Landmark region definitions
LANDMARK_REGIONS = {
    'jaw': list(range(0, 17)),
    'brows': list(range(17, 27)),
    'nose': list(range(27, 36)),
    'eyes': list(range(36, 48)),
    'mouth': list(range(48, 68))
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


def compute_landmark_metrics(py_landmarks, cpp_landmarks):
    """Compute per-region landmark error metrics.

    Args:
        py_landmarks: (n_frames, 68, 2) Python landmarks
        cpp_landmarks: (n_frames, 68, 2) C++ landmarks

    Returns:
        dict with per-region and overall error statistics
    """
    # Euclidean error per landmark per frame
    diff = py_landmarks - cpp_landmarks  # (n, 68, 2)
    per_lm_error = np.sqrt(diff[:, :, 0]**2 + diff[:, :, 1]**2)  # (n, 68)

    results = {
        'overall_mean': float(per_lm_error.mean()),
        'overall_std': float(per_lm_error.std()),
        'overall_max': float(per_lm_error.max()),
        'overall_median': float(np.median(per_lm_error)),
    }

    # Per-region errors
    for region_name, indices in LANDMARK_REGIONS.items():
        region_error = per_lm_error[:, indices]
        results[f'{region_name}_mean'] = float(region_error.mean())
        results[f'{region_name}_std'] = float(region_error.std())
        results[f'{region_name}_max'] = float(region_error.max())

    # Per-landmark mean error (for identifying problematic landmarks)
    results['per_landmark_mean'] = per_lm_error.mean(axis=0).tolist()

    return results


def extract_landmarks_from_cpp(cpp_df):
    """Extract 68x2 landmarks from C++ DataFrame.

    Returns: (n_frames, 68, 2) array
    """
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]

    # Check if columns exist
    missing_x = [c for c in x_cols if c not in cpp_df.columns]
    missing_y = [c for c in y_cols if c not in cpp_df.columns]

    if missing_x or missing_y:
        return None

    n_frames = len(cpp_df)
    landmarks = np.zeros((n_frames, 68, 2), dtype=np.float32)

    landmarks[:, :, 0] = cpp_df[x_cols].values
    landmarks[:, :, 1] = cpp_df[y_cols].values

    return landmarks


def get_video_rotation(video_path: str) -> int:
    """Get video rotation from metadata using ffprobe."""
    import subprocess
    import json
    try:
        cmd = f'ffprobe -v quiet -print_format json -show_streams "{video_path}"'
        output = subprocess.check_output(cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
        metadata = json.loads(output)
        for stream in metadata.get('streams', []):
            rotation = stream.get('tags', {}).get('rotate')
            if rotation is None:
                rotation = stream.get('rotation')
            if rotation:
                return int(rotation)
    except:
        pass
    return 0


def apply_frame_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Apply rotation correction to a frame."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == -90 or rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def main():
    parser = argparse.ArgumentParser(description='AU & Landmark Comparison - Python vs C++ OpenFace')
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
    print("AU & LANDMARK COMPARISON - Python vs C++ OpenFace")
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

    # Extract C++ landmarks if available
    cpp_landmarks = extract_landmarks_from_cpp(cpp_df)
    has_landmarks = cpp_landmarks is not None
    if has_landmarks:
        print(f"C++ landmarks available: {cpp_landmarks.shape}")
    else:
        print("WARNING: No landmark columns in C++ reference")
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

    # On BigRed200: base_path = ~/pyfaceau, weights at pyfaceau/weights/
    weights_path = f'{base_path}/pyfaceau/weights'

    pipeline = FullPythonAUPipeline(
        pdm_file=f'{weights_path}/In-the-wild_aligned_PDM_68.txt',
        au_models_dir=f'{weights_path}/AU_predictors',
        triangulation_file=f'{weights_path}/tris_68_full.txt',
        patch_expert_file=f'{weights_path}/patch_experts/cen_patches_0.25_of.dat',
        track_faces=True,
        verbose=False,
        debug_mode=True  # Enable debug mode to capture landmarks
    )

    print("  Step 3/4: Pipeline initialized!")
    sys.stdout.flush()

    # Process video frame-by-frame to capture landmarks
    frame_msg = f"{args.n_frames} frames" if args.n_frames else "all frames"
    print(f"\nProcessing video with Python pipeline ({frame_msg})...")
    sys.stdout.flush()

    # Get video rotation
    rotation = get_video_rotation(video_path)
    if rotation != 0:
        print(f"  Detected video rotation: {rotation}°")

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.n_frames:
        total_frames = min(total_frames, args.n_frames)

    print(f"  Video: {fps:.1f} FPS, {total_frames} frames")
    sys.stdout.flush()

    # Process frames
    results = []
    py_landmarks_list = []
    frame_idx = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or (args.n_frames and frame_idx >= args.n_frames):
                break

            # Apply rotation correction
            if rotation != 0:
                frame = apply_frame_rotation(frame, rotation)

            timestamp = frame_idx / fps

            # Process frame with debug mode to get landmarks
            frame_result = pipeline._process_frame(frame, frame_idx, timestamp, return_debug=True)
            results.append(frame_result)

            # Extract landmarks from debug info
            if frame_result.get('success') and 'debug_info' in frame_result:
                debug_info = frame_result['debug_info']
                if 'landmark_detection' in debug_info:
                    lm = debug_info['landmark_detection'].get('landmarks_68')
                    if lm is not None:
                        py_landmarks_list.append(lm)
                    else:
                        py_landmarks_list.append(np.zeros((68, 2), dtype=np.float32))
                else:
                    py_landmarks_list.append(np.zeros((68, 2), dtype=np.float32))
            else:
                py_landmarks_list.append(np.zeros((68, 2), dtype=np.float32))

            # Progress update
            if (frame_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (frame_idx + 1) / elapsed
                print(f"  Progress: {frame_idx + 1}/{total_frames} ({rate:.1f} FPS)", flush=True)

            frame_idx += 1

    finally:
        cap.release()

    elapsed = time.time() - start_time
    print(f"  Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS)")

    # Convert to DataFrame and apply post-processing
    py_df = pd.DataFrame(results)

    # Remove debug_info column if present (not needed in CSV)
    if 'debug_info' in py_df.columns:
        py_df = py_df.drop(columns=['debug_info'])

    # Apply finalize_predictions (two-pass, smoothing, etc.)
    print("\nApplying post-processing...")
    py_df = pipeline.finalize_predictions(py_df)
    print(f"  Python frames processed: {len(py_df)}")
    sys.stdout.flush()

    # Convert landmarks to array
    py_landmarks = np.array(py_landmarks_list, dtype=np.float32)

    # Compute landmark metrics if available
    landmark_metrics = None
    if has_landmarks and len(py_landmarks) > 0:
        print("\nComputing landmark metrics...")
        sys.stdout.flush()

        min_len = min(len(py_landmarks), len(cpp_landmarks))
        landmark_metrics = compute_landmark_metrics(py_landmarks[:min_len], cpp_landmarks[:min_len])

        # Print landmark results
        print("\n" + "-" * 60)
        print("LANDMARK ACCURACY (pixels)")
        print("-" * 60)
        print(f"Overall:   {landmark_metrics['overall_mean']:.2f} +/- {landmark_metrics['overall_std']:.2f} (median: {landmark_metrics['overall_median']:.2f})")

        for region in LANDMARK_REGIONS:
            print(f"{region.capitalize():8s}: {landmark_metrics[f'{region}_mean']:.2f} +/- {landmark_metrics[f'{region}_std']:.2f}")

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
    print("AU ZONE SUMMARY")
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
    if landmark_metrics:
        print(f"Landmark Error: {landmark_metrics['overall_mean']:.2f} px (target: <2.0 px)")
    if valid_corrs:
        mean_corr = np.mean(valid_corrs)
        print(f"AU Correlation: {mean_corr:.3f} (target: >=0.95)")
        # Count passing AUs
        passing = sum(1 for c in valid_corrs if c >= 0.95)
        print(f"AUs Passing (>=0.95): {passing}/{len(valid_corrs)}")
    print(f"AU MAE: {np.mean(all_maes):.3f}")

    # Save results to CSV
    os.makedirs(args.output_dir, exist_ok=True)

    # AU metrics CSV
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
    print(f"\nAU results saved to: {results_file}")

    # Landmark metrics CSV
    if landmark_metrics:
        lm_file = os.path.join(args.output_dir, f'landmark_comparison_video_{video_idx}.csv')
        lm_rows = [{
            'video': video_name,
            'video_idx': video_idx,
            'cohort': cohort,
            'overall_mean': landmark_metrics['overall_mean'],
            'overall_std': landmark_metrics['overall_std'],
            'overall_max': landmark_metrics['overall_max'],
            'jaw_mean': landmark_metrics['jaw_mean'],
            'brows_mean': landmark_metrics['brows_mean'],
            'nose_mean': landmark_metrics['nose_mean'],
            'eyes_mean': landmark_metrics['eyes_mean'],
            'mouth_mean': landmark_metrics['mouth_mean'],
        }]
        pd.DataFrame(lm_rows).to_csv(lm_file, index=False)
        print(f"Landmark results saved to: {lm_file}")

    # Raw AU values
    py_au_file = os.path.join(args.output_dir, f'py_aus_video_{video_idx}.csv')
    py_df.to_csv(py_au_file, index=False)
    print(f"Python AU values saved to: {py_au_file}")

    # Raw landmarks (add x_0..x_67, y_0..y_67 columns to a separate file)
    if len(py_landmarks) > 0:
        lm_data = {'frame': list(range(len(py_landmarks)))}
        for i in range(68):
            lm_data[f'x_{i}'] = py_landmarks[:, i, 0]
            lm_data[f'y_{i}'] = py_landmarks[:, i, 1]
        lm_df = pd.DataFrame(lm_data)
        py_lm_file = os.path.join(args.output_dir, f'py_landmarks_video_{video_idx}.csv')
        lm_df.to_csv(py_lm_file, index=False)
        print(f"Python landmarks saved to: {py_lm_file}")

    print("\n" + "=" * 70)
    print(f"COMPLETED: Video {video_idx} ({video_name})")
    print("=" * 70)


if __name__ == '__main__':
    main()
