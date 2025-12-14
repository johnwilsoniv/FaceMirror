#!/usr/bin/env python3
"""
Compare C++ vs Python AU correlations and landmark accuracy for 50 frames.
"""

import numpy as np
import pandas as pd
import cv2
import sys
from scipy import stats

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pymtcnn')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau')

# Videos to test
VIDEOS = [
    ('IMG_0422', '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Normal Cohort/IMG_0422.MOV', '/tmp/cpp_0422/IMG_0422.csv'),
    ('IMG_0434', '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Normal Cohort/IMG_0434.MOV', '/tmp/cpp_0434/IMG_0434.csv'),
]

N_FRAMES = 50

# AU columns
AU_INTENSITY_COLS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
                     'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
                     'AU25_r', 'AU26_r', 'AU45_r']

# AU zones for analysis
AU_ZONES = {
    'Brow (Upper)': ['AU01_r', 'AU02_r', 'AU04_r'],  # Inner/outer brow raiser, brow lowerer
    'Eye': ['AU05_r', 'AU06_r', 'AU07_r', 'AU45_r'],  # Lid raiser, cheek raiser, lid tightener, blink
    'Nose': ['AU09_r'],  # Nose wrinkler
    'Mouth (Lower)': ['AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']
}

def load_cpp_data(cpp_csv, n_frames):
    """Load C++ landmarks and AUs."""
    df = pd.read_csv(cpp_csv)
    df = df.head(n_frames)

    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]

    landmarks = []
    for _, row in df.iterrows():
        lm = np.stack([row[x_cols].values, row[y_cols].values], axis=1).astype(np.float32)
        landmarks.append(lm)

    aus = df[AU_INTENSITY_COLS].values
    return landmarks, aus, df

def run_python_pipeline(video_path, n_frames):
    """Run Python CLNF + AU pipeline."""
    from pathlib import Path
    from pyclnf import CLNF
    from pyfaceau import FullPythonAUPipeline

    # Initialize CLNF for landmarks
    clnf = CLNF(convergence_profile='video', detector='pymtcnn',
                use_validator=False, use_eye_refinement=False)

    # Initialize pyfaceau pipeline for AUs
    weights_dir = Path('/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/weights')

    au_pipeline = FullPythonAUPipeline(
        pdm_file=str(weights_dir / 'In-the-wild_aligned_PDM_68.txt'),
        au_models_dir=str(weights_dir / 'AU_predictors'),
        triangulation_file=str(weights_dir / 'tris_68_full.txt'),
        patch_expert_file=str(weights_dir / 'svr_patches_0.25_general.txt'),
        mtcnn_backend='auto',
        use_batched_predictor=True,
        verbose=False
    )

    # Process video for AUs
    au_df = au_pipeline.process_video(video_path, max_frames=n_frames)

    # Get landmarks using pyclnf on the same frames
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        landmarks, info = clnf.detect_and_fit(frame)
        landmarks_list.append(landmarks)

    cap.release()

    # Extract AUs
    aus = au_df[AU_INTENSITY_COLS].values

    return landmarks_list, aus

def compute_landmark_errors(py_landmarks, cpp_landmarks):
    """Compute per-frame landmark errors."""
    jaw_errors = []
    overall_errors = []

    for py_lm, cpp_lm in zip(py_landmarks, cpp_landmarks):
        jaw_err = np.linalg.norm(py_lm[:17] - cpp_lm[:17], axis=1).mean()
        overall_err = np.linalg.norm(py_lm - cpp_lm, axis=1).mean()
        jaw_errors.append(jaw_err)
        overall_errors.append(overall_err)

    return np.array(jaw_errors), np.array(overall_errors)

def compute_au_correlations(py_aus, cpp_aus):
    """Compute per-AU Pearson correlations."""
    correlations = {}
    for i, au_name in enumerate(AU_INTENSITY_COLS):
        py_col = py_aus[:, i]
        cpp_col = cpp_aus[:, i]

        # Skip if no variance
        if np.std(py_col) < 1e-6 or np.std(cpp_col) < 1e-6:
            correlations[au_name] = np.nan
        else:
            r, _ = stats.pearsonr(py_col, cpp_col)
            correlations[au_name] = r

    return correlations

def compute_zone_correlations(au_corrs):
    """Compute mean correlation for each AU zone."""
    zone_corrs = {}
    for zone_name, zone_aus in AU_ZONES.items():
        zone_vals = [au_corrs.get(au, np.nan) for au in zone_aus]
        valid_vals = [v for v in zone_vals if not np.isnan(v)]
        if valid_vals:
            zone_corrs[zone_name] = np.mean(valid_vals)
        else:
            zone_corrs[zone_name] = np.nan
    return zone_corrs

def main():
    print("=" * 80)
    print("C++ vs PYTHON COMPARISON - 50 FRAMES")
    print("=" * 80)

    all_results = []

    for video_name, video_path, cpp_csv in VIDEOS:
        print(f"\n{'=' * 80}")
        print(f"VIDEO: {video_name}")
        print("=" * 80)

        # Load C++ data
        print("\nLoading C++ data...")
        cpp_landmarks, cpp_aus, cpp_df = load_cpp_data(cpp_csv, N_FRAMES)

        # Run Python pipeline
        print("Running Python pipeline...")
        py_landmarks, py_aus = run_python_pipeline(video_path, N_FRAMES)

        # Compute landmark errors
        jaw_errors, overall_errors = compute_landmark_errors(py_landmarks, cpp_landmarks)

        print(f"\n--- LANDMARK ACCURACY ({len(py_landmarks)} frames) ---")
        print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Max':>10}")
        print("-" * 55)
        print(f"{'Jaw Error (px)':<20} {jaw_errors.mean():>10.2f} {jaw_errors.std():>10.2f} {jaw_errors.max():>10.2f}")
        print(f"{'Overall Error (px)':<20} {overall_errors.mean():>10.2f} {overall_errors.std():>10.2f} {overall_errors.max():>10.2f}")

        # Compute AU correlations
        au_corrs = compute_au_correlations(py_aus, cpp_aus)

        print(f"\n--- AU CORRELATIONS ---")
        print(f"{'AU':<10} {'Correlation':>12} {'Py Mean':>10} {'C++ Mean':>10}")
        print("-" * 45)

        valid_corrs = []
        for i, au_name in enumerate(AU_INTENSITY_COLS):
            r = au_corrs[au_name]
            py_mean = py_aus[:, i].mean()
            cpp_mean = cpp_aus[:, i].mean()

            if not np.isnan(r):
                valid_corrs.append(r)
                marker = " ***" if r < 0.7 else ""
                print(f"{au_name:<10} {r:>12.3f} {py_mean:>10.2f} {cpp_mean:>10.2f}{marker}")
            else:
                print(f"{au_name:<10} {'N/A':>12} {py_mean:>10.2f} {cpp_mean:>10.2f}")

        mean_corr = np.nanmean(valid_corrs) if valid_corrs else np.nan
        print("-" * 45)
        print(f"{'Mean Corr':<10} {mean_corr:>12.3f}")

        # Compute zone correlations
        zone_corrs = compute_zone_correlations(au_corrs)

        print(f"\n--- AU CORRELATIONS BY ZONE ---")
        print(f"{'Zone':<15} {'Correlation':>12} {'AUs':<30}")
        print("-" * 60)
        for zone_name, zone_aus in AU_ZONES.items():
            zone_r = zone_corrs.get(zone_name, np.nan)
            au_list = ', '.join([au.replace('_r', '') for au in zone_aus])
            if not np.isnan(zone_r):
                marker = " ***" if zone_r < 0.7 else ""
                print(f"{zone_name:<15} {zone_r:>12.3f} {au_list:<30}{marker}")
            else:
                print(f"{zone_name:<15} {'N/A':>12} {au_list:<30}")

        # Store results
        all_results.append({
            'video': video_name,
            'n_frames': len(py_landmarks),
            'jaw_error_mean': jaw_errors.mean(),
            'jaw_error_max': jaw_errors.max(),
            'overall_error_mean': overall_errors.mean(),
            'mean_au_corr': mean_corr,
            'au_correlations': au_corrs,
            'zone_correlations': zone_corrs
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Video':<12} {'Frames':>8} {'Jaw Err':>10} {'Overall':>10} {'AU Corr':>10}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['video']:<12} {r['n_frames']:>8} {r['jaw_error_mean']:>10.2f} {r['overall_error_mean']:>10.2f} {r['mean_au_corr']:>10.3f}")

    # Overall average
    avg_jaw = np.mean([r['jaw_error_mean'] for r in all_results])
    avg_overall = np.mean([r['overall_error_mean'] for r in all_results])
    avg_au_corr = np.mean([r['mean_au_corr'] for r in all_results])
    print("-" * 55)
    print(f"{'AVERAGE':<12} {'':<8} {avg_jaw:>10.2f} {avg_overall:>10.2f} {avg_au_corr:>10.3f}")

    # Zone summary
    print("\n" + "=" * 80)
    print("AU CORRELATION BY ZONE (AVERAGED ACROSS VIDEOS)")
    print("=" * 80)
    print(f"\n{'Zone':<15} ", end="")
    for r in all_results:
        print(f"{r['video']:>12}", end="")
    print(f"{'Average':>12}")
    print("-" * (15 + 12 * (len(all_results) + 1)))

    for zone_name in AU_ZONES.keys():
        print(f"{zone_name:<15} ", end="")
        zone_vals = []
        for r in all_results:
            val = r['zone_correlations'].get(zone_name, np.nan)
            zone_vals.append(val)
            if np.isnan(val):
                print(f"{'N/A':>12}", end="")
            else:
                print(f"{val:>12.3f}", end="")
        avg_zone = np.nanmean(zone_vals) if zone_vals else np.nan
        if np.isnan(avg_zone):
            print(f"{'N/A':>12}")
        else:
            marker = " ***" if avg_zone < 0.7 else ""
            print(f"{avg_zone:>12.3f}{marker}")

if __name__ == "__main__":
    main()
