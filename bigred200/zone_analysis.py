#!/usr/bin/env python3
"""
Zone-based analysis of landmark and AU accuracy.
Compares Python pipeline vs C++ OpenFace by facial zone.
"""
import argparse
import numpy as np
import pandas as pd
import cv2
import os
import sys

# Zone definitions
LANDMARK_ZONES = {
    'upper_face': {
        'brows': list(range(17, 27)),      # 17-26
        'eyes': list(range(36, 48)),        # 36-47
    },
    'mid_face': {
        'nose': list(range(27, 36)),        # 27-35
    },
    'lower_face': {
        'jaw': list(range(0, 17)),          # 0-16
        'mouth': list(range(48, 68)),       # 48-67
    }
}

# AU zone definitions (intensity AUs only)
AU_ZONES = {
    'upper_face': ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU45_r'],
    'lower_face': ['AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
                   'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r'],
}

# Video list
VIDEOS = [
    ('video_0', 'IMG_0422.MOV'),
    ('video_1', 'IMG_0428.MOV'),
    ('video_2', 'IMG_0433.MOV'),
    ('video_3', 'IMG_0434.MOV'),
    ('video_4', 'IMG_0435.MOV'),
    ('video_5', 'IMG_0438.MOV'),
    ('video_6', 'IMG_0452.MOV'),
    ('video_7', 'IMG_0453.MOV'),
    ('video_8', 'IMG_0579.MOV'),
    ('video_9', 'IMG_0942.MOV'),
    ('video_10', 'IMG_0592.MOV'),
    ('video_11', 'IMG_0861.MOV'),
    ('video_12', 'IMG_1366.MOV'),
]


def compute_landmark_zone_errors(py_landmarks, cpp_landmarks):
    """Compute per-zone landmark errors."""
    diff = py_landmarks - cpp_landmarks
    per_lm_error = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)

    results = {'overall': per_lm_error.mean()}

    for zone_name, subzones in LANDMARK_ZONES.items():
        zone_indices = []
        for subzone_name, indices in subzones.items():
            zone_indices.extend(indices)
            results[subzone_name] = per_lm_error[indices].mean()
        results[zone_name] = per_lm_error[zone_indices].mean()

    return results


def compute_au_zone_correlations(py_aus, cpp_aus):
    """Compute per-zone AU correlations."""
    results = {}
    all_corrs = []

    for zone_name, au_list in AU_ZONES.items():
        zone_corrs = []
        for au in au_list:
            if au in py_aus.columns and au in cpp_aus.columns:
                py_col = py_aus[au].values
                cpp_col = cpp_aus[au].values
                # Only compute if both have variance
                if py_col.std() > 0 and cpp_col.std() > 0:
                    corr = np.corrcoef(py_col, cpp_col)[0, 1]
                    if not np.isnan(corr):
                        zone_corrs.append(corr)
                        all_corrs.append(corr)
                        results[au] = corr

        if zone_corrs:
            results[f'{zone_name}_mean'] = np.mean(zone_corrs)

    if all_corrs:
        results['overall_mean'] = np.mean(all_corrs)

    return results


def process_video(video_idx, video_name, cpp_ref_dir, video_dir, n_frames=100):
    """Process a single video and return zone metrics."""
    from pyclnf import CLNF

    # Load C++ reference
    cpp_csv = os.path.join(cpp_ref_dir, f'video_{video_idx}', video_name.replace('.MOV', '.csv'))
    if not os.path.exists(cpp_csv):
        print(f"  C++ reference not found: {cpp_csv}")
        return None

    cpp_df = pd.read_csv(cpp_csv)
    cpp_df.columns = cpp_df.columns.str.strip()
    cpp_df = cpp_df.head(n_frames)

    # Find video file
    video_path = None
    for subdir in ['Normal Cohort', 'Paralysis Cohort']:
        test_path = os.path.join(video_dir, subdir, video_name)
        if os.path.exists(test_path):
            video_path = test_path
            break

    if not video_path:
        print(f"  Video not found: {video_name}")
        return None

    # Initialize CLNF with video profile (enables tracking)
    clnf = CLNF(
        convergence_profile='video',
        detector='pymtcnn',
        use_validator=False,
        use_eye_refinement=True
    )

    # Process frames
    cap = cv2.VideoCapture(video_path)

    all_lm_errors = []
    py_au_rows = []
    cpp_au_rows = []
    prev_landmarks = None

    for frame_idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Python CLNF - matches validate_100_frames.py flow
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if frame_idx == 0:
                # First frame: detect + fit
                landmarks, info = clnf.detect_and_fit(frame)
            else:
                # Subsequent frames: track from previous
                x_min, y_min = prev_landmarks.min(axis=0)
                x_max, y_max = prev_landmarks.max(axis=0)
                w, h = x_max - x_min, y_max - y_min
                margin = 0.1
                bbox = (x_min - w*margin, y_min - h*margin, w*(1+2*margin), h*(1+2*margin))
                landmarks, info = clnf.fit(gray, bbox)

            if landmarks is None:
                continue

            prev_landmarks = landmarks.copy()

            # Get C++ landmarks for this frame
            if frame_idx >= len(cpp_df):
                break
            cpp_row = cpp_df.iloc[frame_idx]

            x_cols = [f'x_{i}' for i in range(68)]
            y_cols = [f'y_{i}' for i in range(68)]
            cpp_lm = np.stack([cpp_row[x_cols].values, cpp_row[y_cols].values], axis=1).astype(np.float32)

            # Compute landmark errors
            lm_errors = compute_landmark_zone_errors(landmarks, cpp_lm)
            all_lm_errors.append(lm_errors)

            # Extract C++ AUs for correlation analysis
            cpp_aus_row = {}
            for au in AU_ZONES['upper_face'] + AU_ZONES['lower_face']:
                if au in cpp_df.columns:
                    cpp_aus_row[au] = cpp_row[au]
            if cpp_aus_row:
                cpp_au_rows.append(cpp_aus_row)

            # Print progress every 20 frames
            if frame_idx % 20 == 0:
                print(f"    Frame {frame_idx}: overall={lm_errors['overall']:.2f}px, "
                      f"jaw={lm_errors['jaw']:.2f}px, eyes={lm_errors['eyes']:.2f}px")

        except Exception as e:
            print(f"  Frame {frame_idx} error: {e}")
            continue

    cap.release()

    if not all_lm_errors:
        return None

    # Aggregate landmark errors
    lm_results = {}
    for key in all_lm_errors[0].keys():
        lm_results[key] = np.mean([e[key] for e in all_lm_errors])

    # Note: AU correlations require the full pipeline, not just CLNF
    # For now, just return landmark zone analysis
    au_results = {}

    return {
        'video': video_name,
        'n_frames': len(all_lm_errors),
        'landmarks': lm_results,
        'aus': au_results
    }


def main():
    parser = argparse.ArgumentParser(description='Zone-based accuracy analysis')
    parser.add_argument('--video-index', type=int, default=-1, help='Video index (-1 for all)')
    parser.add_argument('--n-frames', type=int, default=100, help='Frames per video')
    parser.add_argument('--cpp-ref', default='cpp_reference', help='C++ reference directory')
    parser.add_argument('--video-dir', default='S Data', help='Video directory')
    args = parser.parse_args()

    print("=" * 70)
    print("ZONE-BASED ACCURACY ANALYSIS")
    print("=" * 70)

    # Process videos
    if args.video_index >= 0:
        videos_to_process = [VIDEOS[args.video_index]]
    else:
        videos_to_process = VIDEOS

    all_results = []

    for video_dir_name, video_name in videos_to_process:
        video_idx = int(video_dir_name.split('_')[1])
        print(f"\nProcessing {video_name} (index {video_idx})...")

        result = process_video(
            video_idx, video_name,
            args.cpp_ref, args.video_dir,
            args.n_frames
        )

        if result:
            all_results.append(result)
            print(f"  Frames: {result['n_frames']}")
            print(f"  Landmark Errors:")
            for zone in ['upper_face', 'mid_face', 'lower_face', 'overall']:
                if zone in result['landmarks']:
                    print(f"    {zone}: {result['landmarks'][zone]:.3f} px")

    # Summary
    if all_results:
        print("\n" + "=" * 70)
        print("SUMMARY ACROSS ALL VIDEOS")
        print("=" * 70)

        # Landmark zone summary
        print("\nLANDMARK ACCURACY BY ZONE (mean error in pixels):")
        print("-" * 50)

        zone_means = {}
        for zone in ['upper_face', 'mid_face', 'lower_face', 'brows', 'eyes', 'nose', 'jaw', 'mouth', 'overall']:
            values = [r['landmarks'].get(zone, np.nan) for r in all_results if zone in r['landmarks']]
            if values:
                zone_means[zone] = np.mean(values)
                print(f"  {zone:15s}: {zone_means[zone]:.3f} px")

        # AU zone summary
        print("\nAU CORRELATION BY ZONE:")
        print("-" * 50)

        for zone in ['upper_face_mean', 'lower_face_mean', 'overall_mean']:
            values = [r['aus'].get(zone, np.nan) for r in all_results if zone in r.get('aus', {})]
            values = [v for v in values if not np.isnan(v)]
            if values:
                print(f"  {zone:20s}: {np.mean(values):.3f}")

        # Per-AU summary
        print("\nPER-AU CORRELATIONS:")
        print("-" * 50)

        for zone_name, au_list in AU_ZONES.items():
            print(f"\n  {zone_name.upper()}:")
            for au in au_list:
                values = [r['aus'].get(au, np.nan) for r in all_results if au in r.get('aus', {})]
                values = [v for v in values if not np.isnan(v)]
                if values:
                    print(f"    {au:10s}: {np.mean(values):.3f}")


if __name__ == '__main__':
    main()
