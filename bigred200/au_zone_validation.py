#!/usr/bin/env python3
"""
AU Zone Validation - Landmark accuracy by facial zone.
Uses CLNF directly with video tracking mode for landmark comparison.
"""
import argparse
import numpy as np
import pandas as pd
import cv2
import os
import sys

# Zone definitions for landmarks
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


def process_video(video_idx, video_name, cohort, cpp_ref_dir, video_dir, n_frames=100):
    """Process a single video with CLNF and return zone metrics."""
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
    video_path = os.path.join(video_dir, cohort, video_name)
    if not os.path.exists(video_path):
        print(f"  Video not found: {video_path}")
        return None

    # Initialize CLNF with video profile (enables tracking)
    print(f"  Initializing CLNF with video tracking mode...")
    clnf = CLNF(
        convergence_profile='video',
        detector='pymtcnn',
        use_validator=False,
        use_eye_refinement=True
    )

    # Process frames
    cap = cv2.VideoCapture(video_path)

    all_lm_errors = []
    prev_landmarks = None

    print(f"  Processing {n_frames} frames...")

    for frame_idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

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

            # Get C++ reference for this frame
            if frame_idx >= len(cpp_df):
                break
            cpp_row = cpp_df.iloc[frame_idx]

            # Extract C++ landmarks
            x_cols = [f'x_{i}' for i in range(68)]
            y_cols = [f'y_{i}' for i in range(68)]
            cpp_lm = np.stack([cpp_row[x_cols].values, cpp_row[y_cols].values], axis=1).astype(np.float32)

            # Compute landmark errors
            lm_errors = compute_landmark_zone_errors(landmarks, cpp_lm)
            all_lm_errors.append(lm_errors)

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

    return {
        'video': video_name,
        'cohort': cohort,
        'n_frames': len(all_lm_errors),
        'landmarks': lm_results
    }


def print_summary(all_results):
    """Print comprehensive summary of all results."""
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL VIDEOS")
    print("=" * 70)

    # Landmark zone summary
    print("\nLANDMARK ACCURACY BY ZONE (mean error in pixels):")
    print("-" * 50)

    zones = ['upper_face', 'brows', 'eyes', 'mid_face', 'nose',
             'lower_face', 'jaw', 'mouth', 'overall']

    for zone in zones:
        values = [r['landmarks'].get(zone) for r in all_results if zone in r.get('landmarks', {})]
        values = [v for v in values if v is not None]
        if values:
            print(f"  {zone:15s}: {np.mean(values):.3f} px (std: {np.std(values):.3f})")


def main():
    parser = argparse.ArgumentParser(description='Landmark Zone Validation - CLNF vs C++ OpenFace')
    parser.add_argument('--video-index', type=int, default=-1, help='Video index (-1 for all)')
    parser.add_argument('--n-frames', type=int, default=100, help='Frames per video')
    parser.add_argument('--cpp-ref', default='cpp_reference', help='C++ reference directory')
    parser.add_argument('--video-dir', default='S Data', help='Video directory')
    args = parser.parse_args()

    print("=" * 70)
    print("LANDMARK ZONE VALIDATION - CLNF vs C++ OpenFace")
    print("=" * 70)

    # Process videos
    if args.video_index >= 0:
        videos_to_process = [VIDEOS[args.video_index]]
    else:
        videos_to_process = VIDEOS

    all_results = []

    for video_dir_name, video_name, cohort in videos_to_process:
        video_idx = int(video_dir_name.split('_')[1])
        print(f"\nProcessing {video_name} ({cohort}, index {video_idx})...")

        result = process_video(
            video_idx, video_name, cohort,
            args.cpp_ref, args.video_dir,
            args.n_frames
        )

        if result:
            all_results.append(result)
            print(f"  Frames processed: {result['n_frames']}")
            print(f"  Landmark Errors:")
            for zone in ['upper_face', 'lower_face', 'overall']:
                if zone in result['landmarks']:
                    print(f"    {zone}: {result['landmarks'][zone]:.3f} px")

    # Print summary
    if all_results:
        print_summary(all_results)


if __name__ == '__main__':
    main()
