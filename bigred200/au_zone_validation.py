#!/usr/bin/env python3
"""
AU Zone Validation - Full pipeline comparison with C++ OpenFace.
Computes both landmark errors and AU correlations by facial zone.
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

# AU zone definitions (intensity AUs only)
AU_ZONES = {
    'upper_face': ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU45_r'],
    'lower_face': ['AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
                   'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r'],
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


def compute_au_zone_stats(py_aus_df, cpp_aus_df):
    """Compute per-zone AU correlations, MAE, and means."""
    results = {
        'correlations': {},
        'mae': {},
        'py_means': {},
        'cpp_means': {},
    }

    all_corrs = []
    all_maes = []

    for zone_name, au_list in AU_ZONES.items():
        zone_corrs = []
        zone_maes = []

        for au in au_list:
            if au in py_aus_df.columns and au in cpp_aus_df.columns:
                py_col = py_aus_df[au].values
                cpp_col = cpp_aus_df[au].values

                # Store means
                results['py_means'][au] = py_col.mean()
                results['cpp_means'][au] = cpp_col.mean()

                # MAE
                mae = np.abs(py_col - cpp_col).mean()
                results['mae'][au] = mae
                zone_maes.append(mae)
                all_maes.append(mae)

                # Correlation (only if both have variance)
                if py_col.std() > 0.01 and cpp_col.std() > 0.01:
                    corr = np.corrcoef(py_col, cpp_col)[0, 1]
                    if not np.isnan(corr):
                        results['correlations'][au] = corr
                        zone_corrs.append(corr)
                        all_corrs.append(corr)

        if zone_corrs:
            results['correlations'][f'{zone_name}_mean'] = np.mean(zone_corrs)
        if zone_maes:
            results['mae'][f'{zone_name}_mean'] = np.mean(zone_maes)

    if all_corrs:
        results['correlations']['overall_mean'] = np.mean(all_corrs)
    if all_maes:
        results['mae']['overall_mean'] = np.mean(all_maes)

    return results


def process_video(video_idx, video_name, cohort, cpp_ref_dir, video_dir, n_frames=100):
    """Process a single video with full pipeline and return zone metrics."""
    from pyfaceau.pipeline import FullPythonAUPipeline

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

    # Initialize full pipeline with video profile
    print(f"  Initializing pipeline...")
    pipeline = FullPythonAUPipeline(
        clnf_convergence_profile='video',
        detector='pymtcnn',
        use_validator=False,
        use_eye_refinement=True,
        verbose=False
    )

    # Process frames
    cap = cv2.VideoCapture(video_path)

    all_lm_errors = []
    py_au_rows = []
    cpp_au_rows = []

    print(f"  Processing {n_frames} frames...")

    for frame_idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Run full pipeline
            result = pipeline.process_frame(frame)

            if result is None or result.get('landmarks') is None:
                continue

            landmarks = result['landmarks']
            aus = result.get('aus', {})

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

            # Collect AU values
            py_au_row = {}
            cpp_au_row = {}
            for zone_aus in AU_ZONES.values():
                for au in zone_aus:
                    # Python AU (convert AU01_r -> AU01)
                    au_key = au.replace('_r', '')
                    if au_key in aus:
                        py_au_row[au] = aus[au_key]
                    # C++ AU
                    if au in cpp_df.columns:
                        cpp_au_row[au] = cpp_row[au]

            if py_au_row and cpp_au_row:
                py_au_rows.append(py_au_row)
                cpp_au_rows.append(cpp_au_row)

            # Print progress every 20 frames
            if frame_idx % 20 == 0:
                print(f"    Frame {frame_idx}: LM={lm_errors['overall']:.2f}px, "
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

    # Compute AU statistics
    au_results = {}
    if py_au_rows and cpp_au_rows:
        py_aus_df = pd.DataFrame(py_au_rows)
        cpp_aus_df = pd.DataFrame(cpp_au_rows)
        au_results = compute_au_zone_stats(py_aus_df, cpp_aus_df)

    return {
        'video': video_name,
        'cohort': cohort,
        'n_frames': len(all_lm_errors),
        'landmarks': lm_results,
        'aus': au_results
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

    # AU correlation summary by zone
    print("\nAU CORRELATION BY ZONE:")
    print("-" * 50)

    for zone in ['upper_face_mean', 'lower_face_mean', 'overall_mean']:
        values = [r['aus'].get('correlations', {}).get(zone)
                  for r in all_results if r.get('aus')]
        values = [v for v in values if v is not None]
        if values:
            print(f"  {zone:20s}: {np.mean(values):.3f} (std: {np.std(values):.3f})")

    # AU MAE summary by zone
    print("\nAU MAE BY ZONE (intensity units):")
    print("-" * 50)

    for zone in ['upper_face_mean', 'lower_face_mean', 'overall_mean']:
        values = [r['aus'].get('mae', {}).get(zone)
                  for r in all_results if r.get('aus')]
        values = [v for v in values if v is not None]
        if values:
            print(f"  {zone:20s}: {np.mean(values):.3f} (std: {np.std(values):.3f})")

    # Per-AU breakdown
    print("\nPER-AU CORRELATIONS:")
    print("-" * 50)

    for zone_name, au_list in AU_ZONES.items():
        print(f"\n  {zone_name.upper()}:")
        for au in au_list:
            corrs = [r['aus'].get('correlations', {}).get(au)
                     for r in all_results if r.get('aus')]
            corrs = [v for v in corrs if v is not None]

            maes = [r['aus'].get('mae', {}).get(au)
                    for r in all_results if r.get('aus')]
            maes = [v for v in maes if v is not None]

            if corrs and maes:
                print(f"    {au:10s}: corr={np.mean(corrs):.3f}, MAE={np.mean(maes):.3f}")

    # Per-AU means comparison
    print("\nPER-AU MEAN VALUES (Python vs C++):")
    print("-" * 50)

    for zone_name, au_list in AU_ZONES.items():
        print(f"\n  {zone_name.upper()}:")
        for au in au_list:
            py_means = [r['aus'].get('py_means', {}).get(au)
                        for r in all_results if r.get('aus')]
            py_means = [v for v in py_means if v is not None]

            cpp_means = [r['aus'].get('cpp_means', {}).get(au)
                         for r in all_results if r.get('aus')]
            cpp_means = [v for v in cpp_means if v is not None]

            if py_means and cpp_means:
                print(f"    {au:10s}: Python={np.mean(py_means):.3f}, C++={np.mean(cpp_means):.3f}")


def main():
    parser = argparse.ArgumentParser(description='AU Zone Validation - Full Pipeline')
    parser.add_argument('--video-index', type=int, default=-1, help='Video index (-1 for all)')
    parser.add_argument('--n-frames', type=int, default=100, help='Frames per video')
    parser.add_argument('--cpp-ref', default='cpp_reference', help='C++ reference directory')
    parser.add_argument('--video-dir', default='S Data', help='Video directory')
    args = parser.parse_args()

    print("=" * 70)
    print("AU ZONE VALIDATION - FULL PIPELINE")
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

            if result.get('aus') and result['aus'].get('correlations'):
                print(f"  AU Correlations:")
                for zone in ['upper_face_mean', 'lower_face_mean', 'overall_mean']:
                    corr = result['aus']['correlations'].get(zone)
                    if corr is not None:
                        print(f"    {zone}: {corr:.3f}")

    # Print summary
    if all_results:
        print_summary(all_results)


if __name__ == '__main__':
    main()
