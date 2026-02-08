#!/usr/bin/env python3
"""
Calibrate SPIGA alignment to match CLNF alignment.

Measures scale ratio and position offset between SPIGA and CLNF pipelines
across a diverse multi-patient dataset (3 paralysis + 2 normal patients,
10 evenly-spaced frames each = 50 frames total).

The calibration corrects two systematic differences:
1. Zoom: SPIGA rigid landmarks are more tightly clustered → larger Kabsch scale
2. Offset: SPIGA landmark centroid is shifted vs CLNF's CalcParams center

Usage:
    python calibrate_spiga.py
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent / 'pyfaceau'))
sys.path.insert(0, str(script_dir.parent / 'pyclnf'))

from config import apply_environment_settings
apply_environment_settings()

import numpy as np
import cv2
import json
import subprocess
from typing import Optional, Dict, List, Tuple


# Same rigid indices used by face_aligner's Kabsch alignment
RIGID_INDICES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]

# Calibration dataset: 3 paralysis + 2 normal patients
CALIBRATION_VIDEOS = [
    ('S Data/Paralysis Cohort/IMG_3324.MOV', 'paralysis'),
    ('S Data/Paralysis Cohort/IMG_5198.MOV', 'paralysis'),
    ('S Data/Paralysis Cohort/IMG_7251.MOV', 'paralysis'),
    ('S Data/Normal Cohort/IMG_0422.MOV', 'normal'),
    ('S Data/Normal Cohort/IMG_0428.MOV', 'normal'),
]

FRAMES_PER_VIDEO = 10


def get_video_rotation(video_path: str) -> int:
    """Get rotation metadata from video using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-select_streams', 'v:0',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        for stream in data.get('streams', []):
            for side_data in stream.get('side_data_list', []):
                if side_data.get('side_data_type') == 'Display Matrix':
                    if 'rotation' in side_data:
                        return int(side_data['rotation'])
            rotation = stream.get('tags', {}).get('rotate', 0)
            if rotation:
                return int(rotation)
    except Exception as e:
        print(f"Warning: Could not get video rotation: {e}")
    return 0


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate frame based on rotation metadata."""
    if rotation == 90 or rotation == -270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == -90 or rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180 or rotation == -180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def compute_rms_spread(points: np.ndarray) -> float:
    """Compute RMS spread of points from their centroid (same as Kabsch s_src)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    n = points.shape[0]
    return float(np.sqrt(np.sum(centered ** 2) / n))


def get_evenly_spaced_frames(total_frames: int, n: int) -> List[int]:
    """Get n evenly-spaced frame indices across a video (10%-100% of duration)."""
    return [int(total_frames * (i + 1) / (n + 1)) for i in range(n)]


def main():
    project_root = script_dir.parent
    weights_dir = script_dir / 'weights'

    print("=" * 80)
    print("SPIGA Alignment Calibration")
    print("=" * 80)
    print(f"Dataset: {len(CALIBRATION_VIDEOS)} videos x {FRAMES_PER_VIDEO} frames = "
          f"{len(CALIBRATION_VIDEOS) * FRAMES_PER_VIDEO} frames total")

    # Verify all videos exist
    video_paths = []
    for rel_path, cohort in CALIBRATION_VIDEOS:
        full_path = project_root / rel_path
        if not full_path.exists():
            print(f"Error: Video not found: {full_path}")
            return 1
        video_paths.append((full_path, cohort))
    print("All calibration videos found.\n")

    # Output directory for visualizations
    output_dir = script_dir / 'test_output' / 'calibration'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipelines (SPIGA first to avoid segfault from import ordering)
    print("Initializing SPIGA pipeline...")
    from spiga_detector import SPIGALandmarkDetector
    from pyfaceau.alignment.calc_params import CalcParams
    from pyfaceau.alignment.face_aligner import OpenFace22FaceAligner
    from pyfaceau.features.pdm import PDMParser
    import pyfhog

    spiga_detector = SPIGALandmarkDetector(debug_mode=False)

    pdm_file = str(weights_dir / 'In-the-wild_aligned_PDM_68.txt')
    pdm_parser = PDMParser(pdm_file)
    calc_params = CalcParams(pdm_parser, reg_factor=10.0)
    face_aligner = OpenFace22FaceAligner(
        pdm_file=pdm_file,
        sim_scale=0.7,
        output_size=(112, 112)
    )

    print("Initializing CLNF pipeline...")
    from pymtcnn.backends.onnx_backend import ONNXMTCNN
    from pyclnf import CLNF

    face_detector = ONNXMTCNN()
    landmark_detector = CLNF(
        detector=None,
        use_gpu=False,
        convergence_profile='video'
    )
    landmark_detector.optimizer.use_cpp_warp = False

    # Collect measurements
    all_scale_ratios = []
    all_offset_x = []
    all_offset_y = []
    per_patient = {}

    for video_idx, (video_path, cohort) in enumerate(video_paths):
        video_name = video_path.name
        print(f"\n{'='*60}")
        print(f"[{video_idx+1}/{len(video_paths)}] {video_name} ({cohort})")
        print(f"{'='*60}")

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation = get_video_rotation(str(video_path))
        frame_indices = get_evenly_spaced_frames(total_frames, FRAMES_PER_VIDEO)

        print(f"  Total frames: {total_frames}, Rotation: {rotation}°")
        print(f"  Sampling frames: {frame_indices}")

        patient_scale_ratios = []
        patient_offset_x = []
        patient_offset_y = []

        for frame_num in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"  Frame {frame_num}: failed to read")
                continue

            frame = rotate_frame(frame, rotation)

            # Reset temporal state before each non-consecutive frame
            spiga_detector.reset_tracking_history()
            landmark_detector.reset_temporal_state()

            # --- CLNF pipeline ---
            bboxes, _ = face_detector.detect(frame)
            if bboxes is None or len(bboxes) == 0:
                print(f"  Frame {frame_num}: no face detected (MTCNN)")
                continue

            x, y, w, h = bboxes[0][:4]
            clnf_landmarks, clnf_info = landmark_detector.fit(frame, (x, y, w, h), return_params=True)
            if clnf_landmarks is None or 'params' not in clnf_info:
                print(f"  Frame {frame_num}: CLNF failed")
                continue

            clnf_params_global = clnf_info['params'][:6]
            clnf_pose_tx = clnf_params_global[4]
            clnf_pose_ty = clnf_params_global[5]

            # CLNF rigid points RMS spread (from detected landmarks used by Kabsch)
            clnf_rigid = clnf_landmarks[RIGID_INDICES]
            s_src_clnf = compute_rms_spread(clnf_rigid)

            # --- SPIGA pipeline ---
            raw_landmarks, spiga_info = spiga_detector.get_face_mesh(frame, detection_interval=0)
            if raw_landmarks is None:
                print(f"  Frame {frame_num}: SPIGA failed")
                continue

            # Procrustes-correct (same as in pipeline)
            landmarks_for_alignment = calc_params.procrustes_correct(
                raw_landmarks, RIGID_INDICES
            )

            # SPIGA rigid points RMS spread (from Procrustes-corrected landmarks used by Kabsch)
            spiga_rigid = landmarks_for_alignment[RIGID_INDICES]
            s_src_spiga = compute_rms_spread(spiga_rigid)

            # SPIGA centroid (used as pose_tx/ty)
            spiga_centroid_x = float(np.mean(raw_landmarks[:, 0]))
            spiga_centroid_y = float(np.mean(raw_landmarks[:, 1]))

            # Compute calibration values
            scale_ratio = s_src_clnf / s_src_spiga
            offset_x = clnf_pose_tx - spiga_centroid_x
            offset_y = clnf_pose_ty - spiga_centroid_y

            patient_scale_ratios.append(scale_ratio)
            patient_offset_x.append(offset_x)
            patient_offset_y.append(offset_y)

            # --- Generate aligned face comparison images ---
            # CLNF aligned face
            clnf_pose_rz = clnf_params_global[3]
            clnf_aligned = face_aligner.align_face(
                frame, clnf_landmarks, clnf_pose_tx, clnf_pose_ty, p_rz=clnf_pose_rz
            )

            # SPIGA aligned face (uncorrected — raw Procrustes output)
            spiga_pose_tx_raw = spiga_centroid_x
            spiga_pose_ty_raw = spiga_centroid_y
            spiga_aligned_raw = face_aligner.align_face(
                frame, landmarks_for_alignment, spiga_pose_tx_raw, spiga_pose_ty_raw
            )

            # SPIGA aligned face (corrected — with calibration applied)
            corrected_landmarks = landmarks_for_alignment.copy()
            centroid = np.mean(corrected_landmarks, axis=0)
            corrected_landmarks = (corrected_landmarks - centroid) * scale_ratio + centroid
            spiga_pose_tx_corr = spiga_centroid_x + offset_x
            spiga_pose_ty_corr = spiga_centroid_y + offset_y
            spiga_aligned_corr = face_aligner.align_face(
                frame, corrected_landmarks, spiga_pose_tx_corr, spiga_pose_ty_corr
            )

            # Compute HOG correlations
            hog_clnf = pyfhog.extract_fhog_features(clnf_aligned, cell_size=8)
            hog_spiga_raw = pyfhog.extract_fhog_features(spiga_aligned_raw, cell_size=8)
            hog_spiga_corr = pyfhog.extract_fhog_features(spiga_aligned_corr, cell_size=8)
            corr_raw = float(np.corrcoef(hog_clnf.flatten(), hog_spiga_raw.flatten())[0, 1])
            corr_corr = float(np.corrcoef(hog_clnf.flatten(), hog_spiga_corr.flatten())[0, 1])

            # Save 3-panel comparison: CLNF | SPIGA raw | SPIGA corrected
            scale = 3
            panels = []
            labels = [
                f'CLNF',
                f'SPIGA raw (r={corr_raw:.3f})',
                f'SPIGA corr (r={corr_corr:.3f})',
            ]
            for img, label in zip(
                [clnf_aligned, spiga_aligned_raw, spiga_aligned_corr], labels
            ):
                scaled = cv2.resize(img, (112 * scale, 112 * scale), interpolation=cv2.INTER_NEAREST)
                cv2.putText(scaled, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                panels.append(scaled)
            comparison = np.hstack(panels)

            video_stem = video_path.stem
            out_path = output_dir / f'{video_stem}_frame{frame_num:05d}.jpg'
            cv2.imwrite(str(out_path), comparison)

            print(f"  Frame {frame_num:5d}: scale_ratio={scale_ratio:.4f}  "
                  f"offset=({offset_x:+.1f}, {offset_y:+.1f})  "
                  f"s_clnf={s_src_clnf:.1f}  s_spiga={s_src_spiga:.1f}  "
                  f"HOG raw={corr_raw:.3f} corr={corr_corr:.3f}")

        cap.release()

        if patient_scale_ratios:
            per_patient[video_name] = {
                'cohort': cohort,
                'n_frames': len(patient_scale_ratios),
                'scale_ratio': {
                    'mean': np.mean(patient_scale_ratios),
                    'std': np.std(patient_scale_ratios),
                },
                'offset_x': {
                    'mean': np.mean(patient_offset_x),
                    'std': np.std(patient_offset_x),
                },
                'offset_y': {
                    'mean': np.mean(patient_offset_y),
                    'std': np.std(patient_offset_y),
                },
            }
            all_scale_ratios.extend(patient_scale_ratios)
            all_offset_x.extend(patient_offset_x)
            all_offset_y.extend(patient_offset_y)

            print(f"\n  Patient summary ({len(patient_scale_ratios)} frames):")
            print(f"    scale_ratio: {np.mean(patient_scale_ratios):.4f} ± {np.std(patient_scale_ratios):.4f}")
            print(f"    offset_x:    {np.mean(patient_offset_x):+.2f} ± {np.std(patient_offset_x):.2f}")
            print(f"    offset_y:    {np.mean(patient_offset_y):+.2f} ± {np.std(patient_offset_y):.2f}")

    # Overall results
    print(f"\n{'='*80}")
    print("OVERALL CALIBRATION RESULTS")
    print(f"{'='*80}")
    print(f"\nTotal frames measured: {len(all_scale_ratios)}")

    if not all_scale_ratios:
        print("Error: No valid measurements collected!")
        return 1

    print(f"\nPer-patient breakdown:")
    print(f"  {'Video':<20} {'Cohort':<10} {'N':>3}  {'Scale Ratio':>14}  {'Offset X':>14}  {'Offset Y':>14}")
    print(f"  {'-'*80}")
    for name, stats in per_patient.items():
        sr = stats['scale_ratio']
        ox = stats['offset_x']
        oy = stats['offset_y']
        print(f"  {name:<20} {stats['cohort']:<10} {stats['n_frames']:>3}  "
              f"{sr['mean']:>6.4f}±{sr['std']:<6.4f}  "
              f"{ox['mean']:>+6.2f}±{ox['std']:<6.2f}  "
              f"{oy['mean']:>+6.2f}±{oy['std']:<6.2f}")

    mean_scale = np.mean(all_scale_ratios)
    std_scale = np.std(all_scale_ratios)
    mean_ox = np.mean(all_offset_x)
    std_ox = np.std(all_offset_x)
    mean_oy = np.mean(all_offset_y)
    std_oy = np.std(all_offset_y)

    print(f"\n  Overall ({len(all_scale_ratios)} frames):")
    print(f"    SCALE_CORRECTION = {mean_scale:.4f}  (std={std_scale:.4f}, "
          f"cv={std_scale/mean_scale*100:.1f}%)")
    print(f"    OFFSET_X         = {mean_ox:+.2f}    (std={std_ox:.2f})")
    print(f"    OFFSET_Y         = {mean_oy:+.2f}    (std={std_oy:.2f})")

    # Stability check
    cv_scale = std_scale / mean_scale * 100
    print(f"\n  Stability check:")
    print(f"    Scale CV: {cv_scale:.1f}% {'✓ PASS' if cv_scale < 10 else '✗ FAIL'} (threshold: <10%)")
    print(f"    Offset X std: {std_ox:.2f}px")
    print(f"    Offset Y std: {std_oy:.2f}px")

    # Output constants for copy-paste
    print(f"\n{'='*80}")
    print("CALIBRATION CONSTANTS (copy to source files)")
    print(f"{'='*80}")
    print(f"""
# SPIGA-to-CLNF alignment calibration constants
# Measured from {len(all_scale_ratios)} frames across {len(per_patient)} patients
# Scale: corrects SPIGA rigid points being more tightly clustered than CLNF
# Offset: corrects SPIGA landmark centroid shift vs CLNF CalcParams center
SPIGA_SCALE_CORRECTION = {mean_scale:.4f}
SPIGA_OFFSET_X = {mean_ox:.2f}
SPIGA_OFFSET_Y = {mean_oy:.2f}
""")

    print(f"Visualizations saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
