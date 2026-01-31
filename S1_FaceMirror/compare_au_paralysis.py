#!/usr/bin/env python3
"""
AU Comparison Script for Paralysis Analysis

Compares Action Unit (AU) predictions between CLNF and SPIGA landmark detectors
on paralyzed vs normal hemifaces. Uses the expert key file to identify paralyzed side.

Usage:
    python compare_au_paralysis.py --video "../S Data/Paralysis Cohort/IMG_8401.MOV"
    python compare_au_paralysis.py --video "../S Data/Paralysis Cohort/IMG_8401.MOV" --key-file "../S3 Data Analysis/FPRS FP Key.csv"
    python compare_au_paralysis.py --video "../S Data/Paralysis Cohort/IMG_8401.MOV" --detector spiga
    python compare_au_paralysis.py --video "../S Data/Paralysis Cohort/IMG_8401.MOV" --compare-detectors

Dependencies:
    - pyfaceau: AU extraction pipeline
    - spiga, facenet-pytorch: For SPIGA detector (optional)
"""

import argparse
import csv
import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def safe_print(*args, **kwargs):
    """Print wrapper that handles BrokenPipeError."""
    try:
        print(*args, **kwargs)
    except (BrokenPipeError, IOError):
        pass


def load_expert_key(key_file: str) -> Dict[str, dict]:
    """
    Load expert key file with paralysis annotations.

    Args:
        key_file: Path to FPRS FP Key.csv

    Returns:
        Dictionary mapping patient ID to paralysis info
    """
    key_data = {}

    with open(key_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row['Patient']

            # Determine paralyzed side based on annotations
            left_paralysis = any(
                row.get(f'Paralysis - Left {region}', 'None') not in ['None', 'Not Assessed', '']
                for region in ['Upper Face', 'Mid Face', 'Lower Face']
            )
            right_paralysis = any(
                row.get(f'Paralysis - Right {region}', 'None') not in ['None', 'Not Assessed', '']
                for region in ['Upper Face', 'Mid Face', 'Lower Face']
            )

            # Determine paralysis severity
            def get_severity(side: str) -> str:
                regions = [row.get(f'Paralysis - {side} {r}', 'None')
                           for r in ['Upper Face', 'Mid Face', 'Lower Face']]
                if 'Complete' in regions:
                    return 'Complete'
                elif 'Partial' in regions:
                    return 'Partial'
                return 'None'

            key_data[patient_id] = {
                'left_paralysis': left_paralysis,
                'right_paralysis': right_paralysis,
                'left_severity': get_severity('Left'),
                'right_severity': get_severity('Right'),
                'paralyzed_side': 'left' if left_paralysis and not right_paralysis else
                                  'right' if right_paralysis and not left_paralysis else
                                  'bilateral' if left_paralysis and right_paralysis else 'none',
                'fitzpatrick': row.get('Fitzpatrick', ''),
            }

    return key_data


def get_patient_id(video_path: str) -> str:
    """Extract patient ID from video filename."""
    stem = Path(video_path).stem
    # Handle both IMG_XXXX and timestamp formats
    return stem


def create_detector(detector_type: str, debug_mode: bool = False):
    """
    Create landmark detector based on type.

    Args:
        detector_type: 'clnf' or 'spiga'
        debug_mode: Enable debug output

    Returns:
        Detector instance
    """
    if detector_type == 'spiga':
        from spiga_detector import SPIGALandmarkDetector
        return SPIGALandmarkDetector(debug_mode=debug_mode)
    else:
        from pyfaceau_detector import PyFaceAU68LandmarkDetector
        return PyFaceAU68LandmarkDetector(debug_mode=debug_mode)


def extract_hemiface_landmarks(landmarks_68: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split 68-point landmarks into left and right hemiface.

    Dlib 68-point landmark layout:
    - 0-16: Jawline (0-8 right, 8-16 left)
    - 17-21: Right eyebrow
    - 22-26: Left eyebrow
    - 27-35: Nose (midline)
    - 36-41: Right eye
    - 42-47: Left eye
    - 48-59: Outer mouth (48-54 right, 54-59 left)
    - 60-67: Inner mouth (60-63 right, 64-67 left)

    Note: "Left" and "Right" refer to the subject's perspective (anatomical).

    Args:
        landmarks_68: (68, 2) array of landmarks

    Returns:
        right_hemiface: Landmarks for subject's right side (image left)
        left_hemiface: Landmarks for subject's left side (image right)
    """
    # Subject's right side (appears on left of image)
    right_indices = list(range(0, 9)) + list(range(17, 22)) + [27, 28, 29, 30, 31, 33] + \
                    list(range(36, 42)) + [48, 49, 50, 58, 59, 60, 61, 67]

    # Subject's left side (appears on right of image)
    left_indices = list(range(8, 17)) + list(range(22, 27)) + [27, 28, 29, 30, 32, 34, 35] + \
                   list(range(42, 48)) + [52, 53, 54, 55, 56, 62, 63, 64, 65, 66]

    # Extract landmarks for each side
    right_hemiface = landmarks_68[right_indices]
    left_hemiface = landmarks_68[left_indices]

    return right_hemiface, left_hemiface


def calculate_au_asymmetry(landmarks_68: np.ndarray) -> Dict[str, float]:
    """
    Calculate AU-related asymmetry metrics from landmarks.

    Measures:
    - Eye opening asymmetry (AU45)
    - Brow height asymmetry (AU1/AU2/AU4)
    - Mouth corner asymmetry (AU12/AU15)
    - Nasolabial fold depth proxy

    Args:
        landmarks_68: (68, 2) array of landmarks

    Returns:
        Dictionary of asymmetry metrics (positive = right side more active)
    """
    metrics = {}

    # Eye opening (EAR proxy) - AU45 related
    # Right eye: 36-41, Left eye: 42-47
    right_eye = landmarks_68[36:42]
    left_eye = landmarks_68[42:48]

    def eye_aspect_ratio(eye):
        # Vertical distances
        v1 = np.linalg.norm(eye[1] - eye[5])
        v2 = np.linalg.norm(eye[2] - eye[4])
        # Horizontal distance
        h = np.linalg.norm(eye[0] - eye[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0

    right_ear = eye_aspect_ratio(right_eye)
    left_ear = eye_aspect_ratio(left_eye)
    metrics['eye_opening_asymmetry'] = right_ear - left_ear

    # Brow height - AU1/AU2/AU4 related
    # Right brow: 17-21, Left brow: 22-26
    # Measure relative to eye corners
    right_brow_height = np.mean(landmarks_68[17:22, 1]) - np.mean(landmarks_68[36:42, 1])
    left_brow_height = np.mean(landmarks_68[22:27, 1]) - np.mean(landmarks_68[42:48, 1])
    metrics['brow_height_asymmetry'] = right_brow_height - left_brow_height

    # Mouth corner height - AU12 (smile) related
    # Mouth corners: 48 (right), 54 (left)
    # Reference: nose tip (30) or chin (8)
    nose_tip = landmarks_68[30]
    right_mouth_height = nose_tip[1] - landmarks_68[48, 1]
    left_mouth_height = nose_tip[1] - landmarks_68[54, 1]
    metrics['mouth_corner_asymmetry'] = right_mouth_height - left_mouth_height

    # Mouth width asymmetry (relaxed vs contracted)
    face_midline_x = np.mean([landmarks_68[27, 0], landmarks_68[30, 0], landmarks_68[33, 0]])
    right_mouth_width = face_midline_x - landmarks_68[48, 0]
    left_mouth_width = landmarks_68[54, 0] - face_midline_x
    metrics['mouth_width_asymmetry'] = right_mouth_width - left_mouth_width

    # Nasolabial region (approximate via nose-to-mouth distance)
    # Points 31-35 are nose base, 48-54 are outer mouth
    right_nasolabial = np.linalg.norm(landmarks_68[31] - landmarks_68[48])
    left_nasolabial = np.linalg.norm(landmarks_68[35] - landmarks_68[54])
    metrics['nasolabial_asymmetry'] = right_nasolabial - left_nasolabial

    return metrics


def process_video(video_path: str, detector, num_frames: int = 100,
                  visualize: bool = False) -> List[Dict[str, float]]:
    """
    Process video and extract landmark-based asymmetry metrics.

    Args:
        video_path: Path to input video
        detector: Landmark detector instance
        num_frames: Number of frames to sample
        visualize: Show landmark overlay (for debugging)

    Returns:
        List of asymmetry metrics per frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)

    results = []
    frame_idx = 0

    detector.reset_tracking_history()

    safe_print(f"Processing {num_frames} frames from {Path(video_path).name}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            landmarks, info = detector.get_face_mesh(frame)

            if landmarks is not None and info.get('valid', False):
                metrics = calculate_au_asymmetry(landmarks)
                metrics['frame_idx'] = frame_idx
                results.append(metrics)

                if visualize:
                    # Draw landmarks
                    vis_frame = frame.copy()
                    for i, (x, y) in enumerate(landmarks):
                        color = (0, 255, 0) if i < 17 else (255, 0, 0)  # Green jaw, blue features
                        cv2.circle(vis_frame, (int(x), int(y)), 2, color, -1)

                    cv2.imshow('Landmarks', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        frame_idx += 1

    cap.release()
    if visualize:
        cv2.destroyAllWindows()

    return results


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate frame-level metrics into summary statistics.

    Args:
        metrics_list: List of per-frame metric dictionaries

    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not metrics_list:
        return {}

    # Get all metric keys (exclude frame_idx)
    keys = [k for k in metrics_list[0].keys() if k != 'frame_idx']

    summary = {}
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }

    return summary


def print_comparison_report(patient_id: str, paralysis_info: dict,
                            metrics_summary: Dict[str, Dict[str, float]],
                            detector_name: str):
    """
    Print formatted comparison report.

    Args:
        patient_id: Patient identifier
        paralysis_info: Paralysis annotation from expert key
        metrics_summary: Aggregated asymmetry metrics
        detector_name: Name of detector used
    """
    safe_print("\n" + "="*70)
    safe_print(f"AU ASYMMETRY ANALYSIS: {patient_id}")
    safe_print("="*70)
    safe_print(f"Detector: {detector_name}")
    safe_print(f"Paralyzed Side: {paralysis_info.get('paralyzed_side', 'unknown').upper()}")
    safe_print(f"Left Severity: {paralysis_info.get('left_severity', 'N/A')}")
    safe_print(f"Right Severity: {paralysis_info.get('right_severity', 'N/A')}")
    safe_print("-"*70)
    safe_print(f"{'Metric':<30} {'Mean':>10} {'Std':>10} {'Interpretation':<20}")
    safe_print("-"*70)

    interpretations = {
        'eye_opening_asymmetry': ('AU45', 'Positive = right eye more open'),
        'brow_height_asymmetry': ('AU1/2/4', 'Positive = right brow higher'),
        'mouth_corner_asymmetry': ('AU12', 'Positive = right corner higher'),
        'mouth_width_asymmetry': ('AU12/15', 'Positive = right side wider'),
        'nasolabial_asymmetry': ('Region', 'Positive = right fold deeper'),
    }

    for metric, stats in metrics_summary.items():
        au_info, interp = interpretations.get(metric, ('', ''))
        safe_print(f"{metric:<30} {stats['mean']:>10.4f} {stats['std']:>10.4f} {interp:<20}")

    safe_print("-"*70)

    # Interpretation based on paralyzed side
    paralyzed_side = paralysis_info.get('paralyzed_side', 'unknown')
    if paralyzed_side == 'left':
        safe_print("\nExpected: Positive asymmetry values (right side more active)")
        safe_print("If metrics are positive, normal side shows more expression.")
    elif paralyzed_side == 'right':
        safe_print("\nExpected: Negative asymmetry values (left side more active)")
        safe_print("If metrics are negative, normal side shows more expression.")
    elif paralyzed_side == 'bilateral':
        safe_print("\nBilateral paralysis: Asymmetry may be less pronounced.")
    else:
        safe_print("\nNo paralysis annotated: This may be a normal control.")

    safe_print("="*70 + "\n")


def compare_detectors(video_path: str, key_data: dict, num_frames: int = 100):
    """
    Compare CLNF and SPIGA detectors on the same video.

    Args:
        video_path: Path to input video
        key_data: Expert key data
        num_frames: Number of frames to sample
    """
    patient_id = get_patient_id(video_path)
    paralysis_info = key_data.get(patient_id, {
        'paralyzed_side': 'unknown',
        'left_severity': 'N/A',
        'right_severity': 'N/A'
    })

    safe_print(f"\n{'#'*70}")
    safe_print(f"DETECTOR COMPARISON: {patient_id}")
    safe_print(f"{'#'*70}")

    # Process with CLNF
    safe_print("\n[1/2] Processing with CLNF detector...")
    try:
        clnf_detector = create_detector('clnf', debug_mode=False)
        clnf_metrics = process_video(video_path, clnf_detector, num_frames)
        clnf_summary = aggregate_metrics(clnf_metrics)
        print_comparison_report(patient_id, paralysis_info, clnf_summary, "CLNF (pymtcnn + pyclnf)")
    except Exception as e:
        safe_print(f"CLNF processing failed: {e}")
        clnf_summary = {}

    # Process with SPIGA
    safe_print("\n[2/2] Processing with SPIGA detector...")
    try:
        spiga_detector = create_detector('spiga', debug_mode=False)
        spiga_metrics = process_video(video_path, spiga_detector, num_frames)
        spiga_summary = aggregate_metrics(spiga_metrics)
        print_comparison_report(patient_id, paralysis_info, spiga_summary, "SPIGA (facenet-pytorch + spiga)")
    except ImportError as e:
        safe_print(f"SPIGA not available: {e}")
        safe_print("Install with: pip install spiga facenet-pytorch")
        spiga_summary = {}
    except Exception as e:
        safe_print(f"SPIGA processing failed: {e}")
        spiga_summary = {}

    # Side-by-side comparison
    if clnf_summary and spiga_summary:
        safe_print("\n" + "="*70)
        safe_print("SIDE-BY-SIDE COMPARISON")
        safe_print("="*70)
        safe_print(f"{'Metric':<30} {'CLNF Mean':>12} {'SPIGA Mean':>12} {'Diff':>10}")
        safe_print("-"*70)

        for metric in clnf_summary.keys():
            clnf_val = clnf_summary[metric]['mean']
            spiga_val = spiga_summary.get(metric, {}).get('mean', float('nan'))
            diff = spiga_val - clnf_val if not np.isnan(spiga_val) else float('nan')
            safe_print(f"{metric:<30} {clnf_val:>12.4f} {spiga_val:>12.4f} {diff:>10.4f}")

        safe_print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare AU predictions between landmark detectors on paralyzed faces'
    )
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--key-file', default='../S3 Data Analysis/FPRS FP Key.csv',
                        help='Path to expert key CSV file')
    parser.add_argument('--detector', choices=['clnf', 'spiga'], default='clnf',
                        help='Landmark detector to use')
    parser.add_argument('--compare-detectors', action='store_true',
                        help='Compare both CLNF and SPIGA detectors')
    parser.add_argument('--num-frames', type=int, default=100,
                        help='Number of frames to sample')
    parser.add_argument('--visualize', action='store_true',
                        help='Show landmark overlay visualization')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')

    args = parser.parse_args()

    # Resolve paths
    video_path = Path(args.video).resolve()
    key_file = Path(args.key_file).resolve()

    if not video_path.exists():
        safe_print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    # Load expert key
    key_data = {}
    if key_file.exists():
        safe_print(f"Loading expert key from: {key_file}")
        key_data = load_expert_key(str(key_file))
        safe_print(f"Loaded {len(key_data)} patient annotations")
    else:
        safe_print(f"Warning: Expert key file not found: {key_file}")
        safe_print("Proceeding without paralysis annotations")

    # Get patient info
    patient_id = get_patient_id(str(video_path))
    paralysis_info = key_data.get(patient_id, {
        'paralyzed_side': 'unknown',
        'left_severity': 'N/A',
        'right_severity': 'N/A'
    })

    if args.compare_detectors:
        compare_detectors(str(video_path), key_data, args.num_frames)
    else:
        # Single detector mode
        safe_print(f"\nProcessing with {args.detector.upper()} detector...")
        try:
            detector = create_detector(args.detector, debug_mode=args.debug)
            metrics = process_video(str(video_path), detector, args.num_frames, args.visualize)
            summary = aggregate_metrics(metrics)
            detector_name = "CLNF (pymtcnn + pyclnf)" if args.detector == 'clnf' else "SPIGA (facenet-pytorch + spiga)"
            print_comparison_report(patient_id, paralysis_info, summary, detector_name)
        except ImportError as e:
            safe_print(f"Detector not available: {e}")
            if args.detector == 'spiga':
                safe_print("Install with: pip install spiga facenet-pytorch")
            sys.exit(1)


if __name__ == '__main__':
    main()
