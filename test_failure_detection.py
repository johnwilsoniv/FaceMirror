#!/usr/bin/env python3
"""
Test automatic failure detection for landmark placement.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector


def validate_landmarks(landmarks, bbox, frame_shape):
    """
    Validate if landmarks are properly placed.

    Returns:
        is_valid: bool
        failure_reasons: list of strings
        quality_score: float 0-1
    """
    failure_reasons = []
    quality_metrics = {}

    if landmarks is None or bbox is None:
        return False, ["No detection"], 0.0

    frame_h, frame_w = frame_shape[:2]
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]

    # 1. Check landmark spread (should cover most of bbox)
    x_min, x_max = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    y_min, y_max = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
    landmark_w = x_max - x_min
    landmark_h = y_max - y_min

    bbox_coverage_x = landmark_w / bbox_w if bbox_w > 0 else 0
    bbox_coverage_y = landmark_h / bbox_h if bbox_h > 0 else 0

    quality_metrics['bbox_coverage_x'] = bbox_coverage_x
    quality_metrics['bbox_coverage_y'] = bbox_coverage_y

    # Landmarks should cover at least 50% of bbox in each dimension
    if bbox_coverage_x < 0.5:
        failure_reasons.append(f"Insufficient horizontal spread ({bbox_coverage_x:.1%} of bbox)")
    if bbox_coverage_y < 0.5:
        failure_reasons.append(f"Insufficient vertical spread ({bbox_coverage_y:.1%} of bbox)")

    # 2. Check if landmarks are actually within bbox (with tolerance)
    tolerance = 20  # pixels
    landmarks_outside_bbox = 0
    for x, y in landmarks:
        if (x < bbox[0] - tolerance or x > bbox[2] + tolerance or
            y < bbox[1] - tolerance or y > bbox[3] + tolerance):
            landmarks_outside_bbox += 1

    outside_ratio = landmarks_outside_bbox / len(landmarks)
    quality_metrics['outside_ratio'] = outside_ratio

    if outside_ratio > 0.3:
        failure_reasons.append(f"{outside_ratio:.1%} of landmarks outside bbox")

    # 3. Check bbox size relative to frame
    bbox_area_ratio = (bbox_w * bbox_h) / (frame_w * frame_h)
    quality_metrics['bbox_area_ratio'] = bbox_area_ratio

    # Bbox should be 10-50% of frame for typical faces
    if bbox_area_ratio < 0.10:
        failure_reasons.append(f"Bbox too small ({bbox_area_ratio:.1%} of frame)")
    elif bbox_area_ratio > 0.50:
        failure_reasons.append(f"Bbox too large ({bbox_area_ratio:.1%} of frame - face may fill entire frame)")

    # 4. Check landmark distribution (variance)
    # Properly placed landmarks should have good spatial distribution
    x_variance = np.var(landmarks[:, 0])
    y_variance = np.var(landmarks[:, 1])

    expected_variance = (bbox_w * bbox_h) / 100  # Heuristic
    variance_score = min(1.0, (x_variance + y_variance) / (2 * expected_variance))
    quality_metrics['variance_score'] = variance_score

    if variance_score < 0.3:
        failure_reasons.append(f"Landmarks too clustered (variance score: {variance_score:.2f})")

    # 5. Check specific anatomical relationships
    # Jaw width should be wider than eye region
    jaw_points = landmarks[0:17]  # Jaw outline
    jaw_width = np.max(jaw_points[:, 0]) - np.min(jaw_points[:, 0])

    eye_points = landmarks[36:48]  # Eyes
    eye_width = np.max(eye_points[:, 0]) - np.min(eye_points[:, 0])

    if jaw_width < eye_width * 1.2:
        failure_reasons.append(f"Jaw width suspicious ({jaw_width:.0f}px vs eyes {eye_width:.0f}px)")

    # Calculate overall quality score
    quality_score = (
        bbox_coverage_x * 0.25 +
        bbox_coverage_y * 0.25 +
        (1 - outside_ratio) * 0.2 +
        variance_score * 0.3
    )

    # Penalize if bbox size is wrong
    if bbox_area_ratio < 0.10 or bbox_area_ratio > 0.50:
        quality_score *= 0.5

    is_valid = len(failure_reasons) == 0

    return is_valid, failure_reasons, quality_score, quality_metrics


# Test on all videos
test_videos = [
    ("IMG_8401_source", "Paralysis - Surgical Markings", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV"),
    ("IMG_9330_source", "Paralysis - Severe", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV"),
    ("IMG_0434_source", "Normal Cohort", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0434_source.MOV"),
    ("IMG_0942_source", "Normal Cohort", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0942_source.MOV"),
]

print("="*80)
print("AUTOMATIC FAILURE DETECTION TEST")
print("="*80)
print()

detector = PyFaceAU68LandmarkDetector(
    debug_mode=False,
    use_clnf_refinement=True,
    skip_redetection=False
)

for name, category, path in test_videos:
    print()
    print("="*80)
    print(f"{name} ({category})")
    print("="*80)

    # Load frame
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    # Detect
    landmarks, _ = detector.get_face_mesh(frame)
    bbox = detector.cached_bbox.copy() if detector.cached_bbox is not None else None
    detector.reset_tracking_history()

    # Validate
    is_valid, reasons, quality_score, metrics = validate_landmarks(landmarks, bbox, frame.shape)

    print(f"\nValidation Results:")
    print(f"  Quality Score: {quality_score:.3f}")
    print(f"  Status: {'✅ VALID' if is_valid else '❌ INVALID'}")

    if not is_valid:
        print(f"  Failure Reasons:")
        for reason in reasons:
            print(f"    - {reason}")

    print(f"\n  Detailed Metrics:")
    print(f"    Bbox coverage: {metrics['bbox_coverage_x']:.1%} × {metrics['bbox_coverage_y']:.1%}")
    print(f"    Bbox area ratio: {metrics['bbox_area_ratio']:.1%} of frame")
    print(f"    Outside bbox ratio: {metrics['outside_ratio']:.1%}")
    print(f"    Variance score: {metrics['variance_score']:.2f}")

print()
print("="*80)
print("SUMMARY")
print("="*80)
print()
print("Automatic validation can detect failures based on:")
print("  1. Landmark spread (must cover 50%+ of bbox)")
print("  2. Landmarks inside bbox (70%+ must be within bbox)")
print("  3. Bbox size (10-50% of frame)")
print("  4. Spatial distribution (variance score)")
print("  5. Anatomical relationships (jaw vs eye width)")
print()
