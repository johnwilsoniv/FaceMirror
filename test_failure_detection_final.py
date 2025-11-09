#!/usr/bin/env python3
"""
Test final failure detection on all 6 videos.
Goal: Reliably catch IMG_8401 and IMG_9330, pass the other 4.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector

test_videos = [
    ("IMG_8401_source", "SHOULD FAIL", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV"),
    ("IMG_9330_source", "SHOULD FAIL", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV"),
    ("IMG_0434_source", "SHOULD PASS", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0434_source.MOV"),
    ("IMG_0437_source", "SHOULD PASS", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0437_source.MOV"),
    ("IMG_0441_source", "SHOULD PASS", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0441_source.MOV"),
    ("IMG_0942_source", "SHOULD PASS", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0942_source.MOV"),
]

def validate_landmarks_final(landmarks, bbox, frame_shape):
    """
    Final validation using multiple checks.

    Returns:
        is_valid: bool
        reason: str (why it failed)
        confidence: float 0-1
    """
    if landmarks is None or bbox is None or len(landmarks) != 68:
        return False, "No detection", 0.0

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    # Check 1: Chin position (landmark 8)
    chin = landmarks[8]
    chin_to_bottom = bbox[3] - chin[1]
    chin_ratio = chin_to_bottom / bbox_h

    # Lowered threshold from 10% to 6% to catch IMG_9330
    if chin_ratio > 0.06:
        return False, f"Chin too far from bbox bottom ({chin_ratio:.1%})", 1.0 - chin_ratio

    # Check 2: Eyebrows position
    eyebrows = landmarks[17:27]
    eyebrow_y_min = np.min(eyebrows[:, 1])
    eyebrow_to_top = eyebrow_y_min - bbox[1]
    eyebrow_ratio = eyebrow_to_top / bbox_h

    # Eyebrows shouldn't be too close to top
    if eyebrow_ratio < 0.05:
        return False, f"Eyebrows too close to bbox top ({eyebrow_ratio:.1%})", eyebrow_ratio

    # Check 3: Landmark spread (should cover majority of bbox)
    x_min, x_max = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    y_min, y_max = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
    spread_x = x_max - x_min
    spread_y = y_max - y_min

    coverage_x = spread_x / bbox_w
    coverage_y = spread_y / bbox_h

    # Landmarks should cover at least 40% of bbox in each dimension
    if coverage_x < 0.4 or coverage_y < 0.4:
        return False, f"Insufficient spread ({coverage_x:.1%}×{coverage_y:.1%})", min(coverage_x, coverage_y)

    # All checks passed
    confidence = 1.0 - chin_ratio
    return True, "Valid", confidence


print("="*80)
print("FINAL FAILURE DETECTION TEST")
print("="*80)
print()

detector = PyFaceAU68LandmarkDetector(debug_mode=False, use_clnf_refinement=True)

results = []

for name, expected, path in test_videos:
    print(f"{name:<20} Expected: {expected:<12}", end=" ")

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    landmarks, _ = detector.get_face_mesh(frame)
    bbox = detector.cached_bbox.copy() if detector.cached_bbox is not None else None
    detector.reset_tracking_history()

    is_valid, reason, confidence = validate_landmarks_final(landmarks, bbox, frame.shape)

    predicted = "PASS" if is_valid else "FAIL"

    # Check if prediction matches expectation
    expected_result = "PASS" if "SHOULD PASS" in expected else "FAIL"
    correct = predicted == expected_result

    status = "✅" if correct else "❌"

    print(f"→ Predicted: {predicted:<6} {status} {reason} (conf={confidence:.2f})")

    results.append({
        'name': name,
        'expected': expected_result,
        'predicted': predicted,
        'correct': correct,
        'reason': reason,
        'confidence': confidence
    })

print()
print("="*80)
print("SUMMARY")
print("="*80)
print()

correct_count = sum(1 for r in results if r['correct'])
total = len(results)

print(f"Accuracy: {correct_count}/{total} ({correct_count/total*100:.1f}%)")
print()

if correct_count == total:
    print("✅ PERFECT - All videos correctly classified!")
else:
    print("Misclassifications:")
    for r in results:
        if not r['correct']:
            print(f"  ❌ {r['name']}: Expected {r['expected']}, Got {r['predicted']} - {r['reason']}")

print()
print("Failure detection thresholds:")
print("  - Chin-to-bottom: >6% of bbox height")
print("  - Eyebrow-to-top: <5% of bbox height")
print("  - Landmark spread: <40% bbox coverage")
