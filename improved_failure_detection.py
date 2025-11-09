#!/usr/bin/env python3
"""
Improved failure detection based on anatomical landmark coverage.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector


def validate_landmark_coverage(landmarks, bbox, frame_shape):
    """
    Check if all anatomical regions are properly covered by landmarks.

    68-point landmark indices:
    - Jaw: 0-16
    - Eyebrows: 17-26
    - Nose: 27-35
    - Eyes: 36-47
    - Mouth: 48-67
    """
    if landmarks is None or bbox is None:
        return False, ["No detection"], 0.0

    frame_h, frame_w = frame_shape[:2]
    failures = []

    # Extract anatomical regions
    jaw = landmarks[0:17]
    eyebrows = landmarks[17:27]
    nose = landmarks[27:36]
    eyes = landmarks[36:48]
    mouth = landmarks[48:68]

    # Calculate bbox dimensions
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    bbox_center_y = (bbox[1] + bbox[3]) / 2

    # 1. Check vertical coverage
    # Jaw should be in lower portion of bbox
    jaw_y_mean = np.mean(jaw[:, 1])
    jaw_y_max = np.max(jaw[:, 1])

    # Eyebrows should be in upper portion
    eyebrow_y_mean = np.mean(eyebrows[:, 1])
    eyebrow_y_min = np.min(eyebrows[:, 1])

    # Vertical span from eyebrows to jaw
    vertical_span = jaw_y_max - eyebrow_y_min
    expected_face_height = bbox_h * 0.7  # Landmarks should span 70%+ of bbox

    vertical_coverage = vertical_span / bbox_h

    if vertical_coverage < 0.6:
        failures.append(f"Insufficient vertical coverage ({vertical_coverage:.1%} of bbox)")

    # 2. Check if chin (jaw point 8) is near bottom of bbox
    chin = jaw[8]  # Center of jaw
    chin_to_bbox_bottom = bbox[3] - chin[1]

    # Chin should be within 20% of bbox height from bottom
    if chin_to_bbox_bottom > bbox_h * 0.25:
        failures.append(f"Chin too far from bbox bottom ({chin_to_bbox_bottom:.0f}px, {chin_to_bbox_bottom/bbox_h:.1%} of bbox height)")

    # 3. Check if eyebrows are near top of bbox
    eyebrow_to_bbox_top = eyebrow_y_min - bbox[1]

    # Eyebrows should be within 20% of bbox height from top
    if eyebrow_to_bbox_top > bbox_h * 0.25:
        failures.append(f"Eyebrows too far from bbox top ({eyebrow_to_bbox_top:.0f}px)")

    # 4. Check mouth position
    mouth_y_mean = np.mean(mouth[:, 1])
    mouth_position_ratio = (mouth_y_mean - bbox[1]) / bbox_h

    # Mouth should be in lower 60-85% of bbox
    if mouth_position_ratio < 0.6 or mouth_position_ratio > 0.9:
        failures.append(f"Mouth position suspicious ({mouth_position_ratio:.1%} from bbox top)")

    # 5. Check horizontal symmetry
    jaw_left = jaw[0:8]
    jaw_right = jaw[9:17]

    jaw_left_x_mean = np.mean(jaw_left[:, 0])
    jaw_right_x_mean = np.mean(jaw_right[:, 0])

    jaw_width = jaw_right_x_mean - jaw_left_x_mean
    bbox_x_center = (bbox[0] + bbox[2]) / 2
    jaw_center = (jaw_left_x_mean + jaw_right_x_mean) / 2

    horizontal_offset = abs(jaw_center - bbox_x_center)

    # Jaw center should align with bbox center (within 15%)
    if horizontal_offset > bbox_w * 0.15:
        failures.append(f"Poor horizontal alignment (offset: {horizontal_offset:.0f}px)")

    # 6. Check bbox size relative to frame
    # For faces filling the frame, bbox should be 15-40% of frame
    bbox_area_ratio = (bbox_w * bbox_h) / (frame_w * frame_h)

    if bbox_area_ratio < 0.15:
        failures.append(f"Bbox too small ({bbox_area_ratio:.1%} of frame) - likely partial face detection")

    # Calculate quality score
    quality_score = (
        min(vertical_coverage / 0.8, 1.0) * 0.4 +  # Vertical span
        (1 - min(chin_to_bbox_bottom / (bbox_h * 0.2), 1.0)) * 0.3 +  # Chin position
        (1 - min(eyebrow_to_bbox_top / (bbox_h * 0.2), 1.0)) * 0.3  # Eyebrow position
    )

    is_valid = len(failures) == 0

    return is_valid, failures, quality_score


# Test
test_videos = [
    ("IMG_8401_source", "FAILING", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV"),
    ("IMG_9330_source", "FAILING", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV"),
    ("IMG_0434_source", "WORKING", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0434_source.MOV"),
    ("IMG_0942_source", "WORKING", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0942_source.MOV"),
]

print("="*80)
print("IMPROVED FAILURE DETECTION - Anatomical Coverage")
print("="*80)
print()

detector = PyFaceAU68LandmarkDetector(debug_mode=False, use_clnf_refinement=True)

results = []

for name, expected, path in test_videos:
    print(f"\n{name} (Expected: {expected})")
    print("-"*80)

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    landmarks, _ = detector.get_face_mesh(frame)
    bbox = detector.cached_bbox.copy() if detector.cached_bbox is not None else None
    detector.reset_tracking_history()

    is_valid, failures, quality = validate_landmark_coverage(landmarks, bbox, frame.shape)

    predicted = "VALID" if is_valid else "INVALID"
    correct = "✅" if (predicted == "INVALID" and expected == "FAILING") or (predicted == "VALID" and expected == "WORKING") else "❌"

    print(f"Quality Score: {quality:.3f}")
    print(f"Predicted: {predicted}")
    print(f"Actual: {expected}")
    print(f"Result: {correct} {'CORRECT' if correct == '✅' else 'WRONG'}")

    if failures:
        print(f"Failures detected:")
        for f in failures:
            print(f"  - {f}")

    results.append({
        'name': name,
        'expected': expected,
        'predicted': predicted,
        'correct': correct == "✅",
        'quality': quality
    })

print()
print("="*80)
print("SUMMARY")
print("="*80)
print()

correct_count = sum(1 for r in results if r['correct'])
print(f"Accuracy: {correct_count}/{len(results)} ({correct_count/len(results):.1%})")
print()

for r in results:
    status = "✅" if r['correct'] else "❌"
    print(f"{status} {r['name']}: Expected {r['expected']}, Predicted {r['predicted']} (Quality: {r['quality']:.3f})")
