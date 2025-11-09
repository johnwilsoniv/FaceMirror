#!/usr/bin/env python3
"""
Analyze all metrics for all 6 videos to find best failure detection combination.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector

test_videos = [
    ("IMG_8401_source", "FAIL"),
    ("IMG_9330_source", "FAIL"),
    ("IMG_0434_source", "PASS"),
    ("IMG_0437_source", "PASS"),
    ("IMG_0441_source", "PASS"),
    ("IMG_0942_source", "PASS"),
]

base_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/"

detector = PyFaceAU68LandmarkDetector(debug_mode=False, use_clnf_refinement=True)

print("="*100)
print(f"{'Video':<18} {'Should':<6} {'Chin%':<8} {'Brow%':<8} {'Spread%':<10} {'Mouth%':<8} {'Jaw Width':<10}")
print("="*100)

for name, should in test_videos:
    path = base_path + name + ".MOV"

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    landmarks, _ = detector.get_face_mesh(frame)
    bbox = detector.cached_bbox.copy() if detector.cached_bbox is not None else None
    detector.reset_tracking_history()

    if landmarks is None or bbox is None:
        print(f"{name:<18} {should:<6} NO DETECTION")
        continue

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    # Metric 1: Chin position
    chin = landmarks[8]
    chin_to_bottom = bbox[3] - chin[1]
    chin_pct = chin_to_bottom / bbox_h * 100

    # Metric 2: Eyebrow position
    eyebrows = landmarks[17:27]
    eyebrow_y_min = np.min(eyebrows[:, 1])
    eyebrow_to_top = eyebrow_y_min - bbox[1]
    eyebrow_pct = eyebrow_to_top / bbox_h * 100

    # Metric 3: Landmark spread
    x_min, x_max = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    y_min, y_max = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
    spread_x = (x_max - x_min) / bbox_w * 100
    spread_y = (y_max - y_min) / bbox_h * 100

    # Metric 4: Mouth position (should be in lower 60-85% of face)
    mouth = landmarks[48:68]
    mouth_y_mean = np.mean(mouth[:, 1])
    mouth_from_top_pct = (mouth_y_mean - bbox[1]) / bbox_h * 100

    # Metric 5: Jaw width vs eye region width
    jaw_points = landmarks[0:17]
    jaw_width = np.max(jaw_points[:, 0]) - np.min(jaw_points[:, 0])

    eye_points = landmarks[36:48]
    eye_width = np.max(eye_points[:, 0]) - np.min(eye_points[:, 0])

    jaw_to_eye_ratio = jaw_width / eye_width if eye_width > 0 else 0

    print(f"{name:<18} {should:<6} {chin_pct:>6.1f}%  {eyebrow_pct:>6.1f}%  "
          f"{spread_x:>5.1f}×{spread_y:<3.1f}%  {mouth_from_top_pct:>6.1f}%  {jaw_to_eye_ratio:>6.2f}")

print("="*100)
print()
print("Analysis:")
print("  FAIL videos should have:")
print("    - Chin%: HIGHER (bbox cut off at bottom)")
print("    - Brow%: varies")
print("    - Spread%: LOWER (clustered landmarks)")
print("    - Mouth%: varies")
print("    - Jaw/Eye ratio: LOWER (jaw not wide enough)")
