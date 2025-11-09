#!/usr/bin/env python3
"""
Debug bbox coverage - check if landmarks extend outside bbox.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector

test_videos = [
    ("IMG_8401_source", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV"),
    ("IMG_9330_source", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV"),
    ("IMG_0434_source", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0434_source.MOV"),
]

print("="*80)
print("BBOX COVERAGE ANALYSIS")
print("="*80)
print()

detector = PyFaceAU68LandmarkDetector(debug_mode=False, use_clnf_refinement=True)

for name, path in test_videos:
    print(f"\n{name}:")

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    landmarks, _ = detector.get_face_mesh(frame)
    bbox = detector.cached_bbox.copy() if detector.cached_bbox is not None else None
    detector.reset_tracking_history()

    if landmarks is None or bbox is None:
        print("  No detection")
        continue

    # Check bbox coverage
    x1, y1, x2, y2 = bbox

    # Get landmark bounds
    lm_x_min, lm_x_max = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    lm_y_min, lm_y_max = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])

    # Calculate how much landmarks extend outside bbox
    left_overflow = max(0, x1 - lm_x_min)
    right_overflow = max(0, lm_x_max - x2)
    top_overflow = max(0, y1 - lm_y_min)
    bottom_overflow = max(0, lm_y_max - y2)

    total_overflow = left_overflow + right_overflow + top_overflow + bottom_overflow

    print(f"  Bbox: [{x1}, {y1}, {x2}, {y2}] ({x2-x1}×{y2-y1})")
    print(f"  Landmarks: x=[{lm_x_min:.0f}, {lm_x_max:.0f}] y=[{lm_y_min:.0f}, {lm_y_max:.0f}]")
    print(f"  Overflow: left={left_overflow:.0f}px, right={right_overflow:.0f}px, top={top_overflow:.0f}px, bottom={bottom_overflow:.0f}px")

    if total_overflow > 0:
        print(f"  ⚠️  LANDMARKS EXTEND OUTSIDE BBOX by {total_overflow:.0f}px total")
    else:
        print(f"  ✅ All landmarks contained within bbox")

    # Calculate percentage of landmarks outside
    landmarks_outside = 0
    for x, y in landmarks:
        if x < x1 or x > x2 or y < y1 or y > y2:
            landmarks_outside += 1

    pct_outside = landmarks_outside / len(landmarks) * 100
    print(f"  Landmarks outside bbox: {landmarks_outside}/68 ({pct_outside:.1f}%)")

print()
print("="*80)
print("CONCLUSION")
print("="*80)
print()
print("Landmarks extending outside bbox indicates:")
print("1. Bbox is too small/incorrectly positioned")
print("2. Validation should check landmark containment")
print("3. MTCNN fallback should be triggered in these cases")
print()
