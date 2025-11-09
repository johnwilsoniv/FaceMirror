#!/usr/bin/env python3
"""
Detailed diagnostic comparing failing vs working cases.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector

test_videos = [
    ("IMG_8401", "FAIL", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV"),
    ("IMG_9330", "FAIL", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV"),
    ("IMG_0434", "WORK", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0434_source.MOV"),
    ("IMG_0942", "WORK", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0942_source.MOV"),
]

detector = PyFaceAU68LandmarkDetector(debug_mode=False, use_clnf_refinement=True)

print("="*100)
print(f"{'Video':<12} {'Status':<6} {'Bbox':<25} {'Chin-Bot':<10} {'Brow-Top':<10} {'Vert Span':<10} {'Mouth Y%':<10}")
print("="*100)

for name, status, path in test_videos:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    landmarks, _ = detector.get_face_mesh(frame)
    bbox = detector.cached_bbox.copy() if detector.cached_bbox is not None else None
    detector.reset_tracking_history()

    if landmarks is None or bbox is None:
        print(f"{name:<12} {status:<6} NO DETECTION")
        continue

    # Anatomical points
    chin = landmarks[8]  # Jaw center
    eyebrows = landmarks[17:27]
    mouth = landmarks[48:68]

    eyebrow_y_min = np.min(eyebrows[:, 1])
    mouth_y_mean = np.mean(mouth[:, 1])

    # Bbox measurements
    bbox_h = bbox[3] - bbox[1]

    chin_to_bbox_bottom = bbox[3] - chin[1]
    eyebrow_to_bbox_top = eyebrow_y_min - bbox[1]
    vertical_span = chin[1] - eyebrow_y_min
    vertical_coverage = vertical_span / bbox_h

    mouth_position_ratio = (mouth_y_mean - bbox[1]) / bbox_h

    bbox_str = f"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"

    print(f"{name:<12} {status:<6} {bbox_str:<25} "
          f"{chin_to_bbox_bottom:>5.0f}px {chin_to_bbox_bottom/bbox_h:>3.0%}  "
          f"{eyebrow_to_bbox_top:>5.0f}px {eyebrow_to_bbox_top/bbox_h:>3.0%}  "
          f"{vertical_coverage:>8.1%}    "
          f"{mouth_position_ratio:>8.1%}")

print("="*100)
print()
print("Key metric differences:")
print("  Chin-Bot: Distance from chin (landmark 8) to bbox bottom")
print("  Brow-Top: Distance from eyebrows to bbox top")
print("  Vert Span: Vertical coverage of landmarks within bbox")
print("  Mouth Y%: Mouth position as % from bbox top")
print()
print("Hypothesis: FAILING cases have chin far from bbox bottom (bbox cut off too high)")
