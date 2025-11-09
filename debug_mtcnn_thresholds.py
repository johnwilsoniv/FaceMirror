#!/usr/bin/env python3
"""
Debug MTCNN detection on failing videos with different thresholds.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN

test_videos = [
    ("IMG_8401_source", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV"),
    ("IMG_9330_source", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV"),
]

# Try different threshold combinations
threshold_configs = [
    ([0.6, 0.7, 0.7], "Default (strict)"),
    ([0.5, 0.6, 0.6], "Moderate"),
    ([0.4, 0.5, 0.5], "Relaxed"),
    ([0.3, 0.4, 0.4], "Very relaxed"),
]

pyfaceau_dir = Path(__file__).parent / "pyfaceau" / "pyfaceau" / "detectors"
mtcnn_weights = pyfaceau_dir / "openface_mtcnn_weights.pth"

print("="*80)
print("MTCNN THRESHOLD TUNING")
print("="*80)

for name, path in test_videos:
    print(f"\n{'='*80}")
    print(f"Video: {name}")
    print(f"{'='*80}")

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ Failed to read frame")
        continue

    for thresholds, desc in threshold_configs:
        print(f"\n  Thresholds {thresholds} ({desc}):")

        detector = OpenFaceMTCNN(
            weights_path=str(mtcnn_weights),
            min_face_size=60,
            thresholds=thresholds
        )

        try:
            bboxes, landmarks = detector.detect(frame, return_landmarks=True)

            if bboxes is not None and len(bboxes) > 0:
                print(f"    ✅ Found {len(bboxes)} face(s)")
                for i, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox.astype(int)
                    w = x2 - x1
                    h = y2 - y1
                    print(f"      Face {i}: bbox=[{x1},{y1},{x2},{y2}] size={w}x{h}")
            else:
                print(f"    ❌ No faces detected")

        except Exception as e:
            print(f"    ❌ Error: {e}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print()
print("If MTCNN finds faces with relaxed thresholds, we can adjust.")
print("If MTCNN finds no faces even with very relaxed thresholds,")
print("these videos may be too challenging for both detectors.")
print()
