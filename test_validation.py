#!/usr/bin/env python3
"""
Test the chin-based validation on all 4 videos.
"""

import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector

test_videos = [
    ("IMG_8401_source", "SHOULD FAIL VALIDATION"),
    ("IMG_9330_source", "SHOULD FAIL VALIDATION"),
    ("IMG_0434_source", "SHOULD PASS VALIDATION"),
    ("IMG_0942_source", "SHOULD PASS VALIDATION"),
]

base_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/"

print("="*80)
print("VALIDATION TEST - Chin-based failure detection")
print("="*80)
print()

detector = PyFaceAU68LandmarkDetector(debug_mode=True, use_clnf_refinement=True)

results = []

for name, expected in test_videos:
    print()
    print(f"Testing: {name} ({expected})")
    print("-"*80)

    path = base_path + name + ".MOV"

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    landmarks, _ = detector.get_face_mesh(frame)
    detector.reset_tracking_history()

    # Note: validation warning should have been printed during get_face_mesh if debug_mode=True

print()
print("="*80)
print("Check output above for validation warnings (⚠️)")
print("Expected: IMG_8401 and IMG_9330 should show validation failures")
print("Expected: IMG_0434 and IMG_0942 should NOT show validation failures")
print("="*80)
