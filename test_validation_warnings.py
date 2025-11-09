#!/usr/bin/env python3
"""
Test validation warnings for problematic videos.
"""

import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector

test_videos = [
    ("IMG_8401_source", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV", "PROBLEM"),
    ("IMG_9330_source", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV", "PROBLEM"),
    ("IMG_0434_source", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0434_source.MOV", "GOOD"),
]

print("="*80)
print("VALIDATION WARNING TEST")
print("="*80)
print()

# Test with debug mode ON to see all warnings
detector = PyFaceAU68LandmarkDetector(debug_mode=True, use_clnf_refinement=True)

for name, path, expected in test_videos:
    print(f"\n{'='*80}")
    print(f"Testing: {name} (Expected: {expected})")
    print(f"{'='*80}")

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ Failed to read frame")
        continue

    landmarks, validation_info = detector.get_face_mesh(frame)
    detector.reset_tracking_history()

    if landmarks is None:
        print(f"❌ No detection")
        continue

    if validation_info:
        print(f"\nValidation Results:")
        print(f"  Passed: {validation_info['validation_passed']}")
        print(f"  Reason: {validation_info['reason']}")
        print(f"  Confidence: {validation_info['confidence']:.2f}")
        print(f"  Used Fallback: {validation_info['used_fallback']}")

        if not validation_info['validation_passed']:
            print(f"\n  ⚠️  WARNING: This video will show warnings during processing!")
        else:
            print(f"\n  ✅ This video passed validation - no warnings expected")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print()
print("Summary:")
print("  - PROBLEM videos (8401, 9330) should show validation warnings")
print("  - GOOD videos (0434) should pass without warnings")
print()
