#!/usr/bin/env python3
"""
Test MTCNN fallback on the 6 test videos.

Expected results:
- IMG_8401: RetinaFace FAILS → MTCNN fallback should succeed
- IMG_9330: RetinaFace FAILS → MTCNN fallback should succeed
- IMG_0434, IMG_0437, IMG_0441, IMG_0942: RetinaFace PASSES (no fallback needed)
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector

test_videos = [
    ("IMG_8401_source", "SHOULD FAIL → MTCNN", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV"),
    ("IMG_9330_source", "SHOULD FAIL → MTCNN", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV"),
    ("IMG_0434_source", "SHOULD PASS", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0434_source.MOV"),
    ("IMG_0437_source", "SHOULD PASS", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0437_source.MOV"),
    ("IMG_0441_source", "SHOULD PASS", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0441_source.MOV"),
    ("IMG_0942_source", "SHOULD PASS", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0942_source.MOV"),
]

print("="*80)
print("MTCNN FALLBACK TEST")
print("="*80)
print()

# Initialize detector with debug mode to see fallback messages
detector = PyFaceAU68LandmarkDetector(debug_mode=True, use_clnf_refinement=True)

print("\n" + "="*80)
print("TESTING VIDEOS")
print("="*80)

results = []

for name, expected, path in test_videos:
    print(f"\n{'='*80}")
    print(f"Video: {name}")
    print(f"Expected: {expected}")
    print(f"{'='*80}")

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ Failed to read frame from {name}")
        results.append({
            'name': name,
            'result': 'VIDEO_ERROR',
            'expected': expected
        })
        continue

    landmarks, _ = detector.get_face_mesh(frame)
    bbox = detector.cached_bbox.copy() if detector.cached_bbox is not None else None
    detector.reset_tracking_history()

    if landmarks is None or bbox is None:
        print(f"❌ No detection")
        results.append({
            'name': name,
            'result': 'NO_DETECTION',
            'expected': expected
        })
        continue

    # Validate
    from pyfaceau_detector import PyFaceAU68LandmarkDetector as Det
    temp_det = Det(debug_mode=False, use_clnf_refinement=False)
    is_valid, reason, confidence = temp_det._validate_landmarks(landmarks, bbox)

    status = "✅ PASS" if is_valid else "❌ FAIL"
    print(f"\nResult: {status} - {reason} (confidence={confidence:.2f})")

    results.append({
        'name': name,
        'result': 'PASS' if is_valid else 'FAIL',
        'reason': reason,
        'confidence': confidence,
        'expected': expected
    })

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print()

for r in results:
    name = r['name']
    result = r['result']
    expected = r['expected']

    if result in ['VIDEO_ERROR', 'NO_DETECTION']:
        print(f"❌ {name:<20} {result:<15} (Expected: {expected})")
    else:
        reason = r['reason']
        conf = r['confidence']

        # Check if result matches expectation
        if "SHOULD FAIL" in expected and result == "FAIL":
            status = "⚠️  STILL FAILING (fallback didn't help)"
        elif "SHOULD FAIL" in expected and result == "PASS":
            status = "✅ FIXED BY FALLBACK"
        elif "SHOULD PASS" in expected and result == "PASS":
            status = "✅ PASSED (no fallback needed)"
        elif "SHOULD PASS" in expected and result == "FAIL":
            status = "❌ REGRESSION (should pass but failed)"
        else:
            status = "?"

        print(f"{status} {name:<20} {result:<6} {reason:<30} (conf={conf:.2f})")

print()

# Count successes
pass_count = sum(1 for r in results if r['result'] == 'PASS')
total = len(results)

print(f"Success rate: {pass_count}/{total} ({pass_count/total*100:.1f}%)")

if pass_count == total:
    print("\n✅ PERFECT - All videos passed validation!")
else:
    print(f"\n⚠️  {total - pass_count} video(s) still failing")

print()
