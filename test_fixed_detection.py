#!/usr/bin/env python3
"""
Test the fixed bbox selection on failing videos.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector

test_videos = [
    ("IMG_8401_source (Was FAILING)", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV"),
    ("IMG_9330_source (Was FAILING)", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV"),
    ("IMG_0942_source (Was WORKING)", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0942_source.MOV"),
]

output_dir = Path(__file__).parent / "test_output" / "fixed_detection"
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("TESTING FIXED BBOX SELECTION")
print("="*80)
print()

# Initialize detector with debug mode to see bbox selection
detector = PyFaceAU68LandmarkDetector(
    debug_mode=True,
    use_clnf_refinement=True,
    skip_redetection=False
)

for name, path in test_videos:
    print()
    print("="*80)
    print(f"Testing: {name}")
    print("="*80)

    # Load frame
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    # Detect
    landmarks, _ = detector.get_face_mesh(frame)

    if landmarks is None:
        print("❌ FAILED - No landmarks detected")
        # Reset for next video
        detector.reset_tracking_history()
        continue

    # Analyze results (save bbox before reset)
    bbox = detector.cached_bbox.copy() if detector.cached_bbox is not None else None

    # Reset for next video
    detector.reset_tracking_history()
    x_min, x_max = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    y_min, y_max = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
    spread_x = x_max - x_min
    spread_y = y_max - y_min

    print(f"\nResults:")
    print(f"  Bbox: {bbox}")
    print(f"  Bbox size: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}")
    print(f"  Landmark spread: {spread_x:.0f}x{spread_y:.0f} pixels")

    # Check if landmarks seem properly distributed (not clustered)
    expected_spread = min(frame.shape[:2]) * 0.3
    actual_spread = (spread_x + spread_y) / 2

    if actual_spread > expected_spread:
        print(f"  ✅ GOOD: Landmarks well-distributed ({actual_spread:.0f}px spread)")
    else:
        print(f"  ⚠️  WARNING: Landmarks may be clustered ({actual_spread:.0f}px spread, expected >{expected_spread:.0f}px)")

    # Create visualization
    vis = frame.copy()

    # Draw bbox
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 3)

    # Draw landmarks with colors
    for i, (x, y) in enumerate(landmarks):
        if i < 17:  # Jaw
            color = (0, 255, 0)
        elif i < 27:  # Eyebrows
            color = (255, 0, 0)
        elif i < 36:  # Nose
            color = (0, 255, 255)
        elif i < 48:  # Eyes
            color = (255, 0, 255)
        else:  # Mouth
            color = (0, 128, 255)

        cv2.circle(vis, (int(x), int(y)), 5, color, -1)

    # Midline
    glabella, chin = detector.get_facial_midline(landmarks)
    if glabella is not None and chin is not None:
        cv2.line(vis, (int(glabella[0]), int(glabella[1])),
                (int(chin[0]), int(chin[1])), (0, 0, 255), 3)

    # Info
    info_text = [
        name.split('(')[0].strip(),
        "FIXED: Smart Bbox Selection",
        f"Bbox: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}px",
        f"Spread: {spread_x:.0f}x{spread_y:.0f}px"
    ]

    y_offset = 40
    for text in info_text:
        cv2.putText(vis, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 255), 3)
        cv2.putText(vis, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 0, 0), 1)
        y_offset += 35

    # Save
    video_stem = Path(path).stem
    output_path = output_dir / f"{video_stem}_FIXED.jpg"
    cv2.imwrite(str(output_path), vis)
    print(f"  Saved: {output_path}")

print()
print("="*80)
print("TEST COMPLETE")
print("="*80)
print()
print("Check visualizations in: test_output/fixed_detection/")
