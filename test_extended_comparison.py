#!/usr/bin/env python3
"""
Extended comparison test with additional normal cohort videos.
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
    ("IMG_8401_source", "Paralysis - Surgical Markings", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV"),
    ("IMG_9330_source", "Paralysis - Severe", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV"),
    ("IMG_0434_source", "Normal Cohort", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0434_source.MOV"),
    ("IMG_0437_source", "Normal Cohort", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0437_source.MOV"),
    ("IMG_0441_source", "Normal Cohort", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0441_source.MOV"),
    ("IMG_0942_source", "Normal Cohort", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0942_source.MOV"),
]

output_dir = Path(__file__).parent / "test_output" / "extended_comparison"
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EXTENDED COMPARISON TEST - 6 VIDEOS")
print("="*80)
print()

# Initialize detector with debug mode
detector = PyFaceAU68LandmarkDetector(
    debug_mode=True,
    use_clnf_refinement=True,
    skip_redetection=False
)

results = []

for name, category, path in test_videos:
    print()
    print("="*80)
    print(f"{name} ({category})")
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
        results.append({
            'name': name,
            'category': category,
            'status': 'FAILED',
            'bbox': None,
            'spread': None
        })
        detector.reset_tracking_history()
        continue

    # Save bbox before reset
    bbox = detector.cached_bbox.copy() if detector.cached_bbox is not None else None

    # Reset for next video
    detector.reset_tracking_history()

    # Analyze results
    x_min, x_max = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    y_min, y_max = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
    spread_x = x_max - x_min
    spread_y = y_max - y_min

    print(f"\nResults:")
    print(f"  Bbox: {bbox}")
    print(f"  Bbox size: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}")
    print(f"  Landmark spread: {spread_x:.0f}x{spread_y:.0f} pixels")

    # Check quality
    expected_spread = min(frame.shape[:2]) * 0.3
    actual_spread = (spread_x + spread_y) / 2

    if actual_spread > expected_spread:
        status = "GOOD"
        print(f"  ✅ GOOD: Landmarks well-distributed ({actual_spread:.0f}px spread)")
    else:
        status = "POOR"
        print(f"  ❌ POOR: Landmarks clustered ({actual_spread:.0f}px spread, expected >{expected_spread:.0f}px)")

    results.append({
        'name': name,
        'category': category,
        'status': status,
        'bbox': bbox,
        'bbox_size': f"{bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}",
        'spread': f"{spread_x:.0f}x{spread_y:.0f}"
    })

    # Create visualization
    vis = frame.copy()

    # Draw bbox
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = (0, 255, 0) if status == "GOOD" else (0, 0, 255)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 4)

    # Draw landmarks with colors
    for i, (x, y) in enumerate(landmarks):
        if i < 17:  # Jaw
            dot_color = (0, 255, 0)
        elif i < 27:  # Eyebrows
            dot_color = (255, 0, 0)
        elif i < 36:  # Nose
            dot_color = (0, 255, 255)
        elif i < 48:  # Eyes
            dot_color = (255, 0, 255)
        else:  # Mouth
            dot_color = (0, 128, 255)

        cv2.circle(vis, (int(x), int(y)), 5, dot_color, -1)

    # Midline
    glabella, chin = detector.get_facial_midline(landmarks)
    if glabella is not None and chin is not None:
        cv2.line(vis, (int(glabella[0]), int(glabella[1])),
                (int(chin[0]), int(chin[1])), (0, 0, 255), 3)

    # Info overlay
    info_text = [
        f"{name}",
        f"{category}",
        f"Status: {status}",
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
    output_path = output_dir / f"{name}.jpg"
    cv2.imwrite(str(output_path), vis)
    print(f"  Saved: {output_path}")

# Summary table
print()
print("="*80)
print("SUMMARY TABLE")
print("="*80)
print()
print(f"{'Video':<20} {'Category':<25} {'Status':<8} {'Bbox Size':<12} {'Spread':<12}")
print("-"*80)

for r in results:
    bbox_str = r.get('bbox_size', 'N/A')
    spread_str = r.get('spread', 'N/A')
    print(f"{r['name']:<20} {r['category']:<25} {r['status']:<8} {bbox_str:<12} {spread_str:<12}")

print()
print("="*80)

# Count results
good_count = sum(1 for r in results if r['status'] == 'GOOD')
poor_count = sum(1 for r in results if r['status'] == 'POOR')
failed_count = sum(1 for r in results if r['status'] == 'FAILED')

print(f"\nResults: {good_count} GOOD | {poor_count} POOR | {failed_count} FAILED (out of {len(results)} total)")
print()
print(f"Visualizations saved to: {output_dir}")
print()
