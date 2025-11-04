#!/usr/bin/env python3
"""
Test C++ OpenFace (dlib-removed version) with bbox visualization.
"""

import subprocess
import tempfile
import numpy as np
import cv2
from pathlib import Path

print("="*80)
print("C++ OPENFACE (DLIB-REMOVED) TEST")
print("="*80)

# Configuration
OPENFACE_BINARY = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
TEST_IMAGE = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/frames/IMG_8401.jpg"
OUTPUT_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results_cpp")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nTest image: {TEST_IMAGE}")
print(f"OpenFace binary: {OPENFACE_BINARY}")

# Load test image
test_image = cv2.imread(TEST_IMAGE)
h, w = test_image.shape[:2]
print(f"Image size: {w}x{h}")

# Run C++ OpenFace
print(f"\n{'='*80}")
print("Running C++ OpenFace...")
print(f"{'='*80}")

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    cmd = [
        OPENFACE_BINARY,
        "-f", str(TEST_IMAGE),
        "-out_dir", str(tmpdir),
        "-2Dfp",  # Output 2D landmarks
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: OpenFace failed with return code {result.returncode}")
        print(f"stderr: {result.stderr}")
        exit(1)

    print("✓ C++ OpenFace completed successfully")

    # Parse CSV output
    csv_file = tmpdir / f"{Path(TEST_IMAGE).stem}.csv"

    if not csv_file.exists():
        print(f"ERROR: CSV not found: {csv_file}")
        exit(1)

    print(f"\nParsing CSV: {csv_file}")

    with open(csv_file, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split(',')
        values = lines[1].strip().split(',')

    # Extract landmarks
    landmarks = []
    for i in range(68):
        try:
            x_idx = header.index(f'x_{i}')
            y_idx = header.index(f'y_{i}')
        except ValueError:
            x_idx = header.index(f' x_{i}')
            y_idx = header.index(f' y_{i}')

        x = float(values[x_idx])
        y = float(values[y_idx])
        landmarks.append([x, y])

    landmarks = np.array(landmarks)
    print(f"✓ Extracted {len(landmarks)} landmarks")

    # Get confidence
    try:
        conf_idx = header.index('confidence')
    except ValueError:
        conf_idx = header.index(' confidence')
    confidence = float(values[conf_idx])
    print(f"  Confidence: {confidence:.3f}")

    # Estimate bbox from landmarks
    x_min, y_min = landmarks.min(axis=0)
    x_max, y_max = landmarks.max(axis=0)

    # Add margin (10%)
    margin = 0.1
    w_bbox = x_max - x_min
    h_bbox = y_max - y_min
    x_min -= w_bbox * margin
    y_min -= h_bbox * margin
    x_max += w_bbox * margin
    y_max += h_bbox * margin

    bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
    print(f"  Estimated bbox: {bbox}")

# Create visualization
print(f"\n{'='*80}")
print("Creating visualization...")
print(f"{'='*80}")

vis = test_image.copy()

# Draw bbox
cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)

# Draw landmarks
for i, (x, y) in enumerate(landmarks):
    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
    # Label key landmarks
    if i in [0, 8, 16, 27, 30, 33, 36, 45, 48, 54]:
        cv2.putText(vis, str(i), (int(x)+5, int(y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

# Add title and info
cv2.putText(vis, "C++ OpenFace (dlib-removed)", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
cv2.putText(vis, f"Confidence: {confidence:.3f}", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.putText(vis, f"Bbox: {bbox}", (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Save visualization
vis_file = OUTPUT_DIR / "cpp_openface_with_bbox.jpg"
cv2.imwrite(str(vis_file), vis)
print(f"\n✓ Saved visualization: {vis_file}")

# Save landmarks for later comparison
np.save(OUTPUT_DIR / "cpp_landmarks.npy", landmarks)
print(f"✓ Saved landmarks: {OUTPUT_DIR / 'cpp_landmarks.npy'}")

print(f"\n{'='*80}")
print("C++ OPENFACE TEST COMPLETE")
print(f"{'='*80}")
print(f"\nResults:")
print(f"  Landmarks: {len(landmarks)} points")
print(f"  Confidence: {confidence:.3f}")
print(f"  Bbox: {bbox}")
print(f"\n  Status: ✓ SUCCESS")
