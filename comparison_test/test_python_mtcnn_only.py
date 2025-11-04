#!/usr/bin/env python3
"""
Test Python MTCNN only (isolate segfault issue).
"""

import sys
from pathlib import Path

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau')

print("="*80)
print("PYTHON MTCNN ONLY TEST")
print("="*80)

TEST_IMAGE = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/frames/IMG_8401.jpg"
OUTPUT_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results_python")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nTest image: {TEST_IMAGE}")

# Import cv2 carefully
print("\n[1/5] Importing cv2...")
try:
    import cv2
    print("  ✓ cv2 imported")
except Exception as e:
    print(f"  ✗ cv2 import failed: {e}")
    exit(1)

# Load image
print("\n[2/5] Loading image...")
try:
    image = cv2.imread(TEST_IMAGE)
    if image is None:
        print("  ✗ Failed to load image")
        exit(1)
    h, w = image.shape[:2]
    print(f"  ✓ Image loaded: {w}x{h}")
except Exception as e:
    print(f"  ✗ Image loading failed: {e}")
    exit(1)

# Import MTCNN
print("\n[3/5] Importing MTCNN...")
try:
    from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN
    print("  ✓ MTCNN imported")
except Exception as e:
    print(f"  ✗ MTCNN import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Initialize MTCNN
print("\n[4/5] Initializing MTCNN...")
try:
    mtcnn = OpenFaceMTCNN(device='cpu')
    print("  ✓ MTCNN initialized")
except Exception as e:
    print(f"  ✗ MTCNN initialization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Run MTCNN detection
print("\n[5/5] Running MTCNN detection...")
print("  This is where segfault likely occurs...")
try:
    bboxes, landmarks_5pt = mtcnn.detect(image)
    print(f"  ✓ MTCNN detection completed!")
    print(f"    Detected {len(bboxes)} face(s)")

    if len(bboxes) > 0:
        bbox = bboxes[0]
        print(f"    Bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        print(f"    Confidence: {bbox[4]:.3f}")

        # Save bbox info
        import numpy as np
        np.savez(OUTPUT_DIR / "mtcnn_results.npz",
                 bboxes=bboxes,
                 landmarks_5pt=landmarks_5pt)
        print(f"    ✓ Saved results: {OUTPUT_DIR / 'mtcnn_results.npz'}")

        # Visualize
        vis = image.copy()
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

        if len(landmarks_5pt) > 0:
            for x, y in landmarks_5pt[0]:
                cv2.circle(vis, (int(x), int(y)), 4, (255, 255, 0), -1)

        cv2.putText(vis, "Python MTCNN", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(vis, f"Confidence: {bbox[4]:.3f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        vis_file = OUTPUT_DIR / "mtcnn_detection.jpg"
        cv2.imwrite(str(vis_file), vis)
        print(f"    ✓ Saved visualization: {vis_file}")

except Exception as e:
    print(f"  ✗ MTCNN detection failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n{'='*80}")
print("MTCNN TEST COMPLETE - NO SEGFAULT!")
print(f"{'='*80}")
