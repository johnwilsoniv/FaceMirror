#!/usr/bin/env python3
"""
Test pure Python CNN MTCNN detector and compare with C++ MTCNN.
"""

import cv2
import numpy as np
from pure_python_mtcnn_detector import PurePythonMTCNNDetector
from pathlib import Path

def test_pure_python_mtcnn():
    """Test the pure Python CNN MTCNN detector."""
    print("="*80)
    print("TESTING PURE PYTHON CNN MTCNN DETECTOR")
    print("="*80)

    # Load test image
    test_image = Path("calibration_frames/patient1_frame1.jpg")

    if not test_image.exists():
        print(f"Test image not found: {test_image}")
        return

    print(f"\nLoading image: {test_image}")
    img = cv2.imread(str(test_image))
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # Initialize detector
    print("\nInitializing Pure Python CNN MTCNN detector...")
    detector = PurePythonMTCNNDetector()

    # Run detection
    print("\nRunning detection...")
    bboxes, landmarks = detector.detect(img, debug=True)

    print(f"\n{'='*80}")
    print(f"DETECTION RESULTS")
    print(f"{'='*80}")
    print(f"Detected {len(bboxes)} face(s)")

    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        print(f"\nFace {i+1}:")
        print(f"  Position: x={x:.1f}, y={y:.1f}")
        print(f"  Size: w={w:.1f}, h={h:.1f}")
        print(f"  Scale: {np.sqrt(w*h):.1f}")

    # Visualize
    img_vis = img.copy()
    for bbox, lms in zip(bboxes, landmarks):
        x, y, w, h = bbox.astype(int)
        cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)

        for lm in lms:
            cv2.circle(img_vis, tuple(lm.astype(int)), 3, (0, 0, 255), -1)

    output_path = "pure_python_mtcnn_test.jpg"
    cv2.imwrite(output_path, img_vis)
    print(f"\n✓ Saved visualization: {output_path}")

    return bboxes, landmarks


if __name__ == "__main__":
    test_pure_python_mtcnn()
