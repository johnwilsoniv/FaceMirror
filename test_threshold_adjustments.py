#!/usr/bin/env python3
"""
Test different threshold and NMS configurations to match C++ behavior.
"""

import cv2
import numpy as np
from cpp_mtcnn_detector import CPPMTCNNDetector

def test_with_thresholds(img_path, pnet_thresh, rnet_thresh, onet_thresh, nms_thresh):
    """
    Test detection with custom thresholds.
    """
    img = cv2.imread(img_path)

    detector = CPPMTCNNDetector()

    # Override thresholds
    detector.thresholds = [pnet_thresh, rnet_thresh, onet_thresh]

    print(f"\n{'='*80}")
    print(f"Testing with thresholds: PNet={pnet_thresh}, RNet={rnet_thresh}, ONet={onet_thresh}")
    print(f"{'='*80}")

    # Detect (with debug output suppressed for readability)
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()

    bboxes, landmarks = detector.detect(img)

    sys.stdout = old_stdout

    print(f"\nResults:")
    print(f"  Detected {len(bboxes)} faces")

    if len(bboxes) > 0:
        # Show top 3 by width
        widths = [bbox[2] for bbox in bboxes]
        sorted_indices = np.argsort(widths)[::-1][:3]

        for rank, idx in enumerate(sorted_indices):
            bbox = bboxes[idx]
            print(f"    Face {rank+1}: x={bbox[0]:.1f}, y={bbox[1]:.1f}, w={bbox[2]:.1f}, h={bbox[3]:.1f}")

    return len(bboxes)

def test_with_spatial_priority(img_path):
    """
    Test detection with spatial priority for upper regions.
    """
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    detector = CPPMTCNNDetector()

    print(f"\n{'='*80}")
    print(f"Testing with SPATIAL PRIORITY (prefer upper 70% of image)")
    print(f"{'='*80}")

    bboxes, landmarks = detector.detect(img)

    print(f"\nAll detections: {len(bboxes)}")

    # Apply spatial priority: boost score for boxes in upper 70%, penalize bottom 30%
    adjusted_bboxes = []
    for bbox in bboxes:
        x, y, w_bbox, h_bbox = bbox

        # Calculate center Y
        center_y = y + h_bbox / 2

        # Upper 70%: no change
        # Bottom 30%: penalize score
        if center_y > 0.7 * h:
            # Skip artifact boxes
            print(f"  Filtering artifact: y={y:.0f} (bottom 30%)")
            continue
        else:
            adjusted_bboxes.append(bbox)

    print(f"\nAfter spatial filtering: {len(adjusted_bboxes)} faces")

    if len(adjusted_bboxes) > 0:
        # Show top 3 by width
        widths = [bbox[2] for bbox in adjusted_bboxes]
        sorted_indices = np.argsort(widths)[::-1][:3]

        for rank, idx in enumerate(sorted_indices):
            bbox = adjusted_bboxes[idx]
            print(f"    Face {rank+1}: x={bbox[0]:.1f}, y={bbox[1]:.1f}, w={bbox[2]:.1f}, h={bbox[3]:.1f}")

    return len(adjusted_bboxes)

if __name__ == "__main__":
    test_image = "calibration_frames/patient1_frame1.jpg"

    print(f"\nTesting threshold variations on: {test_image}")
    print(f"{'='*80}")

    # Current thresholds (C++ defaults)
    print(f"\n1. Current thresholds (0.6, 0.7, 0.7):")
    test_with_thresholds(test_image, 0.6, 0.7, 0.7, 0.7)

    # Lower PNet threshold (more lenient, might get more boxes like C++)
    print(f"\n2. Lower PNet threshold (0.5, 0.7, 0.7):")
    test_with_thresholds(test_image, 0.5, 0.7, 0.7, 0.7)

    # Lower PNet and RNet thresholds
    print(f"\n3. Lower PNet and RNet (0.5, 0.6, 0.7):")
    test_with_thresholds(test_image, 0.5, 0.6, 0.7, 0.7)

    # Much more lenient (to see max detections)
    print(f"\n4. Very lenient (0.4, 0.5, 0.6):")
    test_with_thresholds(test_image, 0.4, 0.5, 0.6, 0.7)

    # Spatial priority filter
    print(f"\n5. Spatial priority filter (remove bottom 30%):")
    test_with_spatial_priority(test_image)

    print(f"\n{'='*80}")
    print(f"RECOMMENDATION:")
    print(f"{'='*80}")
    print(f"If lowering thresholds doesn't help, use spatial filtering.")
    print(f"C++ likely gets 95 boxes by being more lenient at early stages.")
