#!/usr/bin/env python3
"""
Production pipeline test: PyMTCNN detection + PyCLNF landmark fitting.

Tests the complete production pipeline with all performance optimizations enabled.
"""
import cv2
import numpy as np
import sys
from pathlib import Path

# Add pymtcnn to path
sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))
sys.path.insert(0, str(Path(__file__).parent / "pyclnf"))

from pymtcnn import MTCNN
from pyclnf.clnf import CLNF


def test_production_pipeline():
    """Test complete production pipeline."""

    print("="*80)
    print("PRODUCTION PIPELINE TEST")
    print("="*80)

    # Test image
    image_path = "calibration_frames/patient1_frame1.jpg"
    image = cv2.imread(image_path)

    print(f"\n1. Loading test image: {image_path}")
    print(f"   Image shape: {image.shape}")

    # Step 1: Face Detection with PyMTCNN
    print("\n2. Face Detection (PyMTCNN)")
    detector = MTCNN()
    bboxes, landmarks = detector.detect(image)

    if bboxes is None or len(bboxes) == 0:
        print("   ✗ No faces detected!")
        return

    # Use first detected face
    bbox = bboxes[0]
    pymtcnn_landmarks = landmarks[0] if landmarks is not None else None

    print(f"   ✓ Detected {len(bboxes)} face(s)")
    print(f"   BBox: x={bbox[0]:.1f}, y={bbox[1]:.1f}, w={bbox[2]:.1f}, h={bbox[3]:.1f}")
    if pymtcnn_landmarks is not None:
        print(f"   PyMTCNN 5-point landmarks available")

    # Step 2: Landmark Fitting with PyCLNF
    print("\n3. Landmark Fitting (PyCLNF)")
    print("   Configuration:")
    print("   - Regularization: 35")
    print("   - Max iterations: 10")
    print("   - Sigma: 1.5")
    print("   - Window sizes: [11, 9, 7] (auto-filtered for sigma)")

    clnf = CLNF(
        model_dir="pyclnf/models",
        regularization=35,
        max_iterations=10,
        convergence_threshold=0.005,
        sigma=1.5,
        weight_multiplier=0.0,
        window_sizes=None,  # Auto-select
        detector=None,  # Don't initialize detector (we already have bbox)
        debug_mode=False
    )

    # Convert bbox to (x, y, w, h) format if needed
    if len(bbox) == 4:
        pymtcnn_bbox = tuple(bbox[:4])
    else:
        pymtcnn_bbox = (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])

    # Fit landmarks
    landmarks_68, info = clnf.fit(image, face_bbox=pymtcnn_bbox)

    if landmarks_68 is None:
        print("   ✗ Landmark fitting failed!")
        return

    print(f"\n   ✓ Fitting completed:")
    print(f"   - Converged: {info.get('converged', 'N/A')}")
    print(f"   - Total iterations: {info.get('iterations', 'N/A')}")
    print(f"   - Final update: {info.get('final_update', 'N/A'):.6f}")

    # Step 3: Results
    print("\n4. Results")
    print("   Sample landmarks (PyCLNF 68-point):")
    for lm_idx in [36, 48, 30, 8]:
        x, y = landmarks_68[lm_idx]
        print(f"   - Landmark {lm_idx:2d}: ({x:7.2f}, {y:7.2f})")

    # Compare with C++ OpenFace reference
    print("\n5. Comparison with C++ OpenFace")
    cpp_reference = {
        36: (364.3000, 866.1000),
        48: (420.6000, 1053.5000),
        30: (483.8000, 944.3000),
        8: (503.0000, 1164.3000)
    }

    errors = []
    print("   Landmark-wise errors:")
    for lm_idx in [36, 48, 30, 8]:
        py_x, py_y = landmarks_68[lm_idx]
        cpp_x, cpp_y = cpp_reference[lm_idx]
        dx = py_x - cpp_x
        dy = py_y - cpp_y
        dist = np.sqrt(dx**2 + dy**2)
        errors.append(dist)
        print(f"   - Landmark {lm_idx:2d}: {dist:6.2f}px (dx={dx:6.2f}, dy={dy:6.2f})")

    mean_error = np.mean(errors)
    print(f"\n   Mean error: {mean_error:.2f}px")

    # Convergence status
    print(f"\n6. Convergence Status")
    if info.get('converged', False):
        print(f"   ✓ CONVERGED (update: {info.get('final_update', 0):.6f})")
    else:
        print(f"   ✗ NOT CONVERGED (update: {info.get('final_update', 0):.6f})")
        print(f"   Iterations used: {info.get('iterations', 'N/A')}/10")

    # Final summary
    print("\n" + "="*80)
    print("PRODUCTION PIPELINE SUMMARY")
    print("="*80)
    print(f"Detection: PyMTCNN ✓")
    print(f"Landmark fitting: PyCLNF ✓")
    print(f"Mean error vs C++: {mean_error:.2f}px")
    print(f"Convergence: {'✓ YES' if info.get('converged', False) else '✗ NO'}")

    # Status classification
    if mean_error < 2.0:
        status = "EXCELLENT"
    elif mean_error < 5.0:
        status = "GOOD"
    elif mean_error < 10.0:
        status = "ACCEPTABLE"
    else:
        status = "NEEDS IMPROVEMENT"

    print(f"Status: {status}")
    print("="*80)

    return {
        'mean_error': mean_error,
        'converged': info.get('converged', False),
        'iterations': info.get('iterations', 0),
        'landmarks': landmarks_68
    }


if __name__ == "__main__":
    test_production_pipeline()
