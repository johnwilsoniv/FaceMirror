#!/usr/bin/env python3
"""
Test both landmark detection approaches with bbox visualization:
1. C++ OpenFace (dlib-removed) - via subprocess, parse output directly
2. Python MTCNN + CLNF - pure Python implementation

NO CSV dependencies - extract landmarks directly from outputs
"""

import sys
import os
import subprocess
import tempfile
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau')

print("="*80)
print("LANDMARK DETECTION COMPARISON - BOTH APPROACHES")
print("="*80)

# Configuration
OPENFACE_BINARY = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
MODEL_DIR = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model"
TEST_IMAGE = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/frames/IMG_8401.jpg"
OUTPUT_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results_fixed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nTest image: {TEST_IMAGE}")
print(f"OpenFace binary: {OPENFACE_BINARY}")
print(f"Output directory: {OUTPUT_DIR}")

# Load test image
test_image = cv2.imread(TEST_IMAGE)
h, w = test_image.shape[:2]
print(f"Image size: {w}x{h}")


# ============================================================================
# APPROACH 1: C++ OpenFace (dlib-removed version)
# ============================================================================

def test_cpp_openface(image_path):
    """
    Test C++ OpenFace binary and extract landmarks from CSV.

    Returns:
        landmarks: (68, 2) numpy array
        bbox: [x1, y1, x2, y2] estimated from landmarks
        confidence: float
    """
    print(f"\n{'='*80}")
    print("APPROACH 1: C++ OpenFace (dlib-removed)")
    print(f"{'='*80}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Run OpenFace
        print("Running C++ OpenFace...")
        cmd = [
            OPENFACE_BINARY,
            "-f", str(image_path),
            "-out_dir", str(tmpdir),
            "-2Dfp",  # Output 2D landmarks
            "-q"      # Quiet mode (less output)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"ERROR: OpenFace failed")
            print(f"stderr: {result.stderr}")
            return None, None, None

        print("✓ C++ OpenFace completed")

        # Parse CSV output (temporary - we'll create non-CSV version next)
        csv_file = tmpdir / f"{Path(image_path).stem}.csv"

        if not csv_file.exists():
            print(f"ERROR: CSV not found: {csv_file}")
            return None, None, None

        # Read CSV
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

        # Get confidence
        try:
            conf_idx = header.index('confidence')
        except ValueError:
            conf_idx = header.index(' confidence')
        confidence = float(values[conf_idx])

        print(f"✓ Extracted {len(landmarks)} landmarks")
        print(f"  Confidence: {confidence:.3f}")

        # Estimate bbox from landmarks
        x_min, y_min = landmarks.min(axis=0)
        x_max, y_max = landmarks.max(axis=0)

        # Add margin
        margin = 0.1
        w_bbox = x_max - x_min
        h_bbox = y_max - y_min
        x_min -= w_bbox * margin
        y_min -= h_bbox * margin
        x_max += w_bbox * margin
        y_max += h_bbox * margin

        bbox = [x_min, y_min, x_max, y_max]

        print(f"  Estimated bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")

        return landmarks, bbox, confidence


# ============================================================================
# APPROACH 2: Python MTCNN + CLNF
# ============================================================================

def test_python_pipeline(image_path):
    """
    Test Python MTCNN + CLNF pipeline.

    Returns:
        landmarks: (68, 2) numpy array
        bbox: [x1, y1, x2, y2, confidence] from MTCNN
        mtcnn_landmarks_5pt: (5, 2) numpy array
        initial_68pt: (68, 2) numpy array from bbox initialization
    """
    print(f"\n{'='*80}")
    print("APPROACH 2: Python MTCNN + CLNF")
    print(f"{'='*80}")

    # Load image
    image = cv2.imread(str(image_path))

    # Step 1: MTCNN face detection
    print("\n[1/3] Running MTCNN face detection...")

    try:
        from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN

        mtcnn = OpenFaceMTCNN(device='cpu')
        print("  ✓ MTCNN initialized")

        bboxes, landmarks_5pt = mtcnn.detect(image)

        if len(bboxes) == 0:
            print("  ✗ No faces detected")
            return None, None, None, None

        bbox = bboxes[0]  # First face
        landmarks_5 = landmarks_5pt[0] if len(landmarks_5pt) > 0 else None

        print(f"  ✓ Face detected")
        print(f"    Bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        print(f"    Confidence: {bbox[4]:.3f}")

    except Exception as e:
        print(f"  ✗ MTCNN failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    # Step 2: Initialize 68 landmarks from bbox
    print("\n[2/3] Initializing 68-point landmarks from bbox...")

    def bbox_to_68_landmarks(bbox):
        """Simple bbox to 68-point initialization."""
        x1, y1, x2, y2 = bbox[:4]
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Template coordinates (normalized -0.5 to 0.5)
        template = np.array([
            # Jaw (0-16)
            [-0.3, 0.3], [-0.28, 0.35], [-0.25, 0.4], [-0.2, 0.45], [-0.15, 0.48],
            [-0.1, 0.5], [-0.05, 0.51], [0, 0.52], [0.05, 0.51], [0.1, 0.5],
            [0.15, 0.48], [0.2, 0.45], [0.25, 0.4], [0.28, 0.35], [0.3, 0.3],
            [0.32, 0.25], [0.33, 0.2],
            # Eyebrows (17-26)
            [-0.25, -0.1], [-0.2, -0.12], [-0.15, -0.13], [-0.1, -0.12], [-0.05, -0.1],
            [0.05, -0.1], [0.1, -0.12], [0.15, -0.13], [0.2, -0.12], [0.25, -0.1],
            # Nose (27-35)
            [0, -0.05], [0, 0.0], [0, 0.05], [0, 0.1],
            [-0.08, 0.12], [-0.04, 0.13], [0, 0.14], [0.04, 0.13], [0.08, 0.12],
            # Eyes (36-47)
            [-0.2, -0.05], [-0.17, -0.07], [-0.14, -0.07], [-0.11, -0.05],
            [-0.14, -0.04], [-0.17, -0.04],
            [0.11, -0.05], [0.14, -0.07], [0.17, -0.07], [0.2, -0.05],
            [0.17, -0.04], [0.14, -0.04],
            # Mouth (48-67)
            [-0.12, 0.25], [-0.09, 0.27], [-0.05, 0.28], [0, 0.285], [0.05, 0.28],
            [0.09, 0.27], [0.12, 0.25], [0.1, 0.28], [0.05, 0.3], [0, 0.305],
            [-0.05, 0.3], [-0.1, 0.28], [-0.08, 0.26], [-0.05, 0.27], [0, 0.275],
            [0.05, 0.27], [0.08, 0.26], [0.05, 0.27], [0, 0.275], [-0.05, 0.27],
        ])

        # Scale and translate
        landmarks = template.copy()
        landmarks[:, 0] = cx + landmarks[:, 0] * w
        landmarks[:, 1] = cy + landmarks[:, 1] * h

        return landmarks

    initial_68pt = bbox_to_68_landmarks(bbox)
    print(f"  ✓ Initialized 68 landmarks")

    # Step 3: CLNF refinement
    print("\n[3/3] Running CLNF refinement...")

    try:
        from pyfaceau.clnf.clnf_detector import CLNFDetector

        clnf = CLNFDetector(
            model_dir=MODEL_DIR,
            max_iterations=10,
            convergence_threshold=0.01
        )
        print("  ✓ CLNF detector initialized")

        refined_landmarks, converged, num_iterations = clnf.refine_landmarks(
            image,
            initial_68pt,
            scale_idx=2,
            regularization=0.5,
            multi_scale=True
        )

        print(f"  ✓ CLNF refinement completed")
        print(f"    Converged: {converged}")
        print(f"    Iterations: {num_iterations}")

        return refined_landmarks, bbox, landmarks_5, initial_68pt

    except Exception as e:
        print(f"  ✗ CLNF failed: {e}")
        import traceback
        traceback.print_exc()
        return None, bbox, landmarks_5, initial_68pt


# ============================================================================
# Run both approaches
# ============================================================================

cpp_landmarks, cpp_bbox, cpp_confidence = test_cpp_openface(TEST_IMAGE)
python_landmarks, python_bbox, python_5pt, python_initial = test_python_pipeline(TEST_IMAGE)


# ============================================================================
# Create visualizations with bboxes
# ============================================================================

print(f"\n{'='*80}")
print("Creating visualizations with bboxes...")
print(f"{'='*80}")

# Create comparison visualization
vis = np.zeros((h, w * 3, 3), dtype=np.uint8)

# Column 1: C++ OpenFace
if cpp_landmarks is not None:
    vis[:, :w] = test_image.copy()

    # Draw bbox
    if cpp_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in cpp_bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw landmarks
    for i, (x, y) in enumerate(cpp_landmarks):
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

    cv2.putText(vis, "C++ OpenFace", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(vis, f"Conf: {cpp_confidence:.3f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Column 2: Python MTCNN bbox + initial landmarks
if python_bbox is not None:
    vis[:, w:2*w] = test_image.copy()

    # Draw MTCNN bbox
    x1, y1, x2, y2 = [int(v) for v in python_bbox[:4]]
    cv2.rectangle(vis, (w + x1, y1), (w + x2, y2), (255, 128, 0), 2)

    # Draw 5-point landmarks if available
    if python_5pt is not None:
        for x, y in python_5pt:
            cv2.circle(vis, (int(w + x), int(y)), 3, (255, 255, 0), -1)

    # Draw initial 68-point landmarks
    if python_initial is not None:
        for x, y in python_initial:
            cv2.circle(vis, (int(w + x), int(y)), 2, (255, 128, 0), -1)

    cv2.putText(vis, "Python MTCNN Init", (w + 10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 128, 0), 2)
    cv2.putText(vis, f"Conf: {python_bbox[4]:.3f}", (w + 10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)

# Column 3: Python CLNF refined
if python_landmarks is not None:
    vis[:, 2*w:] = test_image.copy()

    # Draw bbox (same as MTCNN)
    x1, y1, x2, y2 = [int(v) for v in python_bbox[:4]]
    cv2.rectangle(vis, (2*w + x1, y1), (2*w + x2, y2), (0, 0, 255), 2)

    # Draw refined landmarks
    for x, y in python_landmarks:
        cv2.circle(vis, (int(2*w + x), int(y)), 2, (0, 0, 255), -1)

    cv2.putText(vis, "Python CLNF Refined", (2*w + 10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

# Save visualization
vis_file = OUTPUT_DIR / "comparison_with_bboxes.jpg"
cv2.imwrite(str(vis_file), vis)
print(f"\n✓ Saved comparison: {vis_file}")

# Calculate metrics if both succeeded
if cpp_landmarks is not None and python_landmarks is not None:
    errors = np.linalg.norm(python_landmarks - cpp_landmarks, axis=1)
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)

    print(f"\n{'='*80}")
    print("ACCURACY METRICS")
    print(f"{'='*80}")
    print(f"\nPython CLNF vs C++ OpenFace:")
    print(f"  Mean error:   {mean_error:.2f} pixels")
    print(f"  Median error: {median_error:.2f} pixels")
    print(f"  Max error:    {max_error:.2f} pixels")

    if mean_error < 10:
        print(f"\n  ✓ ACCEPTABLE (< 10px)")
    else:
        print(f"\n  ✗ POOR (> 10px) - Bug in Python CLNF")

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}")
