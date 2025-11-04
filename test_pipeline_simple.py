#!/usr/bin/env python3
"""
Simple test of Python MTCNN + CLNF pipeline using a test image
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add pyfaceau to path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

print("="*80)
print("Python MTCNN + CLNF Pipeline Test (Simple)")
print("="*80)

# Step 1: Initialize MTCNN
print("\n[Step 1] Initializing MTCNN Detector")
print("-" * 80)

from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN

mtcnn = OpenFaceMTCNN()
print("✓ MTCNN detector initialized")
print(f"  Device: {mtcnn.device}")

# Step 2: Initialize CLNF
print("\n[Step 2] Initializing CLNF Detector")
print("-" * 80)

from pyfaceau.clnf.clnf_detector import CLNFDetector

model_dir = Path("S1 Face Mirror/weights/clnf")
clnf = CLNFDetector(model_dir=model_dir, max_iterations=5, convergence_threshold=0.01)
print("✓ CLNF detector initialized")

# Step 3: Create a synthetic test face
print("\n[Step 3] Creating Synthetic Test Face")
print("-" * 80)

# Create a simple test image with a face-like pattern
img_size = 640
test_image = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128

# Draw a simple face
face_center = (img_size // 2, img_size // 2)
face_radius = 150

# Face oval
cv2.ellipse(test_image, face_center, (face_radius, int(face_radius * 1.3)), 0, 0, 360, (200, 180, 160), -1)

# Eyes
eye_y = face_center[1] - 40
cv2.circle(test_image, (face_center[0] - 60, eye_y), 15, (50, 50, 50), -1)
cv2.circle(test_image, (face_center[0] + 60, eye_y), 15, (50, 50, 50), -1)

# Nose
nose_points = np.array([
    [face_center[0], face_center[1] - 10],
    [face_center[0] - 15, face_center[1] + 20],
    [face_center[0] + 15, face_center[1] + 20]
], dtype=np.int32)
cv2.fillPoly(test_image, [nose_points], (150, 120, 100))

# Mouth
cv2.ellipse(test_image, (face_center[0], face_center[1] + 60), (40, 20), 0, 0, 180, (80, 50, 50), -1)

print("✓ Created synthetic test face")
cv2.imwrite("/tmp/test_face_input.jpg", test_image)
print("  Saved: /tmp/test_face_input.jpg")

# Step 4: Test MTCNN detection
print("\n[Step 4] Testing MTCNN Detection")
print("-" * 80)

test_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

try:
    bboxes, landmarks = mtcnn.detect(test_rgb, return_landmarks=True)
    print(f"✓ MTCNN detection completed")
    print(f"  Detected {len(bboxes)} face(s)")

    if len(bboxes) == 0:
        print("  Note: No faces detected in synthetic image (this is expected)")
        print("  MTCNN is trained on real faces, not simple drawings")
        print("  Creating manual bbox for testing CLNF...")

        # Create manual bbox around the synthetic face
        bbox = np.array([
            face_center[0] - face_radius - 20,
            face_center[1] - int(face_radius * 1.3) - 20,
            face_center[0] + face_radius + 20,
            face_center[1] + int(face_radius * 1.3) + 20
        ], dtype=np.float32)
        bboxes = np.array([bbox])
        landmarks = None
        print(f"  Manual bbox: ({bbox[0]:.0f}, {bbox[1]:.0f}) -> ({bbox[2]:.0f}, {bbox[3]:.0f})")

except Exception as e:
    print(f"✗ MTCNN detection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test CLNF refinement
print("\n[Step 5] Testing CLNF Refinement")
print("-" * 80)

if len(bboxes) > 0:
    bbox = bboxes[0]
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Create simple 68-point initialization
    init_landmarks = np.zeros((68, 2), dtype=np.float32)

    # Jaw (0-16)
    for i in range(17):
        t = i / 16.0
        init_landmarks[i, 0] = x1 + t * w
        init_landmarks[i, 1] = y2 - h * 0.1 + np.sin(t * np.pi) * h * 0.1

    # Eyebrows (17-26)
    for i in range(17, 27):
        t = (i - 17) / 9.0
        init_landmarks[i, 0] = x1 + 0.2 * w + t * 0.6 * w
        init_landmarks[i, 1] = y1 + 0.3 * h

    # Nose (27-35)
    for i in range(27, 36):
        init_landmarks[i, 0] = cx
        init_landmarks[i, 1] = y1 + 0.4 * h + (i - 27) * 0.03 * h

    # Eyes (36-47)
    for i in range(36, 48):
        t = (i - 36) / 11.0
        init_landmarks[i, 0] = x1 + 0.25 * w + t * 0.5 * w
        init_landmarks[i, 1] = y1 + 0.4 * h

    # Mouth (48-67)
    for i in range(48, 68):
        t = (i - 48) / 19.0
        init_landmarks[i, 0] = x1 + 0.3 * w + t * 0.4 * w
        init_landmarks[i, 1] = y1 + 0.7 * h

    print("✓ Created 68-point initialization")

    # Convert to grayscale
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    try:
        print("  Running CLNF refinement...")
        refined_landmarks, converged, num_iters = clnf.refine_landmarks(
            gray, init_landmarks, scale_idx=2, regularization=0.5, multi_scale=False
        )

        print(f"✓ CLNF refinement completed")
        print(f"  Converged: {converged}")
        print(f"  Iterations: {num_iters}")

        # Save visualization
        vis = test_image.copy()

        # Draw bbox
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Draw initial landmarks (orange)
        for i, (x, y) in enumerate(init_landmarks):
            cv2.circle(vis, (int(x), int(y)), 2, (0, 165, 255), -1)

        # Draw refined landmarks (cyan)
        for i, (x, y) in enumerate(refined_landmarks):
            cv2.circle(vis, (int(x), int(y)), 3, (255, 255, 0), -1)

        cv2.putText(vis, "Green=BBox, Orange=Init, Cyan=CLNF", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imwrite("/tmp/test_clnf_output.jpg", vis)
        print("✓ Saved visualization: /tmp/test_clnf_output.jpg")

        clnf_works = True

    except Exception as e:
        print(f"✗ CLNF refinement failed: {e}")
        import traceback
        traceback.print_exc()
        clnf_works = False
else:
    clnf_works = False

# Final Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

print("✓ MTCNN initialized successfully")
print("✓ CLNF initialized successfully")
if clnf_works:
    print("✓ CLNF refinement executed successfully")
else:
    print("✗ CLNF refinement failed")

print("\n" + "="*80)
if clnf_works:
    print("SUCCESS: Pure Python landmark detection pathway is functional!")
    print("="*80)
    print("\nNote: MTCNN may not detect synthetic faces, but that's expected.")
    print("The important result is that both MTCNN and CLNF run without errors.")
    print("\nVisualizations:")
    print("  /tmp/test_face_input.jpg - Synthetic test face")
    print("  /tmp/test_clnf_output.jpg - CLNF refinement result")
else:
    print("ISSUES FOUND: CLNF refinement not working properly")
    print("="*80)

print()
