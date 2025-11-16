"""
Test CLNF with debug mode to compare against C++ OpenFace output
Uses the EXACT bbox from C++ MTCNN detection for apples-to-apples comparison
"""
import cv2
import numpy as np
from pyclnf.clnf import CLNF

# Test image
image_path = "calibration_frames/patient1_frame1.jpg"

# EXACT bbox from C++ MTCNN detection: DEBUG_BBOX: 301.938,782.149,400.586,400.585
cpp_bbox = (301.938, 782.149, 400.586, 400.585)  # (x, y, width, height)

# Initialize CLNF with debug mode (no detector needed - using manual bbox)
clnf = CLNF(
    model_dir="pyclnf/models",
    regularization=35,
    max_iterations=10,
    convergence_threshold=0.005,
    sigma=1.5,
    weight_multiplier=0.0,
    window_sizes=[11, 9, 7],
    detector=None,
    debug_mode=True,
    tracked_landmarks=[36, 48, 30, 8]
)

# Load image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image {image_path}")
    exit(1)

print(f"Testing with image: {image_path}")
print(f"Image shape: {image.shape}")
print(f"Using C++ bbox: {cpp_bbox}")

# Fit model with manual bbox
landmarks = clnf.fit(image, face_bbox=cpp_bbox)

print(f"\n[PY][FINAL] Final landmarks (tracked):")
for lm_idx in [36, 48, 30, 8]:
    if lm_idx < len(landmarks):
        print(f"[PY][FINAL]   Landmark_{lm_idx}: ({landmarks[lm_idx][0]:.4f}, {landmarks[lm_idx][1]:.4f})")
