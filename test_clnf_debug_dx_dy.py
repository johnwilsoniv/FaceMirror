#!/usr/bin/env python3
"""
Test Python CLNF with detailed debug output to investigate dx/dy divergence.
"""

import sys
import cv2
import numpy as np

# Add pyclnf and pymtcnn to path
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pymtcnn')

from pyclnf import CLNF

# Load test image
image_path = 'calibration_frames/patient1_frame1.jpg'
print(f"Loading image: {image_path}")
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"ERROR: Could not load image from {image_path}")
    sys.exit(1)

print(f"Image shape: {image.shape}")

# Initialize CLNF with debug mode
print("\nInitializing CLNF with debug mode...")
clnf = CLNF(
    model_dir='pyclnf/models',
    detector=None,  # No automatic detection, use manual bbox
    debug_mode=True,  # Enable debug mode for detailed output
    window_sizes=[11]  # Only run ws=11 for focused debugging
)

# Use bbox from OpenFace detection (this matches the C++ baseline)
# Based on the landmarks in the CSV, the face is centered around x~480, y~950
# with a width/height of about 370px
bbox = [310, 780, 370, 370]  # [x, y, width, height]

# Process the image
print("\n" + "="*80)
print("PROCESSING IMAGE WITH DEBUG MODE")
print("="*80)

landmarks, info = clnf.fit(image, face_bbox=bbox)

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

print(f"Detected {len(landmarks)} landmarks")
print(f"Converged: {info['converged']}, Iterations: {info['iterations']}")

# Print specific landmarks we're tracking
tracked = [36, 48, 30, 8]
print("\nTracked landmarks:")
for lm_idx in tracked:
    if lm_idx < len(landmarks):
        x, y = landmarks[lm_idx]
        print(f"  Landmark_{lm_idx}: ({x:.4f}, {y:.4f})")

# Save landmarks for comparison
np.save('/tmp/python_landmarks_debug.npy', landmarks)
print(f"\nSaved landmarks to /tmp/python_landmarks_debug.npy")

print("\nDone!")
