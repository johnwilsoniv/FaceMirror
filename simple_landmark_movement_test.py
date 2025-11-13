#!/usr/bin/env python3
"""
Simple test to observe if landmarks actually move iteration-by-iteration.

Key question: When we compute parameter updates and apply them,
do the landmarks actually move toward where the response maps say they should?
"""

import cv2
import numpy as np
from pyclnf import CLNF

# Load test frame
video_path = 'Patient Data/Normal Cohort/IMG_0433.MOV'
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
face_bbox = (241, 555, 532, 532)

print("=" * 80)
print("LANDMARK MOVEMENT TEST")
print("=" * 80)
print()

# Track a few key landmarks
track_landmarks = [48, 54, 33]  # Mouth corners and nose tip
print(f"Tracking landmarks: {track_landmarks}")
print()

# Test with 5 iterations
clnf = CLNF(model_dir='pyclnf/models', max_iterations=5)

# Get initial landmarks
initial_params = clnf.pdm.init_params(face_bbox)
initial_landmarks = clnf.pdm.params_to_landmarks_2d(initial_params)

print("Initial landmarks:")
for lm_idx in track_landmarks:
    x, y = initial_landmarks[lm_idx]
    print(f"  Landmark {lm_idx}: ({x:.2f}, {y:.2f})")
print()

# Run optimization
landmarks_final, info = clnf.fit(gray, face_bbox, return_params=True)

print("Final landmarks (after 5 iterations):")
for lm_idx in track_landmarks:
    x, y = landmarks_final[lm_idx]
    print(f"  Landmark {lm_idx}: ({x:.2f}, {y:.2f})")
print()

print("Landmark movement:")
for lm_idx in track_landmarks:
    dx = landmarks_final[lm_idx, 0] - initial_landmarks[lm_idx, 0]
    dy = landmarks_final[lm_idx, 1] - initial_landmarks[lm_idx, 1]
    dist = np.sqrt(dx**2 + dy**2)
    print(f"  Landmark {lm_idx}: ({dx:+.2f}, {dy:+.2f}) - {dist:.2f}px")
print()

# Calculate total movement for all landmarks
all_movement = landmarks_final - initial_landmarks
all_distances = np.linalg.norm(all_movement, axis=1)

print("Overall landmark movement:")
print(f"  Mean distance: {all_distances.mean():.2f}px")
print(f"  Median distance: {np.median(all_distances):.2f}px")
print(f"  Max distance: {all_distances.max():.2f}px")
print(f"  Min distance: {all_distances.min():.2f}px")
print()

print("=" * 80)
print("INTERPRETATION:")
print("=" * 80)

if all_distances.mean() < 1.0:
    print("❌ PROBLEM: Landmarks barely moving (<1px average)")
    print("   Parameter updates are NOT translating to landmark motion")
elif all_distances.mean() < 10.0:
    print("⚠️  Landmarks moving slowly (~1-10px average)")
    print("   Updates may be too conservative or in wrong direction")
else:
    print("✓ Landmarks are moving (>10px average)")
    print("  But check if they're converging or diverging!")

print()
print(f"Convergence status: {info['converged']}")
print(f"Final update magnitude: {info['final_update']:.6f}")
print()
print("=" * 80)
