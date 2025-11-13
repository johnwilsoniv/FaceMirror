#!/usr/bin/env python3
"""
Debug landmark movement to understand why convergence is failing.

Track how landmarks actually move from iteration to iteration and compare
with what the response maps say they should do.
"""

import cv2
import numpy as np
from pyclnf import CLNF
from pyclnf.core.optimizer import NURLMSOptimizer

# Load test frame
video_path = 'Patient Data/Normal Cohort/IMG_0433.MOV'
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
face_bbox = (241, 555, 532, 532)

# Initialize CLNF
clnf = CLNF(model_dir='pyclnf/models', max_iterations=5)

# Get initial parameters
initial_params = clnf.pdm.init_params(face_bbox)
print("=" * 80)
print("LANDMARK MOVEMENT DIAGNOSTIC")
print("=" * 80)
print()

# Track landmarks for a few key points
track_landmarks = [48, 54, 33]  # Mouth corners and nose tip
print(f"Tracking landmarks: {track_landmarks}")
print()

# Store landmarks at each iteration
landmarks_history = []
params_history = []

# Manually run optimization to track landmarks
params = initial_params.copy()

for iteration in range(5):
    # Get current landmarks
    landmarks_2d = clnf.pdm.params_to_landmarks_2d(params)
    landmarks_history.append(landmarks_2d.copy())
    params_history.append(params.copy())

    print(f"Iteration {iteration}:")
    print(f"  Current landmarks:")
    for lm_idx in track_landmarks:
        lm_x, lm_y = landmarks_2d[lm_idx]
        print(f"    Landmark {lm_idx}: ({lm_x:.2f}, {lm_y:.2f})")

    # Get patch experts and compute response map for one landmark
    patch_experts = clnf._get_patch_experts(view_idx=0, scale=0.25)
    landmark_idx = 48  # Mouth corner

    if landmark_idx in patch_experts:
        patch_expert = patch_experts[landmark_idx]
        lm_x, lm_y = landmarks_2d[landmark_idx]

        # Compute response map WITHOUT warping for simplicity
        optimizer = NURLMSOptimizer()
        response_map = optimizer._compute_response_map(
            gray, lm_x, lm_y, patch_expert, window_size=11,
            sim_img_to_ref=None,
            sim_ref_to_img=None,
            sigma_components=None
        )

        # Find peak offset
        peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
        peak_y, peak_x = peak_idx
        center = (11 - 1) / 2.0
        offset_x = peak_x - center
        offset_y = peak_y - center

        print(f"  Response map for landmark {landmark_idx}:")
        print(f"    Peak offset: ({offset_x:+.1f}, {offset_y:+.1f}) pixels")
        print(f"    Landmark should move by ~({offset_x:+.1f}, {offset_y:+.1f})")

    # Run one iteration of optimization (without sigma for simplicity)
    optimizer = NURLMSOptimizer(max_iterations=1)
    params, info = optimizer.optimize(
        clnf.pdm,
        gray,
        patch_experts,
        params,
        sigma_components=None,  # Disable sigma for simpler diagnostic
        window_sizes=[11]  # Use single window size
    )

    print(f"  Parameter update magnitude: {info['iteration_history'][0]['update_magnitude']:.6f}")
    print()

    # Show actual landmark movement
    if iteration > 0:
        landmarks_prev = landmarks_history[-2]
        landmarks_curr = landmarks_history[-1]

        print(f"  Actual landmark movement from previous iteration:")
        for lm_idx in track_landmarks:
            dx = landmarks_curr[lm_idx, 0] - landmarks_prev[lm_idx, 0]
            dy = landmarks_curr[lm_idx, 1] - landmarks_prev[lm_idx, 1]
            dist = np.sqrt(dx**2 + dy**2)
            print(f"    Landmark {lm_idx}: ({dx:+.2f}, {dy:+.2f}) - {dist:.2f}px")
        print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)

# Compute total movement
landmarks_initial = landmarks_history[0]
landmarks_final = landmarks_history[-1]
total_movement = landmarks_final - landmarks_initial
total_distances = np.linalg.norm(total_movement, axis=1)

print(f"Total movement over {len(landmarks_history)-1} iterations:")
print(f"  Mean: {total_distances.mean():.2f} pixels")
print(f"  Max: {total_distances.max():.2f} pixels")
print(f"  Tracked landmarks:")
for lm_idx in track_landmarks:
    dx, dy = total_movement[lm_idx]
    dist = total_distances[lm_idx]
    print(f"    Landmark {lm_idx}: ({dx:+.2f}, {dy:+.2f}) - {dist:.2f}px")

print()
print("Question: Are landmarks moving in the direction indicated by response maps?")
print("  If YES but not converging: Need smaller/better parameter updates")
print("  If NO: Sign error or coordinate system mismatch")
