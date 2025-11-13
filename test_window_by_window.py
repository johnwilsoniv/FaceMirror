#!/usr/bin/env python3
"""
Test to track landmark movement at each window size iteration.

The CLNF optimizer loops over window sizes [11, 9, 7]. Maybe the sign error
appears at a specific window size, or is a cumulative effect.
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
print("WINDOW-BY-WINDOW LANDMARK TRACKING")
print("=" * 80)
print()

# Track landmark 33 (nose tip - shows sign error)
track_landmark = 33

# Initialize CLNF
clnf = CLNF(model_dir='pyclnf/models', max_iterations=1)

# Get initial parameters and landmarks
initial_params = clnf.pdm.init_params(face_bbox)
initial_landmarks = clnf.pdm.params_to_landmarks_2d(initial_params)

print(f"Tracking landmark {track_landmark} (nose tip - SHOWS SIGN ERROR)")
print(f"Initial position: ({initial_landmarks[track_landmark, 0]:.2f}, {initial_landmarks[track_landmark, 1]:.2f})")
print()

# Manually run optimization for EACH window size to see what happens
params = initial_params.copy()
landmarks_history = [initial_landmarks.copy()]
window_sizes = clnf.window_sizes  # Should be [11, 9, 7]

print(f"Window sizes: {window_sizes}")
print()

for ws_idx, window_size in enumerate(window_sizes):
    print(f"{'='*80}")
    print(f"Processing window size {window_size} (iteration {ws_idx + 1}/{len(window_sizes)})")
    print(f"{'='*80}")

    # Get current landmarks
    landmarks_before = clnf.pdm.params_to_landmarks_2d(params)
    lm_before = landmarks_before[track_landmark]

    print(f"Landmark {track_landmark} BEFORE ws={window_size}: ({lm_before[0]:.2f}, {lm_before[1]:.2f})")

    # Get patch experts for this window
    scale_idx = clnf.window_to_scale[window_size]
    patch_scale = clnf.patch_scaling[scale_idx]
    patch_experts = clnf._get_patch_experts(view_idx=0, scale=patch_scale)

    # Compute response map for landmark 33 BEFORE optimization
    if track_landmark in patch_experts:
        patch_expert = patch_experts[track_landmark]
        lm_x, lm_y = lm_before

        # Compute response map
        response_map = clnf.optimizer._compute_response_map(
            gray, lm_x, lm_y, patch_expert, window_size,
            sim_img_to_ref=None,
            sim_ref_to_img=None,
            sigma_components=None
        )

        # Find peak offset
        peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
        peak_y, peak_x = peak_idx
        center = (window_size - 1) / 2.0
        offset_x = peak_x - center
        offset_y = peak_y - center

        print(f"  Response map peak offset: ({offset_x:+.2f}, {offset_y:+.2f})")
        print(f"  → Landmark should move by ~({offset_x:+.2f}, {offset_y:+.2f})")

    # Run ONE iteration with THIS window size only
    # Create optimizer with max_iterations=1
    from pyclnf.core.optimizer import NURLMSOptimizer
    temp_optimizer = NURLMSOptimizer(
        max_iterations=1,
        regularization=clnf.optimizer.regularization,
        sigma=clnf.optimizer.sigma,
        weight_multiplier=clnf.optimizer.weight_multiplier
    )

    params_updated, info = temp_optimizer.optimize(
        clnf.pdm,
        params,
        patch_experts,
        gray,
        weights=None,  # Use uniform weights for simplicity
        window_size=window_size,
        patch_scaling=patch_scale,
        sigma_components=None  # Disable sigma for simpler diagnostic
    )

    # Get landmarks AFTER optimization
    landmarks_after = clnf.pdm.params_to_landmarks_2d(params_updated)
    lm_after = landmarks_after[track_landmark]

    # Compute actual movement
    actual_dx = lm_after[0] - lm_before[0]
    actual_dy = lm_after[1] - lm_before[1]

    print(f"Landmark {track_landmark} AFTER ws={window_size}: ({lm_after[0]:.2f}, {lm_after[1]:.2f})")
    print(f"  Actual movement: ({actual_dx:+.2f}, {actual_dy:+.2f})")

    # Check if direction matches
    if track_landmark in patch_experts and abs(offset_x) > 0.5 and abs(offset_y) > 0.5:
        # Compute angle between expected and actual
        expected_vec = np.array([offset_x, offset_y])
        actual_vec = np.array([actual_dx, actual_dy])

        if np.linalg.norm(expected_vec) > 0.5 and np.linalg.norm(actual_vec) > 0.5:
            expected_norm = expected_vec / np.linalg.norm(expected_vec)
            actual_norm = actual_vec / np.linalg.norm(actual_vec)

            dot_product = np.dot(expected_norm, actual_norm)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi

            print(f"  Angle between expected and actual: {angle:.1f}°")

            if angle > 90:
                print(f"  ❌ OPPOSITE DIRECTION (angle > 90°)")
            else:
                print(f"  ✓ Correct direction")

    print(f"  Update magnitude: {info['iteration_history'][0]['update_magnitude']:.6f}")
    print()

    # Update params for next iteration
    params = params_updated.copy()
    landmarks_history.append(landmarks_after.copy())

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

# Compare initial vs final
landmarks_initial = landmarks_history[0]
landmarks_final = landmarks_history[-1]

lm_initial = landmarks_initial[track_landmark]
lm_final = landmarks_final[track_landmark]

total_dx = lm_final[0] - lm_initial[0]
total_dy = lm_final[1] - lm_initial[1]
total_dist = np.sqrt(total_dx**2 + total_dy**2)

print(f"Landmark {track_landmark} total movement:")
print(f"  Initial: ({lm_initial[0]:.2f}, {lm_initial[1]:.2f})")
print(f"  Final:   ({lm_final[0]:.2f}, {lm_final[1]:.2f})")
print(f"  Movement: ({total_dx:+.2f}, {total_dy:+.2f}) - {total_dist:.2f}px")
print()

# Show movement at each window size
print(f"Breakdown by window size:")
for i, ws in enumerate(window_sizes):
    lm_before = landmarks_history[i][track_landmark]
    lm_after = landmarks_history[i + 1][track_landmark]
    dx = lm_after[0] - lm_before[0]
    dy = lm_after[1] - lm_before[1]
    dist = np.sqrt(dx**2 + dy**2)
    print(f"  ws={ws:2d}: ({dx:+.2f}, {dy:+.2f}) - {dist:.2f}px")

print()
print("=" * 80)
print("KEY QUESTION:")
print("=" * 80)
print("Does the sign error occur at a specific window size,")
print("or is it a cumulative effect across all window sizes?")
print("=" * 80)
