#!/usr/bin/env python3
"""
Test for sign error: Compare response map peak directions vs actual landmark movement.

If there's a sign error, landmarks will move OPPOSITE to where response maps say.
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

print("=" * 80)
print("SIGN ERROR TEST: Response Map Direction vs Actual Movement")
print("=" * 80)
print()

# Initialize CLNF
clnf = CLNF(model_dir='pyclnf/models', max_iterations=1)

# Get initial params and landmarks
initial_params = clnf.pdm.init_params(face_bbox)
initial_landmarks = clnf.pdm.params_to_landmarks_2d(initial_params)

# Get patch experts
patch_experts = clnf._get_patch_experts(view_idx=0, scale=0.25)

# Test a few landmarks
test_landmarks = [48, 54, 33]  # Mouth corners, nose
window_size = 11

print(f"Testing landmarks: {test_landmarks}")
print(f"Window size: {window_size}")
print()

# For each landmark, compute response map peak offset BEFORE optimization
optimizer = NURLMSOptimizer()
peak_offsets = {}

print("Computing response map peak offsets BEFORE optimization:")
for lm_idx in test_landmarks:
    if lm_idx not in patch_experts:
        continue

    patch_expert = patch_experts[lm_idx]
    lm_x, lm_y = initial_landmarks[lm_idx]

    # Compute response map
    response_map = optimizer._compute_response_map(
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

    peak_offsets[lm_idx] = (offset_x, offset_y)

    print(f"  Landmark {lm_idx} at ({lm_x:.1f}, {lm_y:.1f})")
    print(f"    Response map peak offset: ({offset_x:+.1f}, {offset_y:+.1f})")
    print(f"    ↳ Landmark should move RIGHT {offset_x:+.1f}px, DOWN {offset_y:+.1f}px")

print()
print("-" * 80)
print("Running 1 iteration of optimization...")
print("-" * 80)
print()

# Run ONE iteration
final_landmarks, info = clnf.fit(gray, face_bbox)

print("Actual landmark movement AFTER 1 iteration:")
for lm_idx in test_landmarks:
    if lm_idx not in peak_offsets:
        continue

    initial_pos = initial_landmarks[lm_idx]
    final_pos = final_landmarks[lm_idx]

    actual_dx = final_pos[0] - initial_pos[0]
    actual_dy = final_pos[1] - initial_pos[1]

    expected_dx, expected_dy = peak_offsets[lm_idx]

    print(f"  Landmark {lm_idx}:")
    print(f"    Expected direction: ({expected_dx:+.1f}, {expected_dy:+.1f})")
    print(f"    Actual movement:    ({actual_dx:+.1f}, {actual_dy:+.1f})")

    # Check sign agreement
    x_sign_match = (np.sign(expected_dx) == np.sign(actual_dx)) if abs(expected_dx) > 0.5 else True
    y_sign_match = (np.sign(expected_dy) == np.sign(actual_dy)) if abs(expected_dy) > 0.5 else True

    if not x_sign_match:
        print(f"    ❌ X direction OPPOSITE! (expected {expected_dx:+.1f}, got {actual_dx:+.1f})")
    else:
        print(f"    ✓ X direction correct")

    if not y_sign_match:
        print(f"    ❌ Y direction OPPOSITE! (expected {expected_dy:+.1f}, got {actual_dy:+.1f})")
    else:
        print(f"    ✓ Y direction correct")

    # Compute dot product (positive = same direction, negative = opposite)
    expected_vec = np.array([expected_dx, expected_dy])
    actual_vec = np.array([actual_dx, actual_dy])

    if np.linalg.norm(expected_vec) > 0.5 and np.linalg.norm(actual_vec) > 0.5:
        # Normalize
        expected_norm = expected_vec / np.linalg.norm(expected_vec)
        actual_norm = actual_vec / np.linalg.norm(actual_vec)

        dot_product = np.dot(expected_norm, actual_norm)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi

        print(f"    Angle between vectors: {angle:.1f}°")

        if angle > 90:
            print(f"    ⚠️  Vectors point in OPPOSITE directions (angle > 90°)")

    print()

print("=" * 80)
print("CONCLUSION:")
print("=" * 80)

# Count how many had wrong signs
wrong_signs = 0
for lm_idx in test_landmarks:
    if lm_idx not in peak_offsets:
        continue

    initial_pos = initial_landmarks[lm_idx]
    final_pos = final_landmarks[lm_idx]

    actual_dx = final_pos[0] - initial_pos[0]
    actual_dy = final_pos[1] - initial_pos[1]

    expected_dx, expected_dy = peak_offsets[lm_idx]

    x_sign_match = (np.sign(expected_dx) == np.sign(actual_dx)) if abs(expected_dx) > 0.5 else True
    y_sign_match = (np.sign(expected_dy) == np.sign(actual_dy)) if abs(expected_dy) > 0.5 else True

    if not x_sign_match or not y_sign_match:
        wrong_signs += 1

if wrong_signs > 0:
    print(f"❌ SIGN ERROR DETECTED: {wrong_signs}/{len(test_landmarks)} landmarks moved in wrong direction!")
    print("   This indicates a sign flip in either:")
    print("   - The mean-shift computation")
    print("   - The parameter update equation")
    print("   - The Jacobian")
else:
    print("✓ Signs are correct - landmarks moved in expected directions")
    print("  The issue may be magnitude/scaling rather than sign")

print()
print("=" * 80)
