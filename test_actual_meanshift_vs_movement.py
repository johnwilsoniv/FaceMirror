#!/usr/bin/env python3
"""
Test to capture the ACTUAL mean-shift vector the optimizer computes
and compare with actual landmark movement.

This is more precise than comparing peak offsets, since the optimizer
uses KDE mean-shift which may differ from simple peak detection.
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
print("ACTUAL MEAN-SHIFT VS LANDMARK MOVEMENT TEST")
print("=" * 80)
print()

# Initialize CLNF
clnf = CLNF(model_dir='pyclnf/models', max_iterations=1)

# Monkey-patch the optimizer to capture the mean-shift vector
captured_data = {}

original_solve_update = clnf.optimizer._solve_update

def capturing_solve_update(J, v, W, Lambda_inv, params):
    """Capture mean-shift vector before solving."""
    captured_data['mean_shift'] = v.copy()
    captured_data['jacobian'] = J.copy()
    captured_data['params_before'] = params.copy()
    return original_solve_update(J, v, W, Lambda_inv, params)

clnf.optimizer._solve_update = capturing_solve_update

# Get initial landmarks
initial_params = clnf.pdm.init_params(face_bbox)
initial_landmarks = clnf.pdm.params_to_landmarks_2d(initial_params)

print("Running 1 iteration to capture mean-shift and actual movement...")
landmarks_final, info = clnf.fit(gray, face_bbox)

print()
print("=" * 80)
print("MEAN-SHIFT ANALYSIS")
print("=" * 80)
print()

# Extract captured mean-shift vector
mean_shift = captured_data['mean_shift']

# Test a few landmarks
test_landmarks = [48, 54, 33]  # Mouth corners, nose

print(f"Testing landmarks: {test_landmarks}")
print()

all_correct = True

for lm_idx in test_landmarks:
    # Mean-shift for this landmark
    ms_x = mean_shift[2 * lm_idx]
    ms_y = mean_shift[2 * lm_idx + 1]

    # Actual landmark movement
    initial_pos = initial_landmarks[lm_idx]
    final_pos = landmarks_final[lm_idx]
    actual_dx = final_pos[0] - initial_pos[0]
    actual_dy = final_pos[1] - initial_pos[1]

    print(f"Landmark {lm_idx}:")
    print(f"  Initial position: ({initial_pos[0]:.1f}, {initial_pos[1]:.1f})")
    print(f"  Mean-shift says move: ({ms_x:+.2f}, {ms_y:+.2f}) pixels")
    print(f"  Actual movement:      ({actual_dx:+.2f}, {actual_dy:+.2f}) pixels")

    # Check if signs match (ignoring very small movements)
    if abs(ms_x) > 0.5:
        x_sign_match = (np.sign(ms_x) == np.sign(actual_dx))
        if not x_sign_match:
            print(f"  ❌ X direction OPPOSITE! (mean-shift: {ms_x:+.2f}, actual: {actual_dx:+.2f})")
            all_correct = False
        else:
            print(f"  ✓ X direction correct")
    else:
        print(f"  ~ X mean-shift too small ({ms_x:.2f}), skipping")

    if abs(ms_y) > 0.5:
        y_sign_match = (np.sign(ms_y) == np.sign(actual_dy))
        if not y_sign_match:
            print(f"  ❌ Y direction OPPOSITE! (mean-shift: {ms_y:+.2f}, actual: {actual_dy:+.2f})")
            all_correct = False
        else:
            print(f"  ✓ Y direction correct")
    else:
        print(f"  ~ Y mean-shift too small ({ms_y:.2f}), skipping")

    # Compute angle between vectors
    if np.linalg.norm([ms_x, ms_y]) > 1.0 and np.linalg.norm([actual_dx, actual_dy]) > 1.0:
        ms_vec = np.array([ms_x, ms_y])
        actual_vec = np.array([actual_dx, actual_dy])

        # Normalize and compute angle
        ms_norm = ms_vec / np.linalg.norm(ms_vec)
        actual_norm = actual_vec / np.linalg.norm(actual_vec)

        dot_product = np.dot(ms_norm, actual_norm)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi

        print(f"  Angle between vectors: {angle:.1f}°")

        if angle > 90:
            print(f"  ⚠️  Vectors point in OPPOSITE directions (angle > 90°)")
            all_correct = False

    print()

print("=" * 80)
print("CONCLUSION:")
print("=" * 80)

if all_correct:
    print("✓ All landmarks moved in the direction indicated by mean-shift!")
    print("  Sign convention is CORRECT.")
    print("  Issue must be magnitude/scaling or Jacobian.")
else:
    print("❌ SIGN ERROR CONFIRMED!")
    print("   Landmarks moved OPPOSITE to mean-shift direction.")
    print("   The bug is in one of:")
    print("   1. Mean-shift sign (unlikely - verified correct)")
    print("   2. Parameter update equation sign")
    print("   3. Jacobian sign")
    print("   4. Parameter application")

print()
print("=" * 80)
