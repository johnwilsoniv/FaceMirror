"""
Quick test of increased iteration limit to check if convergence improves.
"""
import cv2
import numpy as np
from pyclnf import CLNF

# Load test video frame
video_path = "Patient Data/Normal Cohort/IMG_0433.MOV"
cap = cv2.VideoCapture(video_path)

# Seek to frame 50
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Failed to read frame from {video_path}")

# Face bbox (from previous tests)
face_bbox = (241, 555, 532, 532)

print("=" * 80)
print("Testing Increased Iteration Limit")
print("=" * 80)

# Test 1: max_iterations=10 (previous default)
print("\nTest 1: max_iterations=10 (4 windows × 10 = 40 total)")
clnf_10 = CLNF(model_dir="pyclnf/models", max_iterations=10)
landmarks_10, info_10 = clnf_10.fit(frame, face_bbox, return_params=True)

print(f"  Converged: {info_10['converged']}")
print(f"  Total iterations: {info_10['iterations']}")
print(f"  Final update magnitude: {info_10['final_update']:.6f}")
print(f"  Shape params mean |value|: {np.abs(info_10['params'][6:]).mean():.3f}")

# Test 2: max_iterations=25 (new test value)
print("\nTest 2: max_iterations=25 (4 windows × 25 = 100 total)")
clnf_25 = CLNF(model_dir="pyclnf/models", max_iterations=25)
landmarks_25, info_25 = clnf_25.fit(frame, face_bbox, return_params=True)

print(f"  Converged: {info_25['converged']}")
print(f"  Total iterations: {info_25['iterations']}")
print(f"  Final update magnitude: {info_25['final_update']:.6f}")
print(f"  Shape params mean |value|: {np.abs(info_25['params'][6:]).mean():.3f}")

# Compare landmark shift
landmark_shift = np.linalg.norm(landmarks_25 - landmarks_10, axis=1).mean()
print(f"\n  Mean landmark shift from 10→25 iterations: {landmark_shift:.2f} pixels")

# Test 3: max_iterations=50 (to see if convergence happens)
print("\nTest 3: max_iterations=50 (4 windows × 50 = 200 total)")
clnf_50 = CLNF(model_dir="pyclnf/models", max_iterations=50)
landmarks_50, info_50 = clnf_50.fit(frame, face_bbox, return_params=True)

print(f"  Converged: {info_50['converged']}")
print(f"  Total iterations: {info_50['iterations']}")
print(f"  Final update magnitude: {info_50['final_update']:.6f}")
print(f"  Shape params mean |value|: {np.abs(info_50['params'][6:]).mean():.3f}")

# Compare landmark shift
landmark_shift_50 = np.linalg.norm(landmarks_50 - landmarks_25, axis=1).mean()
print(f"\n  Mean landmark shift from 25→50 iterations: {landmark_shift_50:.2f} pixels")

print("\n" + "=" * 80)
print("Summary:")
print(f"  10 iter: converged={info_10['converged']}, total_iter={info_10['iterations']}, final_update={info_10['final_update']:.6f}")
print(f"  25 iter: converged={info_25['converged']}, total_iter={info_25['iterations']}, final_update={info_25['final_update']:.6f}")
print(f"  50 iter: converged={info_50['converged']}, total_iter={info_50['iterations']}, final_update={info_50['final_update']:.6f}")
print("=" * 80)
