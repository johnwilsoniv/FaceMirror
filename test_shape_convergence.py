"""
Test shape-based convergence fix.
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
print("Testing Shape-Based Convergence (OpenFace Early Stopping)")
print("=" * 80)

# Test with shape-based convergence
print("\nTest: max_iterations=10 with shape-based early stopping")
clnf = CLNF(model_dir="pyclnf/models", max_iterations=10)
landmarks, info = clnf.fit(frame, face_bbox, return_params=True)

print(f"  Converged: {info['converged']}")
print(f"  Total iterations: {info['iterations']}")
print(f"  Final update magnitude: {info['final_update']:.6f}")
print(f"  Shape params mean |value|: {np.abs(info['params'][6:]).mean():.3f}")

# Compare with OpenFace C++ (from previous test)
print("\n" + "=" * 80)
print("Expected OpenFace C++ behavior:")
print("  - Converged: True (with shape-based early stopping)")
print("  - Iterations: ~5-15 (stops when shape change < 0.01 pixels)")
print("  - Shape params mean |value|: ~3.0")
print("=" * 80)

if info['converged']:
    print("\n SUCCESS! Shape-based convergence is working!")
else:
    print(f"\n  Still not converging after {info['iterations']} iterations")
    print("  Shape change may be > 0.01 pixels between iterations")
