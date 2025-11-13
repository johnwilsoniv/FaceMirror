"""
Check response map value ranges to see if there's a scaling issue.
"""
import cv2
import numpy as np
from pyclnf import CLNF
import pyclnf.core.optimizer as opt_module

# Monkey-patch to inspect response maps
original_compute_response_map = opt_module.NURLMSOptimizer._compute_response_map

def debug_response_map(self, image, center_x, center_y, patch_expert, window_size, sim_img_to_ref=None, sim_ref_to_img=None):
    """Wrapper to inspect response map values."""
    response_map = original_compute_response_map(self, image, center_x, center_y, patch_expert, window_size, sim_img_to_ref, sim_ref_to_img)

    if response_map is not None:
        print(f"      Response map: min={response_map.min():.6f}, max={response_map.max():.6f}, mean={response_map.mean():.6f}, sum={response_map.sum():.6f}")

    return response_map

opt_module.NURLMSOptimizer._compute_response_map = debug_response_map

# Load test data
video_path = "Patient Data/Normal Cohort/IMG_0433.MOV"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Failed to read frame from {video_path}")

face_bbox = (241, 555, 532, 532)

print("=" * 80)
print("Debugging Response Map Values (First Window Only)")
print("=" * 80)

clnf = CLNF(model_dir="pyclnf/models", max_iterations=1)
landmarks, info = clnf.fit(frame, face_bbox)

print("\n" + "=" * 80)
print("Expected CCNF Response Values:")
print("  - Range: typically 0 to 1 (probability-like)")
print("  - Or: exp(-distance) where distance could be 0-10")
print("  - Mean: ~0.1-0.5 for reasonable responses")
print("  ")
print("If our values are 10-100× larger, we have a scaling bug!")
print("=" * 80)
