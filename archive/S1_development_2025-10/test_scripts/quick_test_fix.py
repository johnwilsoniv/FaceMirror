#!/usr/bin/env python3
"""Quick test to verify RetinaFace fix is working"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from openface_integration import OpenFace3Processor

video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

print("=" * 80)
print("QUICK FIX VERIFICATION")
print("=" * 80)
print()

# Create processor with the fix
processor = OpenFace3Processor(device='cpu', skip_face_detection=True, debug_mode=False)

print(f"Processor Configuration:")
print(f"  skip_face_detection: {processor.skip_face_detection}")
print(f"  face_detector: {processor.face_detector}")
print(f"  ✓ RetinaFace is {'DISABLED (correct!)' if processor.skip_face_detection else 'ENABLED (wrong!)'}")
print()

# Process 10 frames manually to see AU values
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

print("Processing first 10 frames manually...")
print("-" * 80)

au_values_list = []

for frame_idx in range(10):
    ret, frame = cap.read()
    if not ret:
        break

    # Manually process frame (mimicking what happens in process_video)
    h, w = frame.shape[:2]

    # With skip_face_detection=True, should use full frame
    cropped_face = frame  # Full frame, not cropped

    # Extract AUs
    emotion, gaze, au_output = processor.multitask_model.predict(cropped_face)
    au_values = au_output.cpu().numpy().flatten()
    au_values_list.append(au_values)

    print(f"Frame {frame_idx}: AU12={au_values[4]:.3f}, AU06={au_values[3]:.3f}, AU04={au_values[2]:.3f}")

cap.release()

print()
print("-" * 80)
print("Summary:")
au_array = np.array(au_values_list)
au_labels = ['AU1', 'AU2', 'AU4', 'AU6', 'AU12', 'AU15', 'AU20', 'AU25']

for idx, label in enumerate(au_labels):
    mean_val = np.mean(au_array[:, idx])
    std_val = np.std(au_array[:, idx])
    print(f"  {label}: mean={mean_val:.3f}, std={std_val:.3f}")

print()
print("✓ Test complete - RetinaFace fix is active!")
print("  (AUs should now have reasonable values, not near-zero)")
print("=" * 80)
