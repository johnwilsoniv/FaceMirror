#!/usr/bin/env python3
"""
Debug why RetinaFace fails on mirrored videos
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from openface_integration import OpenFace3Processor

# Load a mirrored video
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

print("=" * 80)
print("DEBUGGING RETINAFACE FAILURE")
print("=" * 80)
print()

# Create processor with RetinaFace ENABLED
processor = OpenFace3Processor(
    device='cpu',
    skip_face_detection=False,  # Enable RetinaFace
    debug_mode=True
)

print(f"Processor config:")
print(f"  skip_face_detection: {processor.skip_face_detection}")
print(f"  confidence_threshold: {processor.confidence_threshold}")
print(f"  nms_threshold: {processor.nms_threshold}")
print()

# Open video and test first frame
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read frame!")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"Frame info:")
print(f"  Shape: {w}x{h}")
print(f"  Mean pixel value: {np.mean(frame):.1f}")
print(f"  Min/Max: {np.min(frame)}/{np.max(frame)}")
print()

# Save frame for inspection
cv2.imwrite("/tmp/mirrored_frame_sample.jpg", frame)
print(f"Saved sample frame to: /tmp/mirrored_frame_sample.jpg")
print()

# Try RetinaFace detection with different settings
print("Testing RetinaFace detection:")
print("-" * 80)

# Test 1: Default threshold
print("Test 1: Default threshold (0.5)")
dets = processor.preprocess_image(frame)
print(f"  Detections: {len(dets) if dets is not None and len(dets) > 0 else 0}")
if dets is not None and len(dets) > 0:
    for i, det in enumerate(dets[:5]):
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        conf = float(det[4])
        print(f"    Detection {i}: bbox=[{x1}, {y1}, {x2}, {y2}], conf={conf:.3f}")
else:
    print(f"  → NO DETECTIONS")
print()

# Test 2: Lower threshold
print("Test 2: Lower threshold (0.02)")
processor.confidence_threshold = 0.02
dets = processor.preprocess_image(frame)
print(f"  Detections: {len(dets) if dets is not None and len(dets) > 0 else 0}")
if dets is not None and len(dets) > 0:
    for i, det in enumerate(dets[:5]):
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        conf = float(det[4])
        print(f"    Detection {i}: bbox=[{x1}, {y1}, {x2}, {y2}], conf={conf:.3f}")
else:
    print(f"  → NO DETECTIONS")
print()

# Test 3: Try with rescaling
print("Test 3: With resize=0.5 (downscale)")
processor.confidence_threshold = 0.5
dets = processor.preprocess_image(frame, resize=0.5)
print(f"  Detections: {len(dets) if dets is not None and len(dets) > 0 else 0}")
if dets is not None and len(dets) > 0:
    for i, det in enumerate(dets[:5]):
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        conf = float(det[4])
        print(f"    Detection {i}: bbox=[{x1}, {y1}, {x2}, {y2}], conf={conf:.3f}")
else:
    print(f"  → NO DETECTIONS")
print()

# Test 4: Try with upscaling
print("Test 4: With resize=2.0 (upscale)")
dets = processor.preprocess_image(frame, resize=2.0)
print(f"  Detections: {len(dets) if dets is not None and len(dets) > 0 else 0}")
if dets is not None and len(dets) > 0:
    for i, det in enumerate(dets[:5]):
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        conf = float(det[4])
        print(f"    Detection {i}: bbox=[{x1}, {y1}, {x2}, {y2}], conf={conf:.3f}")
else:
    print(f"  → NO DETECTIONS")
print()

# Visualize the frame
print("Creating visualization...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original frame
ax1 = axes[0]
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
ax1.imshow(frame_rgb)
ax1.set_title("Mirrored Video Frame", fontsize=12, fontweight='bold')
ax1.axis('off')

# Grayscale for analysis
ax2 = axes[1]
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ax2.imshow(frame_gray, cmap='gray')
ax2.set_title("Grayscale", fontsize=12, fontweight='bold')
ax2.axis('off')

plt.tight_layout()
plt.savefig("/tmp/mirrored_frame_visualization.png", dpi=150, bbox_inches='tight')
print(f"Saved visualization to: /tmp/mirrored_frame_visualization.png")
print()

print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print()

if dets is None or len(dets) == 0:
    print("RetinaFace completely failed to detect the face.")
    print()
    print("Possible reasons:")
    print("  1. Face is too large (fills entire frame)")
    print("  2. RetinaFace expects faces at certain scales")
    print("  3. Mirrored frames have unusual characteristics")
    print("  4. Preprocessing or model issue")
    print()
    print("Recommendation:")
    print("  Use skip_face_detection=True for mirrored videos")
    print("  (Full frame is already the aligned face)")
else:
    print("RetinaFace CAN detect faces in mirrored videos!")
    print()
    print("This means the failure during processing might be due to:")
    print("  1. Different thresholds being used")
    print("  2. Frame-specific issues")
    print("  3. Threading/race conditions")

print("=" * 80)
