#!/usr/bin/env python3
"""Quick test of PFLD landmark detector on single frame"""

import cv2
import numpy as np
import sys

from pfld_landmark_detector import PFLDLandmarkDetector, visualize_landmarks

VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
PFLD_MODEL = "weights/pfld_68_landmarks.onnx"

print("Loading PFLD detector...")
detector = PFLDLandmarkDetector(PFLD_MODEL)
print("✅ Model loaded")
print()

print(f"Opening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Could not open video")
    sys.exit(1)

ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Could not read frame")
    sys.exit(1)

print(f"Frame shape: {frame.shape}")
print()

# Test with a simple face bounding box (center of frame)
h, w = frame.shape[:2]
# Assume face is roughly in center, 50% of frame size
face_size = min(h, w) * 0.5
x1 = int(w/2 - face_size/2)
y1 = int(h/2 - face_size/2)
x2 = int(w/2 + face_size/2)
y2 = int(h/2 + face_size/2)

bbox = [x1, y1, x2, y2]
print(f"Test bounding box: {bbox}")
print()

print("Running landmark detection...")
landmarks = detector.detect_landmarks(frame, bbox)

if landmarks is not None:
    print(f"✅ SUCCESS: Detected {len(landmarks)} landmarks")
    print(f"Landmark shape: {landmarks.shape}")
    print()
    print("First 5 landmarks:")
    for i in range(5):
        print(f"  Point {i}: ({landmarks[i][0]:.1f}, {landmarks[i][1]:.1f})")
    print()

    # Visualize
    frame_vis = visualize_landmarks(frame, landmarks)
    output_path = "pfld_test_output.jpg"
    cv2.imwrite(output_path, frame_vis)
    print(f"Visualization saved to: {output_path}")
else:
    print("❌ FAILED: No landmarks detected")
