#!/usr/bin/env python3
"""
Test that cv2.VideoCapture gives correctly oriented frames WITHOUT additional rotation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))

import cv2
from video_rotation import get_video_rotation, normalize_rotation

# Test on the same video
video_path = "Patient Data/Normal Cohort/IMG_0422.MOV"

print(f"Testing: {video_path}")
print()

# Get rotation metadata
rotation_raw = get_video_rotation(video_path)
rotation = normalize_rotation(rotation_raw)

print(f"Raw rotation: {rotation_raw}")
print(f"Normalized rotation: {rotation}")
print()

# Read a frame with OpenCV (NO rotation applied)
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if ret:
    print(f"Frame shape from VideoCapture: {frame.shape}")
    print(f"  Height: {frame.shape[0]}")
    print(f"  Width:  {frame.shape[1]}")
    print()

    # Save WITHOUT applying any rotation
    cv2.imwrite("test_output/rotation_fix_test.jpg", frame)
    print("Saved: test_output/rotation_fix_test.jpg")
    print()
    print("✓ If the person in this image is UPRIGHT, the fix is correct!")
    print("✓ If the person is SIDEWAYS, cv2.VideoCapture is NOT auto-rotating on this system.")
else:
    print("ERROR: Could not read frame")
