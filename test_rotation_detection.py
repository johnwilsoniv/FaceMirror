#!/usr/bin/env python3
"""
Test rotation detection and correction for iPhone videos.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))

import cv2
from video_rotation import get_video_rotation, normalize_rotation

# Test on a sample video
video_path = "Patient Data/Normal Cohort/IMG_0422.MOV"

print(f"Testing: {video_path}")
print()

# Get rotation metadata
rotation_raw = get_video_rotation(video_path)
rotation = normalize_rotation(rotation_raw)

print(f"Raw rotation: {rotation_raw}")
print(f"Normalized rotation: {rotation}")
print()

# Read a frame with OpenCV
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if ret:
    print(f"Frame shape from VideoCapture: {frame.shape}")
    print(f"  Height: {frame.shape[0]}")
    print(f"  Width:  {frame.shape[1]}")
    print()

    # Save original frame
    cv2.imwrite("test_output/rotation_test_original.jpg", frame)
    print("Saved: test_output/rotation_test_original.jpg")

    # Apply rotations and save
    if rotation == 90:
        rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("test_output/rotation_test_90cw.jpg", rotated)
        print("Saved: test_output/rotation_test_90cw.jpg (90° clockwise)")
    elif rotation == 270:
        rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite("test_output/rotation_test_270.jpg", rotated)
        print("Saved: test_output/rotation_test_270.jpg (270° / 90° CCW)")

        # Also try opposite
        rotated2 = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("test_output/rotation_test_90cw_alt.jpg", rotated2)
        print("Saved: test_output/rotation_test_90cw_alt.jpg (90° clockwise - alternative)")

    print()
    print("Check the saved images to see which orientation is correct!")
else:
    print("ERROR: Could not read frame")
