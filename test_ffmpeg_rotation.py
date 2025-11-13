#!/usr/bin/env python3
"""
Test FFmpeg-based rotation approach (same as S1 Face Mirror).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))

# Import the function from collect_bbox_dataset
from collect_bbox_dataset import create_rotated_video_with_ffmpeg
import cv2

# Test video
video_path = "Patient Data/Normal Cohort/IMG_0422.MOV"
output_path = "test_output/ffmpeg_rotated_IMG_0422.mp4"

print("="*70)
print("FF MPEG ROTATION TEST")
print("="*70)
print(f"\nProcessing: {video_path}")

# Create rotated video with FFmpeg
rotated_video = create_rotated_video_with_ffmpeg(video_path, output_path)
print(f"Rotated video: {rotated_video}")

# Extract a frame from rotated video
cap = cv2.VideoCapture(rotated_video)
ret, frame = cap.read()
cap.release()

if ret:
    h, w = frame.shape[:2]
    print(f"Frame shape: {w}x{h} ({'portrait' if h > w else 'landscape'})")

    # Save frame
    frame_path = "test_output/ffmpeg_rotation_frame.jpg"
    cv2.imwrite(frame_path, frame)
    print(f"Saved frame: {frame_path}")
    print("\n" + "="*70)
    print("✓ Check the saved frame - person should be UPRIGHT")
    print("="*70)
else:
    print("ERROR: Could not read frame")
