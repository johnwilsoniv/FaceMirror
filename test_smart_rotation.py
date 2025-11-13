#!/usr/bin/env python3
"""
Test smart rotation detection that handles both auto-rotated and non-auto-rotated videos.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))

import cv2
import numpy as np
from video_rotation import get_video_rotation, normalize_rotation

# Test videos - one that was working, one that wasn't
test_videos = [
    "Patient Data/Normal Cohort/IMG_0422.MOV",  # Was working (auto-rotated)
]

def needs_rotation(frame, metadata_rotation):
    """Check if frame needs manual rotation."""
    h, w = frame.shape[:2]
    is_portrait = h > w

    if metadata_rotation in [90, 270]:
        return is_portrait

    return False

def apply_rotation(frame, rotation):
    """Apply rotation to frame (270° = 90° clockwise)."""
    if rotation == 0:
        return frame
    elif rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    else:
        return frame

print("="*70)
print("SMART ROTATION TEST")
print("="*70)

for video_path in test_videos:
    print(f"\nTesting: {video_path}")

    # Get rotation metadata
    rotation_raw = get_video_rotation(video_path)
    rotation = normalize_rotation(rotation_raw)
    print(f"  Rotation metadata: {rotation}°")

    # Read frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("  ERROR: Could not read frame")
        continue

    h, w = frame.shape[:2]
    print(f"  Raw frame: {w}x{h} ({'portrait' if h > w else 'landscape'})")

    # Check if rotation needed
    needs_rot = needs_rotation(frame, rotation)
    print(f"  Needs manual rotation: {needs_rot}")

    # Apply if needed
    if needs_rot:
        frame = apply_rotation(frame, rotation)
        h2, w2 = frame.shape[:2]
        print(f"  After rotation: {w2}x{h2} ({'portrait' if h2 > w2 else 'landscape'})")

    # Save result
    output_name = Path(video_path).stem
    output_path = f"test_output/smart_rotation_{output_name}.jpg"
    cv2.imwrite(output_path, frame)
    print(f"  Saved: {output_path}")

print("\n" + "="*70)
print("✓ Check saved images - people should be UPRIGHT in all cases")
print("="*70)
