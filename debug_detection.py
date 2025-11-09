#!/usr/bin/env python3
"""
Debug why landmark detection is failing on IMG_8401_source.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector

# Test on the failing video
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV"

print("="*80)
print("DEBUGGING IMG_8401_source DETECTION FAILURE")
print("="*80)
print()

# Load frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

print(f"Frame shape: {frame.shape}")
print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
print()

# Test with detector in debug mode
print("Testing with debug mode enabled...")
detector = PyFaceAU68LandmarkDetector(
    debug_mode=True,
    use_clnf_refinement=True,
    skip_redetection=False
)

# Detect
landmarks, _ = detector.get_face_mesh(frame)

print()
print("="*80)
print("RESULTS:")
print("="*80)

if landmarks is not None:
    print(f"✓ Got {len(landmarks)} landmarks")
    print(f"Bbox: {detector.cached_bbox}")

    if detector.cached_bbox is not None:
        bbox = detector.cached_bbox
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        frame_width, frame_height = frame.shape[1], frame.shape[0]

        print(f"Bbox size: {bbox_width}x{bbox_height} pixels")
        print(f"Bbox coverage: {bbox_width/frame_width*100:.1f}% width, {bbox_height/frame_height*100:.1f}% height")

    # Check landmark spread
    x_min, y_min = np.min(landmarks, axis=0)
    x_max, y_max = np.max(landmarks, axis=0)
    spread_x = x_max - x_min
    spread_y = y_max - y_min

    print(f"Landmark spread: {spread_x:.0f}x{spread_y:.0f} pixels")
    print(f"Landmark range: x=[{x_min:.0f}, {x_max:.0f}], y=[{y_min:.0f}, {y_max:.0f}]")

    # Check if landmarks are clustered (bad) or spread out (good)
    expected_spread = min(frame.shape[:2]) * 0.4  # Should cover ~40% of frame
    actual_spread = (spread_x + spread_y) / 2

    print()
    if actual_spread < expected_spread * 0.3:
        print("❌ PROBLEM: Landmarks are CLUSTERED (too small spread)")
        print(f"   Expected spread: ~{expected_spread:.0f}px")
        print(f"   Actual spread: {actual_spread:.0f}px")
        print("   This indicates face detection or landmark detection failure")
    else:
        print("✓ Landmarks appear to have reasonable spread")

else:
    print("❌ Detection failed - no landmarks returned")

print()
