#!/usr/bin/env python3
"""
Debug face detection on OF2.2 mirrored videos
"""

import config
config.apply_environment_settings()

import cv2
import numpy as np
from pathlib import Path

print("Loading face detector...")
from onnx_retinaface_detector import ONNXRetinaFaceDetector

# Test both videos
videos = [
    "/Users/johnwilsoniv/Documents/open2GR/1_Face_Mirror/output/IMG_0942_left_mirrored.mp4",
    "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/output/IMG_0942_left_mirrored.mp4"  # OF3.0 version
]

for video_path in videos:
    path = Path(video_path)
    if not path.exists():
        print(f"Skipping {path.name} - doesn't exist")
        continue

    print(f"\n{'='*80}")
    print(f"Testing: {path.name}")
    print(f"Path: {path}")
    print(f"{'='*80}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        continue

    # Read first few frames
    for frame_num in range(1, 6):
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {frame_num}: Cannot read")
            break

        print(f"\nFrame {frame_num}:")
        print(f"  Shape: {frame.shape}")
        print(f"  Dtype: {frame.dtype}")
        print(f"  Range: {frame.min()}-{frame.max()}")
        print(f"  Mean: {frame.mean():.1f}")

        # Test face detection with LOW threshold
        detector = ONNXRetinaFaceDetector(
            onnx_model_path="weights/retinaface_mobilenet025_coreml.onnx",
            confidence_threshold=0.01,  # Very low threshold
            nms_threshold=0.4,
            vis_threshold=0.01
        )

        print(f"  Running face detection (threshold=0.01)...")
        dets, img_raw = detector.detect_faces(frame, resize=1.0)

        if dets is None or len(dets) == 0:
            print(f"  ✗ NO FACES DETECTED")
            # Try with resize
            print(f"  Trying with resize=0.5...")
            dets, img_raw = detector.detect_faces(frame, resize=0.5)
            if dets is None or len(dets) == 0:
                print(f"  ✗ Still no faces with resize")
            else:
                print(f"  ✓ Found {len(dets)} faces with resize!")
                for i, det in enumerate(dets):
                    print(f"    Face {i}: bbox={det[:4].astype(int)}, conf={det[4]:.3f}")
        else:
            print(f"  ✓ Found {len(dets)} faces")
            for i, det in enumerate(dets):
                print(f"    Face {i}: bbox={det[:4].astype(int)}, conf={det[4]:.3f}")

        # Only test first frame
        break

    cap.release()

print("\n" + "="*80)
print("Debug complete")
