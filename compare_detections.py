#!/usr/bin/env python3
"""
Compare RetinaFace detection on working vs failing videos.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))
from pyfaceau.detectors.retinaface import ONNXRetinaFaceDetector

# Initialize RetinaFace
detector = ONNXRetinaFaceDetector(
    str(Path(__file__).parent / 'S1 Face Mirror/weights/retinaface_mobilenet025_coreml.onnx'),
    use_coreml=True,
    confidence_threshold=0.5,
    nms_threshold=0.4
)

test_videos = [
    ("IMG_0942_source (WORKING)", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0942_source.MOV"),
    ("IMG_8401_source (FAILING)", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV"),
    ("IMG_9330_source (FAILING)", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV"),
]

print("="*80)
print("RETINAFACE DETECTION COMPARISON")
print("="*80)
print()

for name, path in test_videos:
    print(f"{name}:")

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    print(f"  Frame size: {frame.shape[1]}x{frame.shape[0]}")

    # Detect faces
    detections, landmarks = detector.detect_faces(frame)

    if detections is not None and len(detections) > 0:
        print(f"  Found {len(detections)} face(s)")

        for i, det in enumerate(detections):
            bbox = det[:4].astype(int)
            conf = det[4]

            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            frame_center_x = frame.shape[1] / 2
            frame_center_y = frame.shape[0] / 2

            offset_x = bbox_center_x - frame_center_x
            offset_y = bbox_center_y - frame_center_y

            print(f"  Detection {i}:")
            print(f"    Bbox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            print(f"    Size: {bbox_w}x{bbox_h} pixels")
            print(f"    Coverage: {bbox_w/frame.shape[1]*100:.1f}% width × {bbox_h/frame.shape[0]*100:.1f}% height")
            print(f"    Confidence: {conf:.3f}")
            print(f"    Center: ({bbox_center_x:.0f}, {bbox_center_y:.0f})")
            print(f"    Offset from frame center: ({offset_x:.0f}, {offset_y:.0f})")

            # Check if bbox seems reasonable for a face
            if bbox_w < frame.shape[1] * 0.15 or bbox_h < frame.shape[0] * 0.15:
                print(f"    ⚠️  WARNING: Bbox too small (likely wrong)")
            elif bbox_w > frame.shape[1] * 0.8 or bbox_h > frame.shape[0] * 0.8:
                print(f"    ⚠️  WARNING: Bbox too large (likely wrong)")
            else:
                print(f"    ✓ Bbox size seems reasonable")
    else:
        print(f"  ❌ No face detected!")

    print()

print("="*80)
