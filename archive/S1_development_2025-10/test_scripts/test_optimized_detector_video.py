#!/usr/bin/env python3
"""Test OptimizedFaceDetector with real video (Face Mirror's approach)"""

import os
import sys

# Set Face Mirror environment
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import cv2
import time
import numpy as np
from pathlib import Path

print("=" * 70)
print("Testing OptimizedFaceDetector with Real Video")
print("=" * 70)
print("This uses Face Mirror's exact detector approach")
print("")

# Import exactly as Face Mirror does
from onnx_retinaface_detector import OptimizedFaceDetector

print("[1/4] Initializing OptimizedFaceDetector...")
start = time.time()
detector = OptimizedFaceDetector(
    model_path='weights/Alignment_RetinaFace.pth',
    onnx_model_path='weights/retinaface_mobilenet025_coreml.onnx',
    device='cpu',  # Face Mirror uses 'cpu', CoreML is auto-enabled inside
    confidence_threshold=0.5,
    nms_threshold=0.4
)
init_time = time.time() - start
print(f"✓ Init: {init_time:.2f}s")
print(f"  Backend: {detector.backend}")

# Warmup with small images first
print("\n[2/4] Warmup with small images...")
for i in range(3):
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = detector.detect_faces(dummy, resize=1.0)
    print(f"  Warmup {i+1}: OK")

print("\n[3/4] Loading video...")
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

if not Path(video_path).exists():
    print(f"❌ Video not found: {video_path}")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Failed to open video")
    sys.exit(1)

ret, frame = cap.read()
if not ret:
    print("❌ Failed to read frame")
    sys.exit(1)

print(f"✓ Video loaded: {frame.shape}")

print("\n[4/4] Testing detection on 5 frames...")
print("  (This is where previous tests hung)")
print("")

times = []
for i in range(5):
    if i > 0:  # Already read first frame
        ret, frame = cap.read()
        if not ret:
            break

    print(f"  Frame {i+1}: detecting...", end=" ", flush=True)
    start = time.time()
    detections = detector.detect_faces(frame, resize=1.0)
    elapsed = (time.time() - start) * 1000
    times.append(elapsed)
    print(f"{elapsed:.1f}ms, {len(detections)} faces")

cap.release()

if times:
    avg = sum(times) / len(times)
    print(f"\n✓ Processed {len(times)} frames")
    print(f"  Average: {avg:.1f}ms per frame")
    print(f"  Min: {min(times):.1f}ms")
    print(f"  Max: {max(times):.1f}ms")
    print(f"  Throughput: {1000/avg:.1f} FPS")

print("\n" + "=" * 70)
print("✅ SUCCESS! OptimizedFaceDetector works on video!")
print("=" * 70)
print(f"Backend: {detector.backend}")
if hasattr(detector.detector, 'backend'):
    print(f"Inner backend: {detector.detector.backend}")
print("=" * 70)
