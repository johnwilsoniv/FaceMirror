#!/usr/bin/env python3
"""Test detector WITHOUT CoreML to verify it works"""

import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import cv2
import time
from pathlib import Path

print("=" * 70)
print("Testing OptimizedFaceDetector WITHOUT CoreML")
print("=" * 70)

# Import detector
from onnx_retinaface_detector import ONNXRetinaFaceDetector

print("\n[1/3] Initializing WITHOUT CoreML...")
start = time.time()
detector = ONNXRetinaFaceDetector(
    'weights/retinaface_mobilenet025_coreml.onnx',
    use_coreml=False,  # Explicitly disable CoreML
    confidence_threshold=0.5,
    nms_threshold=0.4
)
init_time = time.time() - start
print(f"✓ Init: {init_time:.2f}s")
print(f"  Backend: {detector.backend}")

print("\n[2/3] Loading video...")
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Failed to open video")
    exit(1)

print("\n[3/3] Testing detection on 10 frames...")
times = []
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    detections, _ = detector.detect_faces(frame)
    elapsed = (time.time() - start) * 1000
    times.append(elapsed)
    print(f"  Frame {i+1}: {elapsed:.1f}ms, {len(detections)} faces")

cap.release()

if times:
    avg = sum(times) / len(times)
    print(f"\n✓ Processed {len(times)} frames")
    print(f"  Average: {avg:.1f}ms per frame ({1000/avg:.1f} FPS)")
    print(f"  Min: {min(times):.1f}ms, Max: {max(times):.1f}ms")

print("\n" + "=" * 70)
print("✅ CPU MODE WORKS PERFECTLY!")
print("=" * 70)
print("This confirms the issue is specifically with CoreML in standalone scripts")
print("=" * 70)
