#!/usr/bin/env python3
"""Test using OptimizedFaceDetector exactly as Face Mirror does"""

import cv2
import time
from pathlib import Path

print("=" * 70)
print("Testing OptimizedFaceDetector (Face Mirror method)")
print("=" * 70)

# Import exactly as Face Mirror does
from onnx_retinaface_detector import OptimizedFaceDetector

print("\n[1/4] Initializing detector...")
start = time.time()
detector = OptimizedFaceDetector(
    model_path='weights/Alignment_RetinaFace.pth',
    onnx_model_path='weights/retinaface_mobilenet025_coreml.onnx',
    device='cpu',
    confidence_threshold=0.5,
    nms_threshold=0.4
)
init_time = time.time() - start
print(f"✓ Initialized in {init_time:.2f}s")
print(f"  Backend: {detector.backend}")

print("\n[2/4] Loading video...")
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Failed to open video")
    exit(1)

ret, frame = cap.read()
if not ret:
    print("❌ Failed to read frame")
    exit(1)
print(f"✓ Frame loaded: {frame.shape}")

print("\n[3/4] Detecting face (first detection)...")
start = time.time()
detections = detector.detect_faces(frame, resize=1.0)
detect_time = (time.time() - start) * 1000
print(f"✓ Detection took {detect_time:.1f}ms")
print(f"  Faces found: {len(detections)}")

if len(detections) > 0:
    print(f"  First face: {detections[0][:4]}")

print("\n[4/4] Testing 10 more frames...")
times = []
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    detections = detector.detect_faces(frame, resize=1.0)
    times.append((time.time() - start) * 1000)

if times:
    avg_time = sum(times) / len(times)
    print(f"✓ Average: {avg_time:.1f}ms per frame")
    print(f"  Min: {min(times):.1f}ms")
    print(f"  Max: {max(times):.1f}ms")
    print(f"  Throughput: {1000/avg_time:.1f} FPS")

cap.release()

print("\n" + "=" * 70)
print("✅ TEST PASSED!")
print("=" * 70)
