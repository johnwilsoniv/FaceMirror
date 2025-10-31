#!/usr/bin/env python3
"""Test CoreML with Face Mirror's environment settings"""

import os
import sys

# Set environment variables BEFORE importing any ML libraries
# (Same as Face Mirror)
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

print("=" * 70)
print("CoreML Test with Face Mirror Environment Settings")
print("=" * 70)
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
print(f"OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS')}")
print("")

import cv2
import time
import numpy as np

# Test 1: Small image warmup
print("[1/5] Import detector...")
from onnx_retinaface_detector import ONNXRetinaFaceDetector

print("\n[2/5] Initialize with CoreML...")
start = time.time()
detector = ONNXRetinaFaceDetector(
    'weights/retinaface_mobilenet025_coreml.onnx',
    use_coreml=True,
    confidence_threshold=0.5,
    nms_threshold=0.4
)
init_time = time.time() - start
print(f"✓ Initialized in {init_time:.2f}s")
print(f"  Backend: {detector.backend}")

# Test 2: Warmup with small images
print("\n[3/5] Warmup with 5 small images (480x640)...")
warmup_times = []
for i in range(5):
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    start = time.time()
    detections, _ = detector.detect_faces(dummy_img)
    warmup_times.append((time.time() - start) * 1000)
    print(f"  Warmup {i+1}: {warmup_times[-1]:.1f}ms")

avg_warmup = sum(warmup_times) / len(warmup_times)
print(f"✓ Warmup avg: {avg_warmup:.1f}ms")

# Test 3: Test on medium image (1280x720)
print("\n[4/5] Test on medium resolution (1280x720)...")
medium_img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
start = time.time()
detections, _ = detector.detect_faces(medium_img)
medium_time = (time.time() - start) * 1000
print(f"✓ Medium image: {medium_time:.1f}ms, {len(detections)} faces")

# Test 4: Test on full HD image (1920x1080)
print("\n[5/5] Test on full HD (1920x1080)...")
hd_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
print("  Starting detection...")
start = time.time()
detections, _ = detector.detect_faces(hd_img)
hd_time = (time.time() - start) * 1000
print(f"✓ Full HD image: {hd_time:.1f}ms, {len(detections)} faces")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print(f"Backend: {detector.backend}")
print(f"Init: {init_time:.2f}s")
print(f"Warmup (480x640): {avg_warmup:.1f}ms avg")
print(f"Medium (1280x720): {medium_time:.1f}ms")
print(f"Full HD (1920x1080): {hd_time:.1f}ms")
print("=" * 70)
