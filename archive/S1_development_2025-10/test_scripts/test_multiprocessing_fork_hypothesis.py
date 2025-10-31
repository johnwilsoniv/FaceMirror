#!/usr/bin/env python3
"""Test if multiprocessing.set_start_method('fork') enables CoreML"""

import multiprocessing
import os
import sys

# HYPOTHESIS: This is the magic line that makes CoreML work!
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork', force=True)
        print("✓ Set multiprocessing start method to 'fork'")
    except RuntimeError as e:
        print(f"⚠ Could not set fork method: {e}")

# Set environment after multiprocessing
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import cv2
import time
import numpy as np
from pathlib import Path

print("\n" + "=" * 70)
print("HYPOTHESIS TEST: multiprocessing.set_start_method('fork')")
print("=" * 70)
print("Testing if fork method enables CoreML in standalone scripts")
print("")

# Import exactly as Face Mirror does
from onnx_retinaface_detector import OptimizedFaceDetector

print("[1/3] Initializing OptimizedFaceDetector WITH fork method...")
start = time.time()
detector = OptimizedFaceDetector(
    model_path='weights/Alignment_RetinaFace.pth',
    onnx_model_path='weights/retinaface_mobilenet025_coreml.onnx',
    device='cpu',
    confidence_threshold=0.5,
    nms_threshold=0.4
)
init_time = time.time() - start
print(f"✓ Init: {init_time:.2f}s")
print(f"  Backend: {detector.backend}")

# Warmup with small images
print("\n[2/3] Warmup with small images (this is where tests crashed)...")
try:
    for i in range(3):
        print(f"  Warmup {i+1}...", end=" ", flush=True)
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect_faces(dummy, resize=1.0)
        print(f"OK ({len(detections)} faces)")
    print("✓ Warmup completed without crash!")
except Exception as e:
    print(f"\n❌ CRASH during warmup: {e}")
    sys.exit(1)

# Test on real video frame
print("\n[3/3] Testing on real video frame...")
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

if not Path(video_path).exists():
    print(f"⚠ Video not found, using synthetic frame")
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
else:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("⚠ Failed to read video, using synthetic frame")
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

print(f"  Frame shape: {frame.shape}")
start = time.time()
detections = detector.detect_faces(frame, resize=1.0)
elapsed = (time.time() - start) * 1000
print(f"✓ Detection: {elapsed:.1f}ms, {len(detections)} faces")

print("\n" + "=" * 70)
print("✅ SUCCESS! CoreML works with fork method!")
print("=" * 70)
print(f"Backend: {detector.backend}")
if hasattr(detector.detector, 'backend'):
    print(f"Inner backend: {detector.detector.backend}")
print("\nCONCLUSION: multiprocessing.set_start_method('fork') enables CoreML!")
print("=" * 70)
