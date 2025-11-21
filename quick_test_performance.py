#!/usr/bin/env python3
"""
Quick test to measure CLNF performance with new settings.
"""

import cv2
import numpy as np
import time
from pathlib import Path
import sys

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))

print("Testing CLNF optimizations...")
print("-" * 60)

# Initialize components
from pymtcnn import MTCNN
from pyclnf import CLNF

detector = MTCNN()
clnf = CLNF(model_dir="pyclnf/models")

print(f"CLNF settings:")
print(f"  Max iterations: 5 (reduced from 10)")
print(f"  Convergence threshold: 0.5 (increased from 0.1)")
print()

# Load test frame
video_path = "Patient Data/Normal Cohort/Shorty.mov"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

# Test 5 iterations
timings = []
iterations_list = []
converged_list = []

print("Running 5 test iterations...")
for i in range(5):
    # Detect face
    detection = detector.detect(frame)
    if detection is not None and isinstance(detection, tuple) and len(detection) == 2:
        bboxes, confidences = detection
        if len(bboxes) > 0:
            bbox = bboxes[0]
            x, y, w, h = [int(v) for v in bbox]

            # Fit landmarks
            start = time.perf_counter()
            landmarks, info = clnf.fit(frame, (x, y, w, h))
            elapsed = (time.perf_counter() - start) * 1000

            timings.append(elapsed)
            iterations_list.append(info.get('iterations', 0))
            converged_list.append(info.get('converged', False))

            print(f"  Test {i+1}: {elapsed:.1f}ms, {info.get('iterations', 0)} iterations, converged: {info.get('converged', False)}")

print()
print("Results:")
print(f"  Average time: {np.mean(timings):.1f}ms (was ~911ms)")
print(f"  Average iterations: {np.mean(iterations_list):.1f} (was 10.0)")
print(f"  Convergence rate: {np.mean(converged_list)*100:.1f}% (was 0%)")
print()
print(f"Speedup: {911.0/np.mean(timings):.1f}x")