#!/usr/bin/env python3
"""Simple CPU-only profiling - no CoreML issues"""

import os
import sys
import time
import numpy as np

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

sys.path.insert(0, '../pyfhog/src')

print("\n" + "=" * 80)
print("SIMPLE CPU PERFORMANCE PROFILE")
print("=" * 80)

# Import components individually to profile
from onnx_retinaface_detector import ONNXRetinaFaceDetector
from cunjian_pfld_detector import CunjianPFLDDetector
from calc_params import CalcParams
from pdm_parser import PDMParser
from openface22_face_aligner import OpenFace22FaceAligner
import pyfhog
import cv2

# Load video frame
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error loading frame")
    sys.exit(1)

print(f"Frame loaded: {frame.shape}\n")

# Initialize components
print("Initializing components...")
detector = ONNXRetinaFaceDetector(
    'weights/retinaface_mobilenet025_coreml.onnx',
    use_coreml=False,  # CPU ONLY
    confidence_threshold=0.5
)
print(f"  Face detector: {detector.backend}")

pfld = CunjianPFLDDetector('weights/pfld_cunjian.onnx')
print(f"  PFLD: {pfld}")

pdm = PDMParser('In-the-wild_aligned_PDM_68.txt')
calc_params = CalcParams(pdm)
print(f"  CalcParams: ready")

aligner = OpenFace22FaceAligner('In-the-wild_aligned_PDM_68.txt', output_size=(112, 112))
print(f"  Aligner: {aligner}")

print("\nProfiling 50 iterations per component...\n")

# Component 1: Face Detection
print("[1/6] Face Detection...")
times = []
for i in range(50):
    t0 = time.perf_counter()
    dets, _ = detector.detect_faces(frame)
    times.append((time.perf_counter() - t0) * 1000)
print(f"  Average: {np.mean(times):.2f}ms (±{np.std(times):.2f}ms)")
print(f"  Min/Max: {np.min(times):.2f}ms / {np.max(times):.2f}ms")

bbox = dets[0][:4].astype(int) if len(dets) > 0 else np.array([0, 0, frame.shape[1], frame.shape[0]])

# Component 2: Landmarks
print("\n[2/6] Landmark Detection...")
times = []
for i in range(50):
    t0 = time.perf_counter()
    lms, _ = pfld.detect_landmarks(frame, bbox)
    times.append((time.perf_counter() - t0) * 1000)
print(f"  Average: {np.mean(times):.2f}ms (±{np.std(times):.2f}ms)")
print(f"  Min/Max: {np.min(times):.2f}ms / {np.max(times):.2f}ms")

# Component 3: CalcParams
print("\n[3/6] Pose Estimation (CalcParams)...")
times = []
conv = False
for i in range(50):
    t0 = time.perf_counter()
    result = calc_params.calc_params(lms.flatten())
    times.append((time.perf_counter() - t0) * 1000)
    if i == 0:  # Check return format on first iteration
        if len(result) == 3:
            pg, pl, conv = result
        else:
            pg, pl = result
            conv = True
print(f"  Average: {np.mean(times):.2f}ms (±{np.std(times):.2f}ms)")
print(f"  Min/Max: {np.min(times):.2f}ms / {np.max(times):.2f}ms")

if conv:
    scale, rx, ry, rz, tx, ty = pg[0], pg[1], pg[2], pg[3], pg[4], pg[5]
else:
    scale, rx, ry, rz, tx, ty = 1.0, 0, 0, 0, 0, 0

# Component 4: Alignment
print("\n[4/6] Face Alignment...")
times = []
for i in range(50):
    t0 = time.perf_counter()
    aligned = aligner.align_face(frame, lms, scale, rx, ry, rz, tx, ty)
    times.append((time.perf_counter() - t0) * 1000)
print(f"  Average: {np.mean(times):.2f}ms (±{np.std(times):.2f}ms)")
print(f"  Min/Max: {np.min(times):.2f}ms / {np.max(times):.2f}ms")

# Component 5: HOG
print("\n[5/6] HOG Extraction (PyFHOG)...")
times = []
for i in range(50):
    t0 = time.perf_counter()
    hog = pyfhog.extract(aligned)
    times.append((time.perf_counter() - t0) * 1000)
print(f"  Average: {np.mean(times):.2f}ms (±{np.std(times):.2f}ms)")
print(f"  Min/Max: {np.min(times):.2f}ms / {np.max(times):.2f}ms")
print(f"  HOG shape: {hog.shape}")

# Component 6: Geometric Features
print("\n[6/6] Geometric Features...")
times = []
for i in range(50):
    t0 = time.perf_counter()
    geom = pdm.calc_params(pl)
    times.append((time.perf_counter() - t0) * 1000)
print(f"  Average: {np.mean(times):.2f}ms (±{np.std(times):.2f}ms)")
print(f"  Min/Max: {np.min(times):.2f}ms / {np.max(times):.2f}ms")

print("\n" + "=" * 80)
print("✅ PROFILING COMPLETE")
print("=" * 80)
