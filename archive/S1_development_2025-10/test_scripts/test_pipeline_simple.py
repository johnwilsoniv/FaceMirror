#!/usr/bin/env python3
"""Simple pipeline test on single small image (no video)"""

import cv2
import numpy as np
import time

print("=" * 70)
print("Simple Pipeline Test - Single Image")
print("=" * 70)

# Create test image (smaller)
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

print("\n[1/3] Importing pipeline...")
from full_python_au_pipeline import FullPythonAUPipeline

print("\n[2/3] Initializing...")
start = time.time()
pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='In-the-wild_aligned_PDM_68.txt',
    au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
    triangulation_file='tris_68_full.txt',
    use_calc_params=True,
    verbose=False
)
init_time = time.time() - start
print(f"✓ Initialized in {init_time:.2f}s")
print(f"  Backend: {pipeline.face_detector.backend}")

print("\n[3/3] Testing face detection on random image...")
start = time.time()
detections, confidences = pipeline.face_detector.detect_faces(test_image)
detect_time = (time.time() - start) * 1000
print(f"✓ Detection: {detect_time:.1f}ms, found {len(detections)} faces")

print("\n" + "=" * 70)
print("✅ PIPELINE INITIALIZES AND RUNS!")
print("=" * 70)
print(f"Backend: {pipeline.face_detector.backend}")
print(f"Init: {init_time:.2f}s")
print(f"Detection: {detect_time:.1f}ms")
print("=" * 70)
