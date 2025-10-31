#!/usr/bin/env python3
"""Diagnostic test - find where pipeline hangs"""

import sys
import time
import cv2

print("=" * 70)
print("DIAGNOSTIC TEST - CoreML Enabled")
print("=" * 70)

# Step 1: Import
print("\n[1/6] Importing...")
sys.path.insert(0, '../pyfhog/src')
from full_python_au_pipeline import FullPythonAUPipeline
print("✓ Imported")

# Step 2: Init
print("\n[2/6] Initializing pipeline...")
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
print(f"✓ Initialized in {time.time() - start:.2f}s")

# Step 3: Load video
print("\n[3/6] Loading video...")
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Failed to open video")
    sys.exit(1)
print("✓ Video opened")

# Step 4: Read frame
print("\n[4/6] Reading frame 0...")
ret, frame = cap.read()
if not ret:
    print("❌ Failed to read frame")
    sys.exit(1)
print(f"✓ Frame read: {frame.shape}")

# Step 5: Detect face
print("\n[5/6] Detecting face...")
start = time.time()
detections, confidences = pipeline.face_detector.detect_faces(frame)
detect_time = (time.time() - start) * 1000
print(f"✓ Detection: {detect_time:.1f}ms, found {len(detections)} faces")

if len(detections) == 0:
    print("⚠️ No faces detected, cannot continue")
    sys.exit(1)

# Step 6: Process full pipeline for this frame
print("\n[6/6] Processing through full pipeline...")
start = time.time()

# Get detection
detection = detections[0]
x1, y1, x2, y2 = detection

# Detect landmarks
print("  → Landmark detection...")
start_landmarks = time.time()
landmarks_68 = pipeline.landmark_detector.detect_landmarks(frame, detection)
print(f"     {(time.time() - start_landmarks) * 1000:.1f}ms")

if landmarks_68 is None:
    print("❌ Landmark detection failed")
    sys.exit(1)

# CalcParams
print("  → CalcParams...")
start_calc = time.time()
params_global, params_local, converged = pipeline.calc_params.calc_params(landmarks_68.flatten())
print(f"     {(time.time() - start_calc) * 1000:.1f}ms, converged={converged}")

if not converged:
    print("⚠️ CalcParams did not converge")
    sys.exit(1)

# Face alignment
print("  → Face alignment...")
start_align = time.time()
aligned_face, warped_landmarks = pipeline.face_aligner.align_face(frame, landmarks_68)
print(f"     {(time.time() - start_align) * 1000:.1f}ms")

if aligned_face is None:
    print("❌ Face alignment failed")
    sys.exit(1)

# HOG extraction
print("  → HOG extraction...")
start_hog = time.time()
hog_features = pipeline.extract_hog_features(aligned_face)
print(f"     {(time.time() - start_hog) * 1000:.1f}ms, shape={hog_features.shape}")

# Geometric features
print("  → Geometric features...")
start_geom = time.time()
geom_features = pipeline.extract_geometric_features(params_local, warped_landmarks)
print(f"     {(time.time() - start_geom) * 1000:.1f}ms, shape={geom_features.shape}")

# Combined features
print("  → Combining features...")
start_comb = time.time()
combined_features = pipeline.combine_features(hog_features, geom_features)
print(f"     {(time.time() - start_comb) * 1000:.1f}ms, shape={combined_features.shape}")

# AU prediction
print("  → AU prediction...")
start_au = time.time()
au_results = pipeline.predict_aus(combined_features, frame_index=0)
print(f"     {(time.time() - start_au) * 1000:.1f}ms, {len(au_results)} AUs")

total_time = (time.time() - start) * 1000
print(f"\n✓ Full pipeline: {total_time:.1f}ms")

print("\n" + "=" * 70)
print("✅ PIPELINE WORKING!")
print("=" * 70)
print(f"CoreML backend: {pipeline.face_detector.backend}")
print(f"Total frame processing: {total_time:.1f}ms")
print(f"Estimated throughput: {1000/total_time:.1f} FPS")
print("=" * 70)
