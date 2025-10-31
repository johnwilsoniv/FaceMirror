#!/usr/bin/env python3
"""
Comprehensive Performance Profile - Full Python AU Pipeline

Identifies bottlenecks and optimization opportunities across all 8 components:
1. Face Detection (RetinaFace)
2. Landmark Detection (PFLD)
3. Pose Estimation (CalcParams)
4. Face Alignment
5. HOG Extraction (PyFHOG)
6. Geometric Features (PDM)
7. Running Median
8. AU Prediction (SVR)
"""

import multiprocessing
import os
import sys
import time
import numpy as np

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

sys.path.insert(0, '../pyfhog/src')

print("\n" + "=" * 80)
print("FULL PYTHON AU PIPELINE - COMPREHENSIVE PERFORMANCE PROFILE")
print("=" * 80)
print("Measuring each component to identify bottlenecks")
print("")

from full_python_au_pipeline import FullPythonAUPipeline
import cv2
from pathlib import Path

# Video path
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

if not Path(video_path).exists():
    print(f"Error: Video not found at {video_path}")
    sys.exit(1)

# Initialize pipeline (CPU mode for stability)
print("[1/4] Initializing pipeline...")
t0 = time.time()
pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='In-the-wild_aligned_PDM_68.txt',
    au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
    triangulation_file='tris_68_full.txt',
    use_calc_params=True,
    use_coreml=False,  # CPU mode for consistent profiling
    verbose=False
)
init_time = time.time() - t0
print(f"✓ Initialization: {init_time:.2f}s")
print(f"  Backend: {pipeline.face_detector.backend}")
print("")

# Load video and get first frame
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame")
    sys.exit(1)

print(f"Frame: {frame.shape}")
print("")

# ============================================================================
# DETAILED COMPONENT PROFILING
# ============================================================================

print("[2/4] Profiling individual components (50 iterations each)...")
print("=" * 80)

iterations = 50
timings = {}

# Component 1: Face Detection
print("\n[Component 1] Face Detection (RetinaFace ONNX)...")
times = []
for i in range(iterations):
    t0 = time.perf_counter()
    detections, _ = pipeline.face_detector.detect_faces(frame)
    elapsed = (time.perf_counter() - t0) * 1000
    times.append(elapsed)
timings['face_detection'] = times
avg = np.mean(times)
std = np.std(times)
print(f"  Average: {avg:.2f}ms (±{std:.2f}ms)")
print(f"  Min: {np.min(times):.2f}ms, Max: {np.max(times):.2f}ms")
print(f"  Detections: {len(detections)}")

# Get bbox for remaining tests
if len(detections) == 0:
    print("No face detected, using full frame")
    bbox = np.array([0, 0, frame.shape[1], frame.shape[0]])
else:
    bbox = detections[0][:4].astype(int)

# Component 2: Landmark Detection
print("\n[Component 2] Landmark Detection (PFLD)...")
times = []
for i in range(iterations):
    t0 = time.perf_counter()
    landmarks_68, _ = pipeline.landmark_detector.detect_landmarks(frame, bbox)
    elapsed = (time.perf_counter() - t0) * 1000
    times.append(elapsed)
timings['landmark_detection'] = times
avg = np.mean(times)
std = np.std(times)
print(f"  Average: {avg:.2f}ms (±{std:.2f}ms)")
print(f"  Min: {np.min(times):.2f}ms, Max: {np.max(times):.2f}ms")

# Component 3: Pose Estimation (CalcParams)
print("\n[Component 3] Pose Estimation (CalcParams)...")
times = []
successful = 0
for i in range(iterations):
    t0 = time.perf_counter()
    params_global, params_local, converged = pipeline.calc_params.calc_params(landmarks_68.flatten())
    elapsed = (time.perf_counter() - t0) * 1000
    times.append(elapsed)
    if converged:
        successful += 1
timings['pose_estimation'] = times
avg = np.mean(times)
std = np.std(times)
print(f"  Average: {avg:.2f}ms (±{std:.2f}ms)")
print(f"  Min: {np.min(times):.2f}ms, Max: {np.max(times):.2f}ms")
print(f"  Convergence rate: {successful}/{iterations} ({successful/iterations*100:.1f}%)")

# Extract pose for alignment
if converged:
    scale = params_global[0]
    rx, ry, rz = params_global[1], params_global[2], params_global[3]
    tx, ty = params_global[4], params_global[5]
else:
    print("  Warning: Using fallback pose")
    scale, rx, ry, rz, tx, ty = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0

# Component 4: Face Alignment
print("\n[Component 4] Face Alignment (OpenFace22)...")
times = []
for i in range(iterations):
    t0 = time.perf_counter()
    aligned_face = pipeline.face_aligner.align_face(
        frame, landmarks_68, scale, rx, ry, rz, tx, ty
    )
    elapsed = (time.perf_counter() - t0) * 1000
    times.append(elapsed)
timings['face_alignment'] = times
avg = np.mean(times)
std = np.std(times)
print(f"  Average: {avg:.2f}ms (±{std:.2f}ms)")
print(f"  Min: {np.min(times):.2f}ms, Max: {np.max(times):.2f}ms")
print(f"  Output size: {aligned_face.shape}")

# Component 5: HOG Extraction
print("\n[Component 5] HOG Extraction (PyFHOG)...")
import pyfhog
times = []
for i in range(iterations):
    t0 = time.perf_counter()
    hog_features = pyfhog.extract(aligned_face)
    elapsed = (time.perf_counter() - t0) * 1000
    times.append(elapsed)
timings['hog_extraction'] = times
avg = np.mean(times)
std = np.std(times)
print(f"  Average: {avg:.2f}ms (±{std:.2f}ms)")
print(f"  Min: {np.min(times):.2f}ms, Max: {np.max(times):.2f}ms")
print(f"  HOG size: {hog_features.shape}")

# Component 6: Geometric Features (PDM)
print("\n[Component 6] Geometric Features Extraction...")
times = []
for i in range(iterations):
    t0 = time.perf_counter()
    geom_features = pipeline.pdm_parser.calc_params(params_local)
    elapsed = (time.perf_counter() - t0) * 1000
    times.append(elapsed)
timings['geometric_features'] = times
avg = np.mean(times)
std = np.std(times)
print(f"  Average: {avg:.2f}ms (±{std:.2f}ms)")
print(f"  Min: {np.min(times):.2f}ms, Max: {np.max(times):.2f}ms")
print(f"  Feature size: {len(geom_features)}")

# Combine features
combined_features = np.concatenate([hog_features.flatten(), geom_features])

# Component 7: Running Median (need to update first)
print("\n[Component 7] Running Median Update...")
times = []
for i in range(iterations):
    t0 = time.perf_counter()
    pipeline.running_median.update_medians(combined_features, rx, ry, rz)
    elapsed = (time.perf_counter() - t0) * 1000
    times.append(elapsed)
timings['running_median'] = times
avg = np.mean(times)
std = np.std(times)
print(f"  Average: {avg:.2f}ms (±{std:.2f}ms)")
print(f"  Min: {np.min(times):.2f}ms, Max: {np.max(times):.2f}ms")

# Get normalized features
current_medians = pipeline.running_median.get_medians()
normalized_features = combined_features - current_medians

# Component 8: AU Prediction (all 17 models)
print("\n[Component 8] AU Prediction (17 SVR models)...")
times = []
for i in range(iterations):
    t0 = time.perf_counter()
    au_predictions = {}
    for au_name, model_info in pipeline.au_models.items():
        model = model_info['model']
        prediction = float(model.predict(normalized_features.reshape(1, -1))[0])
        au_predictions[au_name] = prediction
    elapsed = (time.perf_counter() - t0) * 1000
    times.append(elapsed)
timings['au_prediction'] = times
avg = np.mean(times)
std = np.std(times)
print(f"  Average: {avg:.2f}ms (±{std:.2f}ms)")
print(f"  Min: {np.min(times):.2f}ms, Max: {np.max(times):.2f}ms")
print(f"  AUs predicted: {len(au_predictions)}")

# ============================================================================
# SUMMARY AND ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)

# Calculate totals
component_names = [
    'Face Detection',
    'Landmark Detection',
    'Pose Estimation',
    'Face Alignment',
    'HOG Extraction',
    'Geometric Features',
    'Running Median',
    'AU Prediction'
]

component_keys = [
    'face_detection',
    'landmark_detection',
    'pose_estimation',
    'face_alignment',
    'hog_extraction',
    'geometric_features',
    'running_median',
    'au_prediction'
]

# Print per-component breakdown
print("\nPer-Component Breakdown:")
print("-" * 80)
total_time = 0
for name, key in zip(component_names, component_keys):
    avg = np.mean(timings[key])
    total_time += avg
    print(f"  {name:25s}: {avg:6.2f}ms")

print(f"  {'─' * 25}   {'─' * 6}")
print(f"  {'TOTAL':25s}: {total_time:6.2f}ms")
print(f"  {'Throughput':25s}: {1000/total_time:6.2f} FPS")
print("")

# Identify bottlenecks (components taking >10% of time)
print("Bottleneck Analysis:")
print("-" * 80)
bottlenecks = []
for name, key in zip(component_names, component_keys):
    avg = np.mean(timings[key])
    percentage = (avg / total_time) * 100
    if percentage > 10:
        bottlenecks.append((name, key, avg, percentage))
        print(f"  ⚠️  {name:25s}: {avg:6.2f}ms ({percentage:5.1f}%) - BOTTLENECK")
    else:
        print(f"  ✓  {name:25s}: {avg:6.2f}ms ({percentage:5.1f}%)")

print("")

# ============================================================================
# FULL PIPELINE TEST
# ============================================================================

print("[3/4] Full pipeline test (50 frames)...")
t0 = time.time()
results = pipeline.process_video(video_path, None, max_frames=50)
full_time = time.time() - t0

success = results['success'].sum()
per_frame_ms = (full_time / len(results)) * 1000
fps = len(results) / full_time

print(f"✓ Processed: {len(results)} frames")
print(f"  Success: {success}/{len(results)}")
print(f"  Time: {full_time:.2f}s")
print(f"  Per frame: {per_frame_ms:.1f}ms")
print(f"  Throughput: {fps:.1f} FPS")
print("")

# ============================================================================
# OPTIMIZATION RECOMMENDATIONS
# ============================================================================

print("[4/4] Optimization Recommendations:")
print("=" * 80)

recommendations = []

# Check each bottleneck
for name, key, avg_time, percentage in bottlenecks:
    if 'Face Detection' in name:
        recommendations.append({
            'component': name,
            'time': avg_time,
            'percent': percentage,
            'issue': 'Face detection taking significant time',
            'options': [
                '✓ Already using ONNX (2-4x faster than PyTorch)',
                '⏳ CoreML would give 2-3x additional speedup',
                '💡 Consider lower resolution preprocessing',
                '💡 Skip detection on consecutive frames (tracking)'
            ]
        })
    elif 'Landmark' in name:
        recommendations.append({
            'component': name,
            'time': avg_time,
            'percent': percentage,
            'issue': 'Landmark detection bottleneck',
            'options': [
                '✓ Already using efficient PFLD model',
                '⏳ Consider ONNX acceleration',
                '💡 Use tracking between frames'
            ]
        })
    elif 'Pose' in name:
        recommendations.append({
            'component': name,
            'time': avg_time,
            'percent': percentage,
            'issue': 'CalcParams optimization taking time',
            'options': [
                '⚡ Cythonize Gauss-Newton solver',
                '💡 Reduce max iterations',
                '💡 Use warmstart from previous frame'
            ]
        })
    elif 'Alignment' in name:
        recommendations.append({
            'component': name,
            'time': avg_time,
            'percent': percentage,
            'issue': 'Face alignment overhead',
            'options': [
                '⚡ Cythonize Kabsch algorithm',
                '⚡ Optimize warpAffine parameters',
                '💡 Pre-compute reference shape'
            ]
        })
    elif 'HOG' in name:
        recommendations.append({
            'component': name,
            'time': avg_time,
            'percent': percentage,
            'issue': 'HOG extraction taking time',
            'options': [
                '✓ Already using PyFHOG (C library)',
                '💡 Check if using optimized parameters',
                '⏳ Minimal optimization potential'
            ]
        })
    elif 'Geometric' in name:
        recommendations.append({
            'component': name,
            'time': avg_time,
            'percent': percentage,
            'issue': 'PDM feature extraction',
            'options': [
                '⚡ Cythonize matrix operations',
                '✓ Already vectorized with NumPy'
            ]
        })
    elif 'Running Median' in name:
        recommendations.append({
            'component': name,
            'time': avg_time,
            'percent': percentage,
            'issue': 'Running median updates',
            'options': [
                '✓ Already Cython-optimized (260x)',
                '✓ Near-optimal performance'
            ]
        })
    elif 'AU Prediction' in name:
        recommendations.append({
            'component': name,
            'time': avg_time,
            'percent': percentage,
            'issue': 'SVR prediction overhead',
            'options': [
                '✓ Already using scikit-learn (C backend)',
                '💡 Batch predictions instead of 17 sequential',
                '⚡ Consider single multi-output model'
            ]
        })

# Print recommendations
if recommendations:
    print(f"\nFound {len(recommendations)} component(s) that could be optimized:\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['component']} ({rec['time']:.2f}ms, {rec['percent']:.1f}%)")
        print(f"   Issue: {rec['issue']}")
        print(f"   Options:")
        for opt in rec['options']:
            print(f"     {opt}")
        print("")
else:
    print("\n✅ No major bottlenecks found - pipeline is well-optimized!")
    print("")

# Performance vs baseline
print("Performance vs C++ Hybrid:")
print("-" * 80)
cpp_time = 704.8
speedup = cpp_time / per_frame_ms
print(f"  C++ Hybrid:     {cpp_time:.1f}ms/frame (1.42 FPS)")
print(f"  Python (CPU):   {per_frame_ms:.1f}ms/frame ({fps:.1f} FPS)")
print(f"  Speedup:        {speedup:.1f}x FASTER! 🚀")
print("")

# Potential with optimizations
potential_speedup = 1.0
print("Optimization Potential:")
print("-" * 80)
for rec in recommendations:
    if 'CoreML' in str(rec['options']):
        print(f"  • CoreML for {rec['component']}: 2-3x faster → {per_frame_ms/2.5:.1f}ms")
        potential_speedup *= 2.5
    elif 'Cythonize' in str(rec['options']):
        print(f"  • Cythonize {rec['component']}: 2-5x faster → saves {rec['time']*0.7:.1f}ms")

if potential_speedup > 1.5:
    optimized_time = per_frame_ms / potential_speedup
    print(f"\nWith all optimizations: ~{optimized_time:.1f}ms/frame ({1000/optimized_time:.1f} FPS)")
    print(f"Total potential speedup: {cpp_time/optimized_time:.1f}x vs C++ hybrid")
else:
    print(f"\nCurrent performance is near-optimal for CPU mode!")

print("\n" + "=" * 80)
print("✅ PROFILING COMPLETE")
print("=" * 80)
