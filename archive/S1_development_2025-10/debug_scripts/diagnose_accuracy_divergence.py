#!/usr/bin/env python3
"""
Diagnose Accuracy Divergence - Find where Python pipeline diverges from C++

Strategy:
1. Compare intermediate outputs at each pipeline stage
2. Identify where correlation drops below 0.99
3. Focus on dynamic AU computation path

Test: 100 frames from validation video
Compare Python outputs vs C++ CSV ground truth
"""

import sys
import pandas as pd
import numpy as np
import cv2

sys.path.insert(0, '../pyfhog/src')

from openface22_face_aligner import OpenFace22FaceAligner
from pdm_parser import PDMParser
from triangulation_parser import TriangulationParser
import pyfhog

# Cython or Python running median
try:
    from cython_histogram_median import DualHistogramMedianTrackerCython as DualHistogramMedianTracker
    print("✓ Using Cython running median (260x optimized)")
except ImportError:
    from histogram_median_tracker import DualHistogramMedianTracker
    print("⚠ Using Python running median (fallback)")

from openface22_model_parser import OF22ModelParser

print("\n" + "=" * 80)
print("ACCURACY DIVERGENCE DIAGNOSIS")
print("=" * 80)

# Load C++ ground truth
csv_path = "of22_validation/IMG_0942_left_mirrored.csv"
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

print(f"\n1. Loading C++ ground truth from CSV...")
df_cpp = pd.read_csv(csv_path)
print(f"   ✓ Loaded {len(df_cpp)} frames")

# Initialize Python components
print(f"\n2. Initializing Python components...")

aligner = OpenFace22FaceAligner('In-the-wild_aligned_PDM_68.txt')
pdm = PDMParser('In-the-wild_aligned_PDM_68.txt')
triangulation = TriangulationParser('tris_68_full.txt')
median_tracker = DualHistogramMedianTracker()
model_parser = OF22ModelParser('/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors')
au_models = model_parser.load_all_models(use_recommended=True, use_combined=True)

print(f"   ✓ All components initialized")

# Open video
cap = cv2.VideoCapture(video_path)

# Track divergences at each stage
divergences = {
    'frame': [],
    'hog_correlation': [],
    'geom_correlation': [],
    'hog_median_correlation': [],
    'geom_median_correlation': [],
    'static_au_mean_corr': [],
    'dynamic_au_mean_corr': [],
}

print(f"\n3. Processing 100 frames and tracking divergences...")
print(f"   (This will take a minute...)")

frame_idx = 0
test_frames = 100

# Store all Python outputs for final correlation
py_aus = {au: [] for au in ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r',
                              'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r',
                              'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r',
                              'AU26_r', 'AU45_r']}

while frame_idx < test_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Get C++ ground truth for this frame
    cpp_row = df_cpp.iloc[frame_idx]

    # Extract landmarks from CSV (68 points)
    landmark_cols = [f'x_{i}' for i in range(68)] + [f'y_{i}' for i in range(68)]
    landmarks = cpp_row[landmark_cols].values
    landmarks_68 = np.column_stack([landmarks[:68], landmarks[68:]])

    # Extract pose from CSV
    p_scale = cpp_row['p_scale']
    p_rx = cpp_row['p_rx']
    p_ry = cpp_row['p_ry']
    p_rz = cpp_row['p_rz']
    p_tx = cpp_row['p_tx']
    p_ty = cpp_row['p_ty']
    params_local = cpp_row[[f'p_{i}' for i in range(34)]].values

    # === STAGE 1: Face Alignment ===
    aligned = aligner.align_face(
        frame, landmarks_68, p_tx, p_ty, p_rz,
        apply_mask=True, triangulation=triangulation
    )
    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

    # === STAGE 2: HOG Features ===
    hog_features = pyfhog.extract_fhog_features(aligned_rgb)

    # Compare to C++ HOG (if we had it - skip for now)
    # hog_correlation = ...

    # === STAGE 3: Geometric Features ===
    shape_3d = pdm.mean_shape + pdm.princ_comp @ params_local.reshape(-1, 1)
    shape_3d_flat = shape_3d.flatten()
    geom_features = np.concatenate([shape_3d_flat, params_local])

    # === STAGE 4: Running Median Update ===
    update_histogram = (frame_idx % 2 == 1)
    median_tracker.update(hog_features, geom_features, update_histogram=update_histogram)

    hog_median = median_tracker.get_hog_median()
    geom_median = median_tracker.get_geom_median()

    # === STAGE 5: AU Prediction ===
    for au_name in py_aus.keys():
        if au_name not in au_models:
            py_aus[au_name].append(0.0)
            continue

        model = au_models[au_name]

        # Static or dynamic?
        if model['model_type'] == 'dynamic':
            # Normalize by running median
            hog_norm = hog_features - hog_median
            geom_norm = geom_features - geom_median
            full_vector = np.concatenate([hog_norm, geom_norm])
        else:
            # Static - use original features
            full_vector = np.concatenate([hog_features, geom_features])

        # SVR prediction
        centered = full_vector - model['means'].flatten()
        score = np.dot(centered, model['support_vectors'].flatten()) + model['bias']

        # Apply cutoff for dynamic models
        if model['model_type'] == 'dynamic' and score < model['cutoff']:
            prediction = 0.0
        else:
            prediction = score

        py_aus[au_name].append(prediction)

    frame_idx += 1

cap.release()

print(f"\n   ✓ Processed {frame_idx} frames")

# === FINAL ANALYSIS ===
print(f"\n4. Computing correlations with C++ outputs...")

# Convert Python AUs to arrays
for au in py_aus:
    py_aus[au] = np.array(py_aus[au])

# Compute correlations
correlations = {}
for au in py_aus.keys():
    cpp_vals = df_cpp[au].values[:test_frames]
    py_vals = py_aus[au]

    if len(cpp_vals) != len(py_vals):
        print(f"⚠ Length mismatch for {au}: C++={len(cpp_vals)}, Py={len(py_vals)}")
        continue

    # Remove any NaN values
    mask = ~(np.isnan(cpp_vals) | np.isnan(py_vals))
    cpp_clean = cpp_vals[mask]
    py_clean = py_vals[mask]

    if len(cpp_clean) > 0:
        corr = np.corrcoef(cpp_clean, py_clean)[0, 1]
        correlations[au] = corr
    else:
        correlations[au] = 0.0

# === REPORT ===
print(f"\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# Separate by model type
static_aus = ['AU04_r', 'AU06_r', 'AU07_r', 'AU10_r', 'AU12_r', 'AU14_r']
dynamic_aus = ['AU01_r', 'AU02_r', 'AU05_r', 'AU09_r', 'AU15_r', 'AU17_r',
               'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

print(f"\nSTATIC AUs (no running median):")
static_corrs = []
for au in static_aus:
    if au in correlations:
        r = correlations[au]
        static_corrs.append(r)
        status = "✅" if r > 0.95 else ("✓" if r > 0.90 else "⚠️")
        print(f"  {au}: r={r:.4f} {status}")

if static_corrs:
    print(f"\n  Mean static AU correlation: {np.mean(static_corrs):.4f}")

print(f"\nDYNAMIC AUs (with running median normalization):")
dynamic_corrs = []
problem_aus = []

for au in dynamic_aus:
    if au in correlations:
        r = correlations[au]
        dynamic_corrs.append(r)
        status = "✅" if r > 0.95 else ("✓" if r > 0.85 else "⚠️")
        print(f"  {au}: r={r:.4f} {status}")

        if r < 0.70:
            problem_aus.append(au)

if dynamic_corrs:
    print(f"\n  Mean dynamic AU correlation: {np.mean(dynamic_corrs):.4f}")

if correlations:
    all_corrs = list(correlations.values())
    print(f"\n  Overall mean correlation: {np.mean(all_corrs):.4f}")

# === VARIANCE ANALYSIS ===
print(f"\n" + "=" * 80)
print("VARIANCE ANALYSIS (Problem AUs)")
print("=" * 80)

if problem_aus:
    print(f"\nProblem AUs with r < 0.70:")
    for au in problem_aus:
        cpp_vals = df_cpp[au].values[:test_frames]
        py_vals = py_aus[au]

        cpp_std = np.std(cpp_vals)
        py_std = np.std(py_vals)
        ratio = (py_std / cpp_std * 100) if cpp_std > 0 else 0

        print(f"\n  {au}:")
        print(f"    Correlation:    r={correlations[au]:.4f}")
        print(f"    C++ std:        {cpp_std:.4f}")
        print(f"    Python std:     {py_std:.4f}")
        print(f"    Variance ratio: {ratio:.1f}%")

        if ratio > 150:
            print(f"    🔴 OVER-PREDICTION: Python variance {ratio:.0f}% of C++")
        elif ratio < 50:
            print(f"    🔴 UNDER-PREDICTION: Python variance {ratio:.0f}% of C++")
else:
    print(f"\n  ✅ No problem AUs found (all r > 0.70)")

# === RECOMMENDATIONS ===
print(f"\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if np.mean(all_corrs) > 0.95:
    print(f"\n✅ EXCELLENT! Overall correlation {np.mean(all_corrs):.4f} > 0.95")
    print(f"   Pipeline is matching C++ very well!")
elif np.mean(all_corrs) > 0.90:
    print(f"\n✓ GOOD! Overall correlation {np.mean(all_corrs):.4f} > 0.90")
    print(f"   Minor improvements possible:")
    if static_corrs and np.mean(static_corrs) < 0.95:
        print(f"   - Check face alignment (static AUs not perfect)")
    if dynamic_corrs and np.mean(dynamic_corrs) < 0.90:
        print(f"   - Check running median computation")
else:
    print(f"\n⚠️ Correlation {np.mean(all_corrs):.4f} needs improvement")

    if static_corrs and np.mean(static_corrs) < 0.90:
        print(f"\n🔴 Priority 1: Fix Static AUs (r={np.mean(static_corrs):.4f})")
        print(f"   Issue likely in: Face alignment or HOG extraction")

    if dynamic_corrs and np.mean(dynamic_corrs) < 0.80:
        print(f"\n🔴 Priority 2: Fix Dynamic AUs (r={np.mean(dynamic_corrs):.4f})")
        print(f"   Issue likely in: Running median normalization")

        if problem_aus:
            print(f"\n   Specific problem AUs: {', '.join(problem_aus)}")
            print(f"   Next step: Compare running median values frame-by-frame")

print(f"\n" + "=" * 80)
print("Diagnosis complete!")
print("=" * 80)
