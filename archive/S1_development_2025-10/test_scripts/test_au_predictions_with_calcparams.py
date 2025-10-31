#!/usr/bin/env python3
"""
Test AU predictions with CalcParams integration

Full pipeline with CalcParams for improved pose estimation:
1. Run CalcParams to optimize 3D pose from 2D landmarks
2. Use optimized pose for face alignment
3. Extract HOG with PyFHOG
4. Running median normalization
5. AU prediction with Python SVR
6. Compare to C++ baseline
"""

import sys
sys.path.insert(0, '../pyfhog/src')

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from openface22_face_aligner import OpenFace22FaceAligner
from triangulation_parser import TriangulationParser
from openface22_model_parser import OF22ModelParser
from histogram_median_tracker import DualHistogramMedianTracker
from pdm_parser import PDMParser
from calc_params import CalcParams
import pyfhog

print("=" * 80)
print("Python AU Prediction Pipeline Test WITH CalcParams")
print("=" * 80)

# Configuration
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
PDM_FILE = "In-the-wild_aligned_PDM_68.txt"
TRIS_FILE = "tris_68_full.txt"
MODELS_DIR = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors"

# Load components
print("\n1. Loading components...")

# Load PDM for CalcParams and geometric features
pdm = PDMParser(PDM_FILE)

# Create CalcParams instance
calc_params = CalcParams(pdm)
print("✓ CalcParams instance created")

# Load face aligner
aligner = OpenFace22FaceAligner(PDM_FILE)

# Load triangulation
triangulation = TriangulationParser(TRIS_FILE)

print("Loading AU models...")
parser = OF22ModelParser(MODELS_DIR)
au_models = parser.load_all_models(use_recommended=True, use_combined=True)
print(f"✓ Loaded {len(au_models)} AU models")
print(f"  Available AUs: {sorted(au_models.keys())}")

# Count dynamic vs static models
dynamic_aus = [au for au, model in au_models.items() if model['model_type'] == 'dynamic']
static_aus = [au for au, model in au_models.items() if model['model_type'] == 'static']
print(f"  Dynamic AUs ({len(dynamic_aus)}): {sorted(dynamic_aus)}")
print(f"  Static AUs ({len(static_aus)}): {sorted(static_aus)}")

# Create running median tracker
print("\n✓ Initializing running median tracker")
print("  HOG: 1000 bins, range [-0.005, 1.0]")
print("  Geometric: 10000 bins, range [-60.0, 60.0]")
median_tracker = DualHistogramMedianTracker(
    hog_dim=4464,
    geom_dim=238,
    hog_bins=1000,
    hog_min=-0.005,
    hog_max=1.0,
    geom_bins=10000,
    geom_min=-60.0,
    geom_max=60.0
)

# Load data
print("\n2. Loading test data...")
df = pd.read_csv(CSV_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"  Video: {frame_count} frames")
print(f"  CSV: {len(df)} rows")

# Test on 200 frames first (for speed)
test_frames = list(range(1, min(201, len(df) + 1)))

print(f"\n3. Testing on {len(test_frames)} frames with CalcParams...")
print("   This will take ~10-15 minutes (CalcParams optimization is slow)...")

python_au_predictions = {au: [] for au in au_models.keys()}
cpp_au_baseline = {au: [] for au in au_models.keys()}
frame_numbers = []

# Track iteration index for running median update frequency (every 2nd frame)
frame_idx = 0

calcparams_successes = 0
calcparams_failures = 0

for frame_num in test_frames:
    if frame_num % 50 == 0 or frame_num == test_frames[0]:
        print(f"  Processing frame {frame_num}...")

    row = df[df['frame'] == frame_num]
    if len(row) == 0:
        continue
    row = row.iloc[0]

    # Read frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()
    if not ret:
        continue

    # Get 2D landmarks from CSV
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_2d = np.stack([x, y], axis=1)  # (68, 2)

    try:
        # RUN CALCPARAMS to optimize pose
        params_global, params_local = calc_params.calc_params(landmarks_2d)

        if params_global is None or params_local is None:
            calcparams_failures += 1
            continue

        calcparams_successes += 1

        # Extract optimized pose parameters
        scale, rx, ry, rz, tx, ty = params_global

        # Use rz from CalcParams for alignment (instead of CSV p_rz)
        p_rz = rz
        pose_tx = tx
        pose_ty = ty

        # Align face with masking using CalcParams-optimized pose
        aligned = aligner.align_face(frame, landmarks_2d, pose_tx, pose_ty, p_rz,
                                     apply_mask=True, triangulation=triangulation)

        # Convert to RGB for PyFHOG
        aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

        # Extract HOG features
        hog_features = pyfhog.extract_fhog_features(aligned_rgb)

        # Use CalcParams-optimized params_local (instead of CSV p_0...p_33)
        # Reconstruct 3D PDM shape from CalcParams params_local
        shape_3d = pdm.mean_shape + pdm.princ_comp @ params_local.reshape(-1, 1)
        shape_3d_flat = shape_3d.flatten()  # 204 dims

        # Geometric features: 204 (3D shape) + 34 (params_local) = 238 dims
        geom_features = np.concatenate([shape_3d_flat, params_local.flatten()])

        # Combine HOG + geometric for complete feature vector
        complete_features = np.concatenate([hog_features, geom_features])

        if frame_num == test_frames[0]:
            print(f"    Feature breakdown: HOG={len(hog_features)}, Geom={len(geom_features)} (shape3d=204 + params=34), Total={len(complete_features)}")
            print(f"    Using CalcParams-optimized pose: rz={p_rz:.4f}, tx={pose_tx:.2f}, ty={pose_ty:.2f}")

        # Update running median tracker (update histogram every 2nd frame)
        update_histogram = (frame_idx % 2 == 1)
        median_tracker.update(hog_features, geom_features, update_histogram=update_histogram)

        # Get current running medians
        hog_median = median_tracker.get_hog_median()
        geom_median = median_tracker.get_geom_median()

        # Predict AUs
        for au_name, model_data in au_models.items():
            # For dynamic models, use running median normalized features
            # For static models, use original features
            if model_data['model_type'] == 'dynamic':
                # Normalize features by subtracting running median
                hog_normalized = hog_features - hog_median
                geom_normalized = geom_features - geom_median
                features_for_prediction = np.concatenate([hog_normalized, geom_normalized])
            else:
                # Static models use original features
                features_for_prediction = complete_features

            # Use parser's predict_au method
            prediction = parser.predict_au(features_for_prediction, model_data)

            python_au_predictions[au_name].append(prediction)

            # Get C++ baseline
            cpp_col = au_name  # Column name is already 'AU01_r' format
            if cpp_col in row:
                cpp_au_baseline[au_name].append(row[cpp_col])
            else:
                cpp_au_baseline[au_name].append(0.0)

        frame_numbers.append(frame_num)
        frame_idx += 1  # Increment for next iteration

    except Exception as e:
        print(f"  ✗ Error on frame {frame_num}: {e}")
        import traceback
        traceback.print_exc()
        calcparams_failures += 1
        continue

cap.release()

print(f"\n✓ Processed {len(frame_numbers)} frames successfully")
print(f"  CalcParams successes: {calcparams_successes}")
print(f"  CalcParams failures: {calcparams_failures}")

# Compute correlations
print("\n" + "=" * 80)
print("AU Correlation Analysis: Python WITH CalcParams vs C++ Baseline")
print("=" * 80)

correlations = {}
for au_name in sorted(au_models.keys()):
    python_vals = np.array(python_au_predictions[au_name])
    cpp_vals = np.array(cpp_au_baseline[au_name])

    if len(python_vals) == 0:
        continue

    # Compute correlation
    if np.std(python_vals) > 0 and np.std(cpp_vals) > 0:
        r = np.corrcoef(python_vals, cpp_vals)[0, 1]
    else:
        r = 0.0

    # Compute RMSE
    rmse = np.sqrt(np.mean((python_vals - cpp_vals) ** 2))

    # Compute variance for diagnostics
    py_std = np.std(python_vals)
    cpp_std = np.std(cpp_vals)

    correlations[au_name] = {'r': r, 'rmse': rmse, 'py_std': py_std, 'cpp_std': cpp_std}

    print(f"  {au_name}: r={r:.4f}, RMSE={rmse:.4f}, Py_σ={py_std:.4f}, C++_σ={cpp_std:.4f}")

# Summary statistics
print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)

rs = [c['r'] for c in correlations.values() if not np.isnan(c['r'])]

if len(rs) == 0:
    print("  ✗ No valid correlations - check for errors above")
else:
    mean_r = np.mean(rs)
    min_r = np.min(rs)
    max_r = np.max(rs)

    print(f"  Mean correlation: {mean_r:.4f}")
    print(f"  Min correlation:  {min_r:.4f}")
    print(f"  Max correlation:  {max_r:.4f}")

    print("\nInterpretation:")
    if mean_r > 0.95:
        print("  ✓ EXCELLENT - CalcParams brings Python pipeline to C++ level!")
        print("  → Mission accomplished! CalcParams was the missing piece!")
    elif mean_r > 0.90:
        print("  ✓ VERY GOOD - CalcParams improved Python pipeline significantly")
        print("  → This is production-ready quality")
    elif mean_r > 0.85:
        print("  ✓ GOOD - CalcParams helped improve results")
        print("  → Small remaining gap, but acceptable for most uses")
    elif mean_r > 0.80:
        print("  ~ ACCEPTABLE - CalcParams provided minor improvement")
        print("  → May need additional tuning or investigation")
    else:
        print("  ⚠ CalcParams did not improve results significantly")
        print("  → Need to investigate why")

print("\n" + "=" * 80)
print(f"PyFHOG feature count: {len(hog_features)} (expected ~4464)")
print("=" * 80)
