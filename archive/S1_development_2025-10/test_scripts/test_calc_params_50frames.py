#!/usr/bin/env python3
"""
Test CalcParams on 50 frames for statistical significance

Measures correlation coefficient for each parameter to determine
how close we are to 99% target.
"""

import numpy as np
import pandas as pd
import cv2
from calc_params import CalcParams
from pdm_parser import PDMParser
from scipy.stats import pearsonr

print("=" * 80)
print("CalcParams 50-Frame Statistical Test")
print("=" * 80)

# Configuration
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
PDM_FILE = "In-the-wild_aligned_PDM_68.txt"

# Load PDM
print("\n1. Loading PDM...")
pdm = PDMParser(PDM_FILE)
calc_params = CalcParams(pdm)
print(f"✓ CalcParams ready")

# Load test data
print("\n2. Loading test data...")
df = pd.read_csv(CSV_PATH)
print(f"  CSV: {len(df)} rows")

# Test on 50 evenly spaced frames
test_frames = np.linspace(1, 1100, 50, dtype=int)

print(f"\n3. Testing on {len(test_frames)} frames...")
print("-" * 80)

# Storage for results
cpp_params_global_all = []
py_params_global_all = []
cpp_params_local_all = []
py_params_local_all = []

for i, frame_num in enumerate(test_frames):
    row = df[df['frame'] == frame_num]
    if len(row) == 0:
        continue
    row = row.iloc[0]

    # Get 2D landmarks
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_2d = np.stack([x, y], axis=1)

    # Get C++ baseline
    cpp_params_global = np.array([
        row['p_scale'], row['p_rx'], row['p_ry'],
        row['p_rz'], row['p_tx'], row['p_ty']
    ], dtype=np.float32)

    cpp_params_local_cols = [f'p_{i}' for i in range(34)]
    cpp_params_local = row[cpp_params_local_cols].values.astype(np.float32)

    # Run Python CalcParams
    try:
        params_global, params_local = calc_params.calc_params(landmarks_2d)

        if params_global is not None and params_local is not None:
            cpp_params_global_all.append(cpp_params_global)
            py_params_global_all.append(params_global)
            cpp_params_local_all.append(cpp_params_local)
            py_params_local_all.append(params_local)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(test_frames)} frames...")

    except Exception as e:
        print(f"  Frame {frame_num}: ERROR - {e}")

# Convert to arrays
cpp_global = np.array(cpp_params_global_all)  # (n, 6)
py_global = np.array(py_params_global_all)    # (n, 6)
cpp_local = np.array(cpp_params_local_all)    # (n, 34)
py_local = np.array(py_params_local_all)      # (n, 34)

print("\n" + "=" * 80)
print("RESULTS: GLOBAL PARAMETERS")
print("=" * 80)

param_names = ['scale', 'rx', 'ry', 'rz', 'tx', 'ty']
print("\nParameter-by-parameter analysis:")
print(f"{'Param':<8} {'C++ Mean':<12} {'Py Mean':<12} {'RMSE':<12} {'r':<8} {'Status':<8}")
print("-" * 80)

global_correlations = []
for i, name in enumerate(param_names):
    cpp_vals = cpp_global[:, i]
    py_vals = py_global[:, i]

    rmse = np.sqrt(np.mean((cpp_vals - py_vals) ** 2))
    r, _ = pearsonr(cpp_vals, py_vals)
    global_correlations.append(r)

    status = "✅" if r > 0.99 else "⚠️" if r > 0.95 else "❌"

    print(f"{name:<8} {np.mean(cpp_vals):<12.6f} {np.mean(py_vals):<12.6f} "
          f"{rmse:<12.6f} {r:<8.4f} {status:<8}")

mean_global_r = np.mean(global_correlations)
print("-" * 80)
print(f"{'OVERALL':<8} {'':<12} {'':<12} {'':<12} {mean_global_r:<8.4f} "
      f"{'✅' if mean_global_r > 0.99 else '⚠️' if mean_global_r > 0.98 else '❌'}")

print("\n" + "=" * 80)
print("RESULTS: LOCAL PARAMETERS (Shape)")
print("=" * 80)

print("\nLocal parameter correlations:")
local_correlations = []
poor_performers = []

for i in range(34):
    cpp_vals = cpp_local[:, i]
    py_vals = py_local[:, i]

    r, _ = pearsonr(cpp_vals, py_vals)
    local_correlations.append(r)

    if r < 0.90:
        poor_performers.append((i, r))

mean_local_r = np.mean(local_correlations)

print(f"Mean correlation: {mean_local_r:.4f}")
print(f"Min correlation: {np.min(local_correlations):.4f}")
print(f"Max correlation: {np.max(local_correlations):.4f}")
print(f"Std correlation: {np.std(local_correlations):.4f}")

if poor_performers:
    print(f"\nPoor performers (r < 0.90):")
    for i, r in poor_performers:
        print(f"  p_{i}: r = {r:.4f}")
else:
    print("\n✅ All local parameters > 0.90 correlation")

print("\n" + "=" * 80)
print("FINAL ASSESSMENT")
print("=" * 80)

overall_mean_r = (mean_global_r + mean_local_r) / 2

print(f"\nGlobal params mean r: {mean_global_r:.4f}")
print(f"Local params mean r:  {mean_local_r:.4f}")
print(f"Overall mean r:       {overall_mean_r:.4f}")

if overall_mean_r >= 0.99:
    print("\n🎯 TARGET ACHIEVED: >99% correlation! 💧💧💧")
elif overall_mean_r >= 0.98:
    print("\n✓ Very close to target (98%+), minor improvements needed")
elif overall_mean_r >= 0.95:
    print("\n⚠️ Good but not excellent (95-98%), significant improvements needed")
else:
    print("\n❌ Below target (<95%), major work needed")

print("\n" + "=" * 80)
