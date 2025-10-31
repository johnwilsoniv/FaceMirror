#!/usr/bin/env python3
"""
Diagnose why CalcParams params differ from CSV params

Compare CalcParams-generated params_local with CSV p_0...p_33
to understand why variance collapses in the AU pipeline.
"""

import numpy as np
import pandas as pd
from pdm_parser import PDMParser
from calc_params import CalcParams

print("=" * 80)
print("CalcParams vs CSV Params Diagnostic")
print("=" * 80)

# Configuration
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
PDM_FILE = "In-the-wild_aligned_PDM_68.txt"

# Load PDM
print("\n1. Loading PDM...")
pdm = PDMParser(PDM_FILE)

# Create CalcParams instance
calc_params = CalcParams(pdm)
print("✓ CalcParams instance created")

# Load CSV
print("\n2. Loading CSV...")
df = pd.read_csv(CSV_PATH)
print(f"  {len(df)} frames")

# Test on first 10 frames
test_frames = list(range(1, 11))

print(f"\n3. Comparing CalcParams vs CSV params on {len(test_frames)} frames...")
print("-" * 80)

all_csv_params = []
all_calcparams_params = []

for frame_num in test_frames:
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

    # Get CSV params
    csv_params_local_cols = [f'p_{i}' for i in range(34)]
    csv_params_local = row[csv_params_local_cols].values.astype(np.float32)

    # Run CalcParams
    try:
        params_global, params_local = calc_params.calc_params(landmarks_2d)

        if params_global is not None and params_local is not None:
            # Compare
            diff = params_local - csv_params_local
            rmse = np.sqrt(np.mean(diff ** 2))
            max_diff = np.max(np.abs(diff))

            all_csv_params.append(csv_params_local)
            all_calcparams_params.append(params_local)

            print(f"\nFrame {frame_num}:")
            print(f"  params_local RMSE: {rmse:.6f}")
            print(f"  Max absolute diff: {max_diff:.6f}")
            print(f"  CSV params range: [{csv_params_local.min():.3f}, {csv_params_local.max():.3f}]")
            print(f"  CalcParams range: [{params_local.min():.3f}, {params_local.max():.3f}]")
            print(f"  CSV std: {csv_params_local.std():.6f}")
            print(f"  CalcParams std: {params_local.std():.6f}")

            # Show first 5 params for comparison
            print(f"  First 5 params:")
            for i in range(min(5, len(csv_params_local))):
                print(f"    p_{i}: CSV={csv_params_local[i]:8.4f}, CalcParams={params_local[i]:8.4f}, diff={diff[i]:8.4f}")

    except Exception as e:
        print(f"\nFrame {frame_num}: Error - {e}")
        import traceback
        traceback.print_exc()

# Analyze variance across frames
print("\n" + "=" * 80)
print("Variance Analysis Across Frames")
print("=" * 80)

if len(all_csv_params) > 0 and len(all_calcparams_params) > 0:
    csv_params_array = np.array(all_csv_params)  # (n_frames, 34)
    calcparams_array = np.array(all_calcparams_params)  # (n_frames, 34)

    # Compute variance across frames for each parameter
    csv_var_per_param = np.var(csv_params_array, axis=0)  # (34,)
    calcparams_var_per_param = np.var(calcparams_array, axis=0)  # (34,)

    print(f"\nVariance across frames (per parameter):")
    print(f"  CSV mean variance: {csv_var_per_param.mean():.6f}")
    print(f"  CalcParams mean variance: {calcparams_var_per_param.mean():.6f}")
    print(f"  Variance ratio (CalcParams/CSV): {calcparams_var_per_param.mean() / csv_var_per_param.mean():.4f}")

    print(f"\nPer-parameter variance comparison:")
    for i in range(34):
        ratio = calcparams_var_per_param[i] / (csv_var_per_param[i] + 1e-10)
        print(f"  p_{i:2d}: CSV_var={csv_var_per_param[i]:8.4f}, CalcParams_var={calcparams_var_per_param[i]:8.4f}, ratio={ratio:6.3f}")

    # Check if CalcParams params are all very similar
    print(f"\nFrame-to-frame similarity:")
    for i in range(len(all_calcparams_params) - 1):
        diff = all_calcparams_params[i+1] - all_calcparams_params[i]
        print(f"  Frame {i+1} → {i+2}: mean_diff={np.mean(np.abs(diff)):.6f}, max_diff={np.max(np.abs(diff)):.6f}")

print("\n" + "=" * 80)
