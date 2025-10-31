#!/usr/bin/env python3
"""
Test CalcParams implementation

Validates that our Python CalcParams implementation:
1. Converges correctly on validation frames
2. Produces similar parameters to C++ baseline
3. Improves alignment quality
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from calc_params import CalcParams
from pdm_parser import PDMParser

print("=" * 80)
print("CalcParams Implementation Test")
print("=" * 80)

# Configuration
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
PDM_FILE = "In-the-wild_aligned_PDM_68.txt"

# Load PDM
print("\n1. Loading PDM...")
pdm = PDMParser(PDM_FILE)
print(f"✓ Loaded PDM: {pdm.mean_shape.shape[0]//3} landmarks, {pdm.princ_comp.shape[1]} PCA components")

# Create CalcParams instance
calc_params = CalcParams(pdm)
print("✓ CalcParams instance created")

# Load test data
print("\n2. Loading test data...")
df = pd.read_csv(CSV_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
print(f"  CSV: {len(df)} rows")

# Test on a few representative frames
test_frames = [1, 100, 493, 617, 863, 1000]  # Mix of easy and challenging frames

print(f"\n3. Testing CalcParams on {len(test_frames)} frames...")
print("-" * 80)

results = []

for frame_num in test_frames:
    row = df[df['frame'] == frame_num]
    if len(row) == 0:
        continue
    row = row.iloc[0]

    # Get 2D landmarks from CSV
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_2d = np.stack([x, y], axis=1)  # (68, 2)

    # Get C++ baseline parameters for comparison
    cpp_params_global = np.array([
        row['p_scale'],
        row['p_rx'],
        row['p_ry'],
        row['p_rz'],
        row['p_tx'],
        row['p_ty']
    ], dtype=np.float32)

    cpp_params_local_cols = [f'p_{i}' for i in range(34)]
    cpp_params_local = row[cpp_params_local_cols].values.astype(np.float32)

    # Run CalcParams
    try:
        params_global, params_local = calc_params.calc_params(landmarks_2d)

        if params_global is not None and params_local is not None:
            # Compute parameter differences
            global_diff = params_global - cpp_params_global
            local_diff = params_local - cpp_params_local

            global_rmse = np.sqrt(np.mean(global_diff ** 2))
            local_rmse = np.sqrt(np.mean(local_diff ** 2))

            # Compute 2D reprojection error
            # Reconstruct 3D shape
            shape_3d = pdm.mean_shape + pdm.princ_comp @ params_local.reshape(-1, 1)
            shape_3d = shape_3d.reshape(68, 3)  # (68, 3)

            # Apply transformation
            scale, rx, ry, rz, tx, ty = params_global
            R = calc_params.euler_to_rotation_matrix(np.array([rx, ry, rz]))

            # Project to 2D
            projected_2d = np.zeros((68, 2), dtype=np.float32)
            for i in range(68):
                pt_3d = shape_3d[i]
                pt_rot = R @ pt_3d
                projected_2d[i, 0] = scale * pt_rot[0] + tx
                projected_2d[i, 1] = scale * pt_rot[1] + ty

            # Compute reprojection error
            reproj_error = np.sqrt(np.mean((projected_2d - landmarks_2d) ** 2))

            results.append({
                'frame': frame_num,
                'success': True,
                'global_rmse': global_rmse,
                'local_rmse': local_rmse,
                'reproj_error': reproj_error,
                'params_global': params_global,
                'cpp_params_global': cpp_params_global
            })

            print(f"\nFrame {frame_num}:")
            print(f"  ✓ Converged successfully")
            print(f"  Global params RMSE: {global_rmse:.6f}")
            print(f"    scale: Python={params_global[0]:.4f}, C++={cpp_params_global[0]:.4f}, diff={global_diff[0]:.4f}")
            print(f"    rx:    Python={params_global[1]:.4f}, C++={cpp_params_global[1]:.4f}, diff={global_diff[1]:.4f}")
            print(f"    ry:    Python={params_global[2]:.4f}, C++={cpp_params_global[2]:.4f}, diff={global_diff[2]:.4f}")
            print(f"    rz:    Python={params_global[3]:.4f}, C++={cpp_params_global[3]:.4f}, diff={global_diff[3]:.4f}")
            print(f"    tx:    Python={params_global[4]:.4f}, C++={cpp_params_global[4]:.4f}, diff={global_diff[4]:.4f}")
            print(f"    ty:    Python={params_global[5]:.4f}, C++={cpp_params_global[5]:.4f}, diff={global_diff[5]:.4f}")
            print(f"  Local params RMSE: {local_rmse:.6f}")
            print(f"  2D reprojection error: {reproj_error:.4f} pixels")

        else:
            print(f"\nFrame {frame_num}:")
            print(f"  ✗ Failed to converge")
            results.append({
                'frame': frame_num,
                'success': False
            })

    except Exception as e:
        print(f"\nFrame {frame_num}:")
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            'frame': frame_num,
            'success': False,
            'error': str(e)
        })

cap.release()

# Summary
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

successful = [r for r in results if r['success']]
failed = [r for r in results if not r['success']]

print(f"\nSuccessful: {len(successful)}/{len(results)}")
print(f"Failed: {len(failed)}/{len(results)}")

if len(successful) > 0:
    avg_global_rmse = np.mean([r['global_rmse'] for r in successful])
    avg_local_rmse = np.mean([r['local_rmse'] for r in successful])
    avg_reproj_error = np.mean([r['reproj_error'] for r in successful])

    print(f"\nAverage global params RMSE: {avg_global_rmse:.6f}")
    print(f"Average local params RMSE: {avg_local_rmse:.6f}")
    print(f"Average reprojection error: {avg_reproj_error:.4f} pixels")

    print("\nInterpretation:")
    if avg_global_rmse < 0.01 and avg_local_rmse < 0.1:
        print("  ✓ EXCELLENT - CalcParams matches C++ nearly exactly!")
        print("  → Ready for integration into AU pipeline")
    elif avg_global_rmse < 0.1 and avg_local_rmse < 0.5:
        print("  ✓ GOOD - CalcParams is close to C++")
        print("  → Should test if this improves AU predictions")
    elif avg_reproj_error < 3.0:
        print("  ~ ACCEPTABLE - CalcParams converges with low reprojection error")
        print("  → Different from C++ but may still be valid")
    else:
        print("  ✗ POOR - CalcParams may have issues")
        print("  → Need to debug implementation")

print("\n" + "=" * 80)
