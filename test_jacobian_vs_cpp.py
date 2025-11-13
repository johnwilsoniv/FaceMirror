#!/usr/bin/env python3
"""
Test Jacobian computation against OpenFace C++ to find sign error.

Strategy:
1. Run OpenFace C++ on a frame to get its landmarks
2. Extract OpenFace's Jacobian (by examining C++ code behavior)
3. Compute PyCLNF Jacobian for the same parameters
4. Compare element-by-element, focusing on landmark 33 which shows sign error
"""

import cv2
import numpy as np
from pyclnf import CLNF
from pyclnf.core.pdm import PDM
import subprocess
import tempfile
from pathlib import Path
import pandas as pd


def extract_openface_landmarks_and_params(image_path: str, output_dir: str):
    """Run OpenFace C++ and extract landmarks."""
    openface_bin = Path("~/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction").expanduser()

    cmd = [
        str(openface_bin),
        "-f", image_path,
        "-out_dir", output_dir,
        "-2Dfp"
    ]

    subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    # Read landmarks from CSV
    csv_file = Path(output_dir) / (Path(image_path).stem + ".csv")
    df = pd.read_csv(csv_file)

    # Extract landmarks
    landmarks = np.zeros((68, 2))
    for i in range(68):
        x_col = f'x_{i}'
        y_col = f'y_{i}'
        if x_col in df.columns:
            landmarks[i, 0] = df[x_col].iloc[0]
            landmarks[i, 1] = df[y_col].iloc[0]

    return landmarks


def compute_numerical_jacobian(pdm: PDM, params: np.ndarray, epsilon: float = 1e-6):
    """
    Compute Jacobian numerically using finite differences.
    This provides ground truth to verify analytical Jacobian.

    J[i, j] = ∂(landmark_coord_i) / ∂(param_j)
    """
    n_params = len(params)
    landmarks_base = pdm.params_to_landmarks_2d(params)
    n_landmarks = len(landmarks_base)

    J = np.zeros((n_landmarks * 2, n_params))

    for j in range(n_params):
        params_plus = params.copy()
        params_plus[j] += epsilon
        landmarks_plus = pdm.params_to_landmarks_2d(params_plus)

        params_minus = params.copy()
        params_minus[j] -= epsilon
        landmarks_minus = pdm.params_to_landmarks_2d(params_minus)

        # Central difference
        diff = (landmarks_plus - landmarks_minus) / (2 * epsilon)

        # Fill in Jacobian column
        J[0::2, j] = diff[:, 0]  # x coordinates
        J[1::2, j] = diff[:, 1]  # y coordinates

    return J


def main():
    print("=" * 80)
    print("JACOBIAN VERIFICATION: PyCLNF vs Numerical (Ground Truth)")
    print("=" * 80)
    print()

    # Load test frame
    video_path = 'Patient Data/Normal Cohort/IMG_0433.MOV'
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    # Extract frame to file for OpenFace C++
    frame_path = "/tmp/test_frame.jpg"
    cv2.imwrite(frame_path, frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_bbox = (241, 555, 532, 532)

    print("Initializing PyCLNF...")
    clnf = CLNF(model_dir='pyclnf/models', max_iterations=1)

    # Get initial parameters
    initial_params = clnf.pdm.init_params(face_bbox)

    print(f"Initial parameters: {len(initial_params)} params")
    print(f"  Scale: {initial_params[0]:.3f}")
    print(f"  Rotation: [{initial_params[1]:.3f}, {initial_params[2]:.3f}, {initial_params[3]:.3f}]")
    print(f"  Translation: [{initial_params[4]:.1f}, {initial_params[5]:.1f}]")
    print(f"  Shape params: {len(initial_params) - 6} values")
    print()

    # Compute analytical Jacobian (PyCLNF implementation)
    print("Computing PyCLNF analytical Jacobian...")
    J_analytical = clnf.pdm.compute_jacobian(initial_params)
    print(f"  Shape: {J_analytical.shape}")

    # Compute numerical Jacobian (ground truth)
    print("Computing numerical Jacobian (ground truth)...")
    J_numerical = compute_numerical_jacobian(clnf.pdm, initial_params, epsilon=1e-5)
    print(f"  Shape: {J_numerical.shape}")
    print()

    # Compare Jacobians
    print("=" * 80)
    print("JACOBIAN COMPARISON")
    print("=" * 80)
    print()

    # Focus on landmark 33 (nose tip) which shows sign error
    landmark_idx = 33
    print(f"Analyzing Jacobian for landmark {landmark_idx} (nose tip - SHOWS SIGN ERROR):")
    print()

    # Get rows for this landmark
    row_x = 2 * landmark_idx
    row_y = 2 * landmark_idx + 1

    print(f"Landmark {landmark_idx} - X coordinate (row {row_x}):")
    print(f"{'Parameter':<20} {'Analytical':>15} {'Numerical':>15} {'Diff':>15} {'Sign Match':>15}")
    print("-" * 80)

    x_errors = []
    for param_idx in range(min(10, J_analytical.shape[1])):  # Check first 10 params
        analytical_val = J_analytical[row_x, param_idx]
        numerical_val = J_numerical[row_x, param_idx]
        diff = analytical_val - numerical_val

        # Check if signs match
        if abs(analytical_val) > 1e-6 and abs(numerical_val) > 1e-6:
            sign_match = np.sign(analytical_val) == np.sign(numerical_val)
            sign_str = "✓" if sign_match else "❌ WRONG"
        else:
            sign_match = True
            sign_str = "~ (small)"

        param_name = ["scale", "rx", "ry", "rz", "tx", "ty"] + [f"q{i}" for i in range(100)]
        print(f"{param_name[param_idx]:<20} {analytical_val:>15.6f} {numerical_val:>15.6f} {diff:>15.6f} {sign_str:>15}")

        if not sign_match:
            x_errors.append((param_idx, param_name[param_idx]))

    print()
    print(f"Landmark {landmark_idx} - Y coordinate (row {row_y}):")
    print(f"{'Parameter':<20} {'Analytical':>15} {'Numerical':>15} {'Diff':>15} {'Sign Match':>15}")
    print("-" * 80)

    y_errors = []
    for param_idx in range(min(10, J_analytical.shape[1])):
        analytical_val = J_analytical[row_y, param_idx]
        numerical_val = J_numerical[row_y, param_idx]
        diff = analytical_val - numerical_val

        if abs(analytical_val) > 1e-6 and abs(numerical_val) > 1e-6:
            sign_match = np.sign(analytical_val) == np.sign(numerical_val)
            sign_str = "✓" if sign_match else "❌ WRONG"
        else:
            sign_match = True
            sign_str = "~ (small)"

        param_name = ["scale", "rx", "ry", "rz", "tx", "ty"] + [f"q{i}" for i in range(100)]
        print(f"{param_name[param_idx]:<20} {analytical_val:>15.6f} {numerical_val:>15.6f} {diff:>15.6f} {sign_str:>15}")

        if not sign_match:
            y_errors.append((param_idx, param_name[param_idx]))

    print()
    print("=" * 80)
    print("OVERALL JACOBIAN COMPARISON")
    print("=" * 80)
    print()

    # Compute overall statistics
    diff_matrix = J_analytical - J_numerical
    relative_error = np.abs(diff_matrix) / (np.abs(J_numerical) + 1e-10)

    # Exclude very small values from statistics
    mask = np.abs(J_numerical) > 1e-6

    print(f"Overall statistics (excluding small values |numerical| < 1e-6):")
    print(f"  Mean absolute error: {np.abs(diff_matrix[mask]).mean():.6f}")
    print(f"  Max absolute error: {np.abs(diff_matrix[mask]).max():.6f}")
    print(f"  Mean relative error: {relative_error[mask].mean():.2%}")
    print(f"  Max relative error: {relative_error[mask].max():.2%}")
    print()

    # Check sign agreement
    sign_analytical = np.sign(J_analytical[mask])
    sign_numerical = np.sign(J_numerical[mask])
    sign_agreement = (sign_analytical == sign_numerical).sum() / len(sign_analytical)

    print(f"Sign agreement: {sign_agreement:.2%}")
    print()

    if sign_agreement < 0.99:
        print(f"⚠️  WARNING: Only {sign_agreement:.2%} of Jacobian elements have correct sign!")
        print()

        # Find elements with wrong signs
        wrong_signs = np.where(sign_analytical != sign_numerical)[0]
        print(f"Found {len(wrong_signs)} elements with wrong signs")

        if len(wrong_signs) > 0:
            print("\nFirst 10 wrong-sign elements:")
            for idx in wrong_signs[:10]:
                # Convert flat index to row, col
                all_indices = np.where(mask)[0]
                flat_idx = all_indices[idx]
                row = flat_idx // J_analytical.shape[1]
                col = flat_idx % J_analytical.shape[1]

                landmark_num = row // 2
                coord = 'x' if row % 2 == 0 else 'y'
                param_name = ["scale", "rx", "ry", "rz", "tx", "ty"] + [f"q{i}" for i in range(100)]

                print(f"  Landmark {landmark_num} {coord} vs {param_name[col]}: "
                      f"analytical={J_analytical[row, col]:+.6f}, "
                      f"numerical={J_numerical[row, col]:+.6f}")

    print()
    print("=" * 80)
    print("CONCLUSION FOR LANDMARK 33 (Sign Error Landmark):")
    print("=" * 80)

    if len(x_errors) > 0 or len(y_errors) > 0:
        print(f"❌ JACOBIAN SIGN ERRORS FOUND for landmark {landmark_idx}!")
        if len(x_errors) > 0:
            print(f"   X-coordinate has wrong signs for parameters: {[name for _, name in x_errors]}")
        if len(y_errors) > 0:
            print(f"   Y-coordinate has wrong signs for parameters: {[name for _, name in y_errors]}")
        print()
        print("   This explains why landmark moves opposite to mean-shift!")
        print("   The bug is in the Jacobian computation in pdm.py")
    else:
        print(f"✓ Jacobian for landmark {landmark_idx} appears correct (signs match numerical)")
        print("  The sign error may be elsewhere in the pipeline")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
