#!/usr/bin/env python3
"""
Test fit_to_landmarks with REAL landmarks from Python CLNF.

Uses actual landmarks from:
1. C++ OpenFace (ground truth)
2. Python CLNF (before eye refinement)
3. Python CLNF after eye refinement

Compares how well fit_to_landmarks recovers the eye refinement changes.
"""

import numpy as np
from pathlib import Path
import sys
import subprocess
import tempfile
import cv2

sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))

from pyclnf.core.pdm import PDM
from pyclnf import CLNF
from pymtcnn import MTCNN


def get_cpp_landmarks(image_path: str) -> np.ndarray:
    """Get ground truth landmarks from C++ OpenFace."""
    openface_bin = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run([
            openface_bin,
            "-f", image_path,
            "-out_dir", tmpdir,
            "-2Dfp"
        ], capture_output=True, text=True)

        csv_file = Path(tmpdir) / (Path(image_path).stem + ".csv")
        if csv_file.exists():
            import csv
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                row = next(reader)
                landmarks = np.zeros((68, 2), dtype=np.float32)
                for i in range(68):
                    landmarks[i, 0] = float(row[f"x_{i}"])
                    landmarks[i, 1] = float(row[f"y_{i}"])
                return landmarks
    return None


def test_with_real_landmarks():
    """Test fit_to_landmarks with real CLNF output."""

    print("="*70)
    print("FIT_TO_LANDMARKS TEST WITH REAL LANDMARKS")
    print("="*70)

    # Use first frame from video
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read video frame")
        return

    print(f"Frame size: {frame.shape}")

    # Save frame for OpenFace
    frame_path = "/tmp/test_frame.png"
    cv2.imwrite(frame_path, frame)

    # Get C++ ground truth
    print("\n1. Getting C++ OpenFace landmarks...")
    cpp_landmarks = get_cpp_landmarks(frame_path)
    if cpp_landmarks is None:
        print("ERROR: Could not get C++ landmarks")
        return
    print(f"   C++ LM36: ({cpp_landmarks[36, 0]:.4f}, {cpp_landmarks[36, 1]:.4f})")

    # Initialize Python CLNF
    print("\n2. Running Python CLNF...")
    pdm = PDM("pyclnf/models/exported_pdm")

    clnf = CLNF(
        model_dir="pyclnf/models",
        max_iterations=40,
        use_eye_refinement=True,
        debug_mode=False
    )

    detector = MTCNN()
    faces, _ = detector.detect(frame)

    if len(faces) == 0:
        print("ERROR: No face detected")
        return

    bbox_np = faces[0]  # faces is array of [x1, y1, x2, y2]

    # Get Python landmarks (with eye refinement)
    py_landmarks, info = clnf.fit(frame, bbox_np, return_params=True)
    params = info.get('params')

    print(f"   Python LM36: ({py_landmarks[36, 0]:.4f}, {py_landmarks[36, 1]:.4f})")
    print(f"   Error to C++: {np.linalg.norm(py_landmarks[36] - cpp_landmarks[36]):.4f} px")

    # Get Python landmarks WITHOUT eye refinement for comparison
    clnf_no_eye = CLNF(
        model_dir="pyclnf/models",
        max_iterations=40,
        use_eye_refinement=False,
        debug_mode=False
    )

    py_no_eye_landmarks, info_no_eye = clnf_no_eye.fit(frame, bbox_np, return_params=True)
    params_no_eye = info_no_eye.get('params')
    print(f"   Python LM36 (no eye): ({py_no_eye_landmarks[36, 0]:.4f}, {py_no_eye_landmarks[36, 1]:.4f})")

    # Now test fit_to_landmarks
    print("\n3. Testing fit_to_landmarks...")
    print("   Scenario: eye refinement moves LM36, fit_to_landmarks converts back to params")

    # Get main model params (before eye refinement would modify)
    main_params = params_no_eye.copy()
    main_landmarks = pdm.params_to_landmarks_2d(main_params)

    print(f"   Main model LM36: ({main_landmarks[36, 0]:.4f}, {main_landmarks[36, 1]:.4f})")

    # Simulate eye refinement moving landmarks
    # Use the actual movement from py_landmarks vs py_no_eye_landmarks for eyes
    eye_refined_landmarks = main_landmarks.copy()

    # Copy all landmarks, but apply eye refinement deltas to eye region
    # Left eye: 36-41
    for i in range(36, 42):
        eye_refined_landmarks[i] = py_landmarks[i]

    print(f"\n   Eye refinement delta for LM36:")
    print(f"     X: {eye_refined_landmarks[36, 0] - main_landmarks[36, 0]:+.4f}")
    print(f"     Y: {eye_refined_landmarks[36, 1] - main_landmarks[36, 1]:+.4f}")

    # Now run fit_to_landmarks
    fitted_params = fit_with_logging(pdm, eye_refined_landmarks)

    fitted_landmarks = pdm.params_to_landmarks_2d(fitted_params)

    print(f"\n4. Results:")
    print(f"   Main model LM36: ({main_landmarks[36, 0]:.4f}, {main_landmarks[36, 1]:.4f})")
    print(f"   Eye-refined LM36: ({eye_refined_landmarks[36, 0]:.4f}, {eye_refined_landmarks[36, 1]:.4f})")
    print(f"   Fitted LM36:     ({fitted_landmarks[36, 0]:.4f}, {fitted_landmarks[36, 1]:.4f})")
    print(f"   C++ LM36:        ({cpp_landmarks[36, 0]:.4f}, {cpp_landmarks[36, 1]:.4f})")

    # Analyze the movement
    eye_delta = eye_refined_landmarks[36] - main_landmarks[36]
    fit_delta = fitted_landmarks[36] - main_landmarks[36]

    print(f"\n   Movement analysis:")
    print(f"     Eye requested:    ({eye_delta[0]:+.4f}, {eye_delta[1]:+.4f})")
    print(f"     fit_to_landmarks: ({fit_delta[0]:+.4f}, {fit_delta[1]:+.4f})")

    if eye_delta[0] != 0:
        print(f"     X recovery: {fit_delta[0]/eye_delta[0]*100:.1f}%")
    if eye_delta[1] != 0:
        print(f"     Y recovery: {fit_delta[1]/eye_delta[1]*100:.1f}%")

    # Check direction
    if eye_delta[0] * fit_delta[0] < 0:
        print(f"     ⚠️  X DIRECTION INVERTED!")
    if eye_delta[1] * fit_delta[1] < 0:
        print(f"     ⚠️  Y DIRECTION INVERTED!")


def fit_with_logging(pdm, landmarks: np.ndarray):
    """Fit PDM parameters with brief logging."""

    n = pdm.n_points
    m = pdm.n_modes

    landmarks = landmarks.reshape(-1, 2)

    # All visible
    visi_count = n

    # Landmark locations in blocked format
    landmark_locs_vis = np.zeros((visi_count * 2, 1), dtype=np.float32)
    for i in range(n):
        landmark_locs_vis[i] = landmarks[i, 0]
        landmark_locs_vis[i + visi_count] = landmarks[i, 1]

    # Initial params from bbox
    min_x, max_x = landmarks[:, 0].min(), landmarks[:, 0].max()
    min_y, max_y = landmarks[:, 1].min(), landmarks[:, 1].max()

    width = abs(max_x - min_x)
    height = abs(max_y - min_y)

    # Model bbox
    neutral_params = np.zeros(pdm.n_params, dtype=np.float32)
    neutral_params[0] = 1.0
    neutral_lm = pdm.params_to_landmarks_2d(neutral_params)
    model_width = neutral_lm[:, 0].max() - neutral_lm[:, 0].min()
    model_height = neutral_lm[:, 1].max() - neutral_lm[:, 1].min()

    scaling = ((width / model_width) + (height / model_height)) / 2.0
    translation = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0], dtype=np.float32)

    # Initialize
    loc_params = np.zeros(m, dtype=np.float32)
    glob_params = np.array([scaling, 0.0, 0.0, 0.0, translation[0], translation[1]], dtype=np.float32)

    # Regularization
    reg_factor = 1.0
    regularisations = np.zeros(6 + m, dtype=np.float32)
    regularisations[6:] = reg_factor / pdm.eigen_values.flatten()
    reg_diag = np.diag(regularisations)

    W = np.eye(visi_count * 2, dtype=np.float32)
    M = pdm.mean_shape.flatten().reshape(-1, 1)
    V = pdm.princ_comp

    prev_error = float('inf')
    not_improved_in = 0

    for iteration in range(1000):
        # Current shape
        shape_3d = M + V @ loc_params.reshape(-1, 1)
        shape_3d = shape_3d.reshape(3, n).T

        rotation_init = glob_params[1:4]
        R = pdm._euler_to_rotation_matrix(rotation_init)
        R_2d = R[:2, :]

        curr_shape_2d = glob_params[0] * (shape_3d @ R_2d.T)
        curr_shape_2d[:, 0] += glob_params[4]
        curr_shape_2d[:, 1] += glob_params[5]

        # Flatten
        curr_shape = np.zeros((n * 2, 1), dtype=np.float32)
        curr_shape[:n, 0] = curr_shape_2d[:, 0]
        curr_shape[n:, 0] = curr_shape_2d[:, 1]

        error_resid = landmark_locs_vis - curr_shape
        error = np.linalg.norm(error_resid)

        # Check convergence - use RELATIVE like C++ (0.1% improvement)
        if prev_error * 0.999 < error:
            not_improved_in += 1
            if not_improved_in >= 3:
                break
        else:
            not_improved_in = 0

        prev_error = error

        # Jacobian
        J = pdm._compute_jacobian_subsampled(loc_params, glob_params, M, V, n)

        J_w_t = J.T @ W
        J_w_t_m = J_w_t @ error_resid
        J_w_t_m[6:] = J_w_t_m[6:] - reg_diag[6:, 6:] @ loc_params.reshape(-1, 1)

        Hessian = J_w_t @ J + reg_diag

        try:
            param_update = np.linalg.solve(Hessian, J_w_t_m).flatten()
        except:
            break

        # Damping
        param_update *= 0.75

        # Update
        full_params = np.concatenate([glob_params, loc_params])
        full_params = pdm.update_params(full_params, param_update)

        glob_params = full_params[:6]
        loc_params = full_params[6:]

    print(f"   fit_to_landmarks: {iteration} iterations, final error={error:.4f}")

    return np.concatenate([glob_params, loc_params])


if __name__ == "__main__":
    test_with_real_landmarks()
