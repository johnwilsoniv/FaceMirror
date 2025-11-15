"""
Debug coordinate system, transposition, and row/column ordering issues in CLNF.

This script tests different variations to find coordinate system bugs:
1. Row-major vs column-major reshaping
2. Response map transposition
3. Sigma component transposition
4. X/Y coordinate swapping
"""

import numpy as np
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "pyclnf"))

from core.pdm import PDM
from core.patch_expert import CCNFModel
import cv2
import json

PROJECT_ROOT = Path(__file__).parent
PDM_DIR = PROJECT_ROOT / "pyclnf" / "models" / "exported_pdm"
MODELS_DIR = PROJECT_ROOT / "pyclnf" / "models"
TEST_IMAGE = PROJECT_ROOT / "calibration_frames" / "patient1_frame1.jpg"
PYTHON_RESULT = PROJECT_ROOT / "validation_output" / "python_baseline" / "patient1_frame1_result.json"


def test_response_map_reshape_order():
    """Test if reshape order (row-major vs column-major) affects Sigma transformation."""

    print("="*80)
    print("TEST 1: Response Map Reshape Order")
    print("="*80)

    # Load test data
    img = cv2.imread(str(TEST_IMAGE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    with open(PYTHON_RESULT, 'r') as f:
        data = json.load(f)
    bbox_xyxy = data['debug_info']['face_detection']['bbox']
    bbox = (bbox_xyxy[0], bbox_xyxy[1],
            bbox_xyxy[2] - bbox_xyxy[0],
            bbox_xyxy[3] - bbox_xyxy[1])

    # Initialize
    pdm = PDM(str(PDM_DIR))
    params_init = pdm.init_params(bbox)
    landmarks_init = pdm.params_to_landmarks_2d(params_init)

    # Load CCNF model with sigma components
    ccnf = CCNFModel(str(MODELS_DIR), scales=[0.25])
    view_idx = 0
    landmark_idx = 36  # Left eye corner

    scale_model = ccnf.scale_models.get(0.25)
    view_data = scale_model['views'].get(view_idx)
    patch_expert = view_data['patches'][landmark_idx]

    # Get sigma components
    sigma_comps_11 = ccnf.sigma_components[11]

    # Compute response map
    lm_x, lm_y = landmarks_init[landmark_idx]
    window_size = 11
    half_window = window_size // 2

    response_map = np.zeros((window_size, window_size))
    start_x = int(lm_x) - half_window
    start_y = int(lm_y) - half_window

    for i in range(window_size):
        for j in range(window_size):
            patch_x = start_x + j
            patch_y = start_y + i

            # Extract patch
            half_w = patch_expert.width // 2
            half_h = patch_expert.height // 2
            x1 = patch_x - half_w
            y1 = patch_y - half_h
            x2 = x1 + patch_expert.width
            y2 = y1 + patch_expert.height

            if 0 <= x1 and 0 <= y1 and x2 < gray.shape[1] and y2 < gray.shape[0]:
                patch = gray[y1:y2, x1:x2]
                response_map[i, j] = patch_expert.compute_response(patch)
            else:
                response_map[i, j] = -1e10

    # Compute Sigma matrix
    Sigma = patch_expert.compute_sigma(sigma_comps_11, window_size=window_size)

    print(f"\nResponse map shape: {response_map.shape}")
    print(f"Sigma matrix shape: {Sigma.shape}")

    # Original peak
    peak_orig = np.unravel_index(response_map.argmax(), response_map.shape)
    center = window_size // 2
    offset_orig = (peak_orig[1] - center, peak_orig[0] - center)
    dist_orig = np.sqrt(offset_orig[0]**2 + offset_orig[1]**2)

    print(f"\n--- ORIGINAL ---")
    print(f"Peak at: {peak_orig} (row={peak_orig[0]}, col={peak_orig[1]})")
    print(f"Offset: ({offset_orig[0]:+d}, {offset_orig[1]:+d}) = {dist_orig:.1f}px")
    print(f"Peak value: {response_map[peak_orig]:.3f}")

    # Test 1: Row-major reshape (default, current implementation)
    response_vec_rowmajor = response_map.reshape(-1, 1, order='C')
    response_transformed_rowmajor = Sigma @ response_vec_rowmajor
    response_map_rowmajor = response_transformed_rowmajor.reshape(window_size, window_size, order='C')

    peak_rowmajor = np.unravel_index(response_map_rowmajor.argmax(), response_map_rowmajor.shape)
    offset_rowmajor = (peak_rowmajor[1] - center, peak_rowmajor[0] - center)
    dist_rowmajor = np.sqrt(offset_rowmajor[0]**2 + offset_rowmajor[1]**2)

    print(f"\n--- ROW-MAJOR RESHAPE (Current) ---")
    print(f"Peak at: {peak_rowmajor}")
    print(f"Offset: ({offset_rowmajor[0]:+d}, {offset_rowmajor[1]:+d}) = {dist_rowmajor:.1f}px")
    print(f"Peak value: {response_map_rowmajor[peak_rowmajor]:.3f}")
    print(f"Offset changed: {offset_rowmajor != offset_orig}")

    # Test 2: Column-major reshape (Fortran order)
    response_vec_colmajor = response_map.reshape(-1, 1, order='F')
    response_transformed_colmajor = Sigma @ response_vec_colmajor
    response_map_colmajor = response_transformed_colmajor.reshape(window_size, window_size, order='F')

    peak_colmajor = np.unravel_index(response_map_colmajor.argmax(), response_map_colmajor.shape)
    offset_colmajor = (peak_colmajor[1] - center, peak_colmajor[0] - center)
    dist_colmajor = np.sqrt(offset_colmajor[0]**2 + offset_colmajor[1]**2)

    print(f"\n--- COLUMN-MAJOR RESHAPE (Fortran) ---")
    print(f"Peak at: {peak_colmajor}")
    print(f"Offset: ({offset_colmajor[0]:+d}, {offset_colmajor[1]:+d}) = {dist_colmajor:.1f}px")
    print(f"Peak value: {response_map_colmajor[peak_colmajor]:.3f}")
    print(f"Offset changed: {offset_colmajor != offset_orig}")

    print(f"\n--- COMPARISON ---")
    print(f"Row-major offset distance: {dist_rowmajor:.1f}px")
    print(f"Column-major offset distance: {dist_colmajor:.1f}px")
    print(f"Original offset distance: {dist_orig:.1f}px")

    if dist_colmajor < dist_rowmajor and dist_colmajor < dist_orig:
        print(f"\n✓ COLUMN-MAJOR RESHAPE IMPROVES PEAK CENTERING!")
        return "column-major"
    elif dist_rowmajor < dist_orig:
        print(f"\n→ Row-major is current behavior (no improvement)")
        return "row-major"
    else:
        print(f"\n✗ Neither reshape order improves peak centering")
        return None


def test_response_map_transpose():
    """Test if transposing the response map before Sigma transformation helps."""

    print("\n" + "="*80)
    print("TEST 2: Response Map Transposition")
    print("="*80)

    # Load test data
    img = cv2.imread(str(TEST_IMAGE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    with open(PYTHON_RESULT, 'r') as f:
        data = json.load(f)
    bbox_xyxy = data['debug_info']['face_detection']['bbox']
    bbox = (bbox_xyxy[0], bbox_xyxy[1],
            bbox_xyxy[2] - bbox_xyxy[0],
            bbox_xyxy[3] - bbox_xyxy[1])

    # Initialize
    pdm = PDM(str(PDM_DIR))
    params_init = pdm.init_params(bbox)
    landmarks_init = pdm.params_to_landmarks_2d(params_init)

    # Load CCNF model
    ccnf = CCNFModel(str(MODELS_DIR), scales=[0.25])
    view_idx = 0
    landmark_idx = 36

    scale_model = ccnf.scale_models.get(0.25)
    view_data = scale_model['views'].get(view_idx)
    patch_expert = view_data['patches'][landmark_idx]
    sigma_comps_11 = ccnf.sigma_components[11]

    # Compute response map
    lm_x, lm_y = landmarks_init[landmark_idx]
    window_size = 11
    half_window = window_size // 2

    response_map = np.zeros((window_size, window_size))
    start_x = int(lm_x) - half_window
    start_y = int(lm_y) - half_window

    for i in range(window_size):
        for j in range(window_size):
            patch_x = start_x + j
            patch_y = start_y + i

            half_w = patch_expert.width // 2
            half_h = patch_expert.height // 2
            x1 = patch_x - half_w
            y1 = patch_y - half_h
            x2 = x1 + patch_expert.width
            y2 = y1 + patch_expert.height

            if 0 <= x1 and 0 <= y1 and x2 < gray.shape[1] and y2 < gray.shape[0]:
                patch = gray[y1:y2, x1:x2]
                response_map[i, j] = patch_expert.compute_response(patch)
            else:
                response_map[i, j] = -1e10

    Sigma = patch_expert.compute_sigma(sigma_comps_11, window_size=window_size)

    # Original
    peak_orig = np.unravel_index(response_map.argmax(), response_map.shape)
    center = window_size // 2
    offset_orig = (peak_orig[1] - center, peak_orig[0] - center)
    dist_orig = np.sqrt(offset_orig[0]**2 + offset_orig[1]**2)

    print(f"\n--- ORIGINAL ---")
    print(f"Offset: ({offset_orig[0]:+d}, {offset_orig[1]:+d}) = {dist_orig:.1f}px")

    # Test: Transpose response map before and after Sigma
    response_T = response_map.T
    response_vec_T = response_T.reshape(-1, 1)
    response_transformed_T = Sigma @ response_vec_T
    response_map_T = response_transformed_T.reshape(window_size, window_size).T

    peak_T = np.unravel_index(response_map_T.argmax(), response_map_T.shape)
    offset_T = (peak_T[1] - center, peak_T[0] - center)
    dist_T = np.sqrt(offset_T[0]**2 + offset_T[1]**2)

    print(f"\n--- TRANSPOSED RESPONSE ---")
    print(f"Offset: ({offset_T[0]:+d}, {offset_T[1]:+d}) = {dist_T:.1f}px")

    if dist_T < dist_orig:
        print(f"\n✓ TRANSPOSING RESPONSE MAP IMPROVES PEAK CENTERING!")
        return True
    else:
        print(f"\n✗ Transposing response map doesn't help")
        return False


def test_sigma_transpose():
    """Test if Sigma components need to be transposed."""

    print("\n" + "="*80)
    print("TEST 3: Sigma Component Transposition")
    print("="*80)

    # Similar setup...
    img = cv2.imread(str(TEST_IMAGE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    with open(PYTHON_RESULT, 'r') as f:
        data = json.load(f)
    bbox_xyxy = data['debug_info']['face_detection']['bbox']
    bbox = (bbox_xyxy[0], bbox_xyxy[1],
            bbox_xyxy[2] - bbox_xyxy[0],
            bbox_xyxy[3] - bbox_xyxy[1])

    pdm = PDM(str(PDM_DIR))
    params_init = pdm.init_params(bbox)
    landmarks_init = pdm.params_to_landmarks_2d(params_init)

    ccnf = CCNFModel(str(MODELS_DIR), scales=[0.25])
    view_idx = 0
    landmark_idx = 36

    scale_model = ccnf.scale_models.get(0.25)
    view_data = scale_model['views'].get(view_idx)
    patch_expert = view_data['patches'][landmark_idx]

    # Check if sigma components are symmetric
    print("\nChecking Sigma component symmetry...")
    for i, sigma_comp in enumerate(ccnf.sigma_components[11][:3]):
        is_symmetric = np.allclose(sigma_comp, sigma_comp.T)
        print(f"  Sigma component {i}: {'SYMMETRIC' if is_symmetric else 'ASYMMETRIC'}")
        if not is_symmetric:
            max_diff = np.abs(sigma_comp - sigma_comp.T).max()
            print(f"    Max asymmetry: {max_diff:.6f}")


def main():
    print("CLNF Coordinate System Debug")
    print("=" * 80)

    # Run tests
    reshape_result = test_response_map_reshape_order()
    transpose_response_result = test_response_map_transpose()
    test_sigma_transpose()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nBest reshape order: {reshape_result}")
    print(f"Should transpose response map: {transpose_response_result}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if reshape_result == "column-major":
        print("\n✓ Change response_map.reshape(-1, 1) to use order='F' (Fortran/column-major)")
        print("  This matches C++ OpenFace's column-major memory layout")

    if transpose_response_result:
        print("\n✓ Transpose response map before Sigma transformation")
        print("  Suggests X/Y coordinate system mismatch")


if __name__ == "__main__":
    main()
