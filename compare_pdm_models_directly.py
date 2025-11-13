#!/usr/bin/env python3
"""
Compare PDM models and initialization directly.

This extracts the actual PDM models from both PyCLNF and OpenFace,
then compares how they initialize from the same bounding box.
"""

import cv2
import numpy as np
import sys

sys.path.insert(0, 'pyclnf')
from pyclnf.core.pdm import PDM

# Test configuration
FACE_BBOX = (241, 555, 532, 532)  # Known bbox for frame 50


def analyze_pyclnf_pdm():
    """Analyze PyCLNF's PDM model and initialization."""
    print("=" * 80)
    print("ANALYZING PYCLNF PDM MODEL")
    print("=" * 80)
    print()

    pdm = PDM('pyclnf/models/exported_pdm')

    print(f"PDM parameters:")
    print(f"  n_params: {pdm.n_params}")
    print(f"  n_points: {pdm.n_points}")
    print(f"  mean_shape: {pdm.mean_shape.shape}")
    print(f"  principal_components: {pdm.principal_components.shape}")
    print(f"  eigenvalues: {pdm.eigenvalues.shape}")
    print()

    # Initialize from bbox
    params = pdm.init_params(FACE_BBOX)

    print(f"Initialization from bbox {FACE_BBOX}:")
    print(f"  Scale (param[0]): {params[0]:.6f}")
    print(f"  Rotation (param[1:4]): [{params[1]:.6f}, {params[2]:.6f}, {params[3]:.6f}]")
    print(f"  Translation (param[4:6]): [{params[4]:.2f}, {params[5]:.2f}]")
    print(f"  Shape params (param[6:]): all zeros? {np.allclose(params[6:], 0)}")
    print()

    # Convert to landmarks
    landmarks = pdm.params_to_landmarks_2d(params)

    print(f"Initial landmarks (2D):")
    print(f"  Shape: {landmarks.shape}")
    print(f"  First 5 landmarks:")
    for i in range(5):
        print(f"    {i}: ({landmarks[i, 0]:.2f}, {landmarks[i, 1]:.2f})")
    print()

    # Get bounding box of mean shape
    mean_shape_3d = pdm.mean_shape.reshape(-1, 3).T
    R = cv2.Rodrigues(np.array([0.0, 0.0, 0.0]))[0]
    rotated = R @ mean_shape_3d

    min_x, max_x = rotated[0, :].min(), rotated[0, :].max()
    min_y, max_y = rotated[1, :].min(), rotated[1, :].max()

    model_width = abs(max_x - min_x)
    model_height = abs(max_y - min_y)

    print(f"Mean shape bounding box:")
    print(f"  Width: {model_width:.2f}")
    print(f"  Height: {model_height:.2f}")
    print(f"  Aspect ratio: {model_width/model_height:.4f}")
    print()

    # Verify initialization formula
    x, y, w, h = FACE_BBOX
    expected_scale = ((w / model_width) + (h / model_height)) / 2.0
    expected_tx = x + w/2.0 - expected_scale * (min_x + max_x) / 2.0
    expected_ty = y + h/2.0 - expected_scale * (min_y + max_y) / 2.0

    print(f"Verification of initialization formula:")
    print(f"  Expected scale: {expected_scale:.6f}")
    print(f"  Actual scale:   {params[0]:.6f}")
    print(f"  Match: {np.isclose(params[0], expected_scale)}")
    print()
    print(f"  Expected tx: {expected_tx:.2f}")
    print(f"  Actual tx:   {params[4]:.2f}")
    print(f"  Match: {np.isclose(params[4], expected_tx)}")
    print()
    print(f"  Expected ty: {expected_ty:.2f}")
    print(f"  Actual ty:   {params[5]:.2f}")
    print(f"  Match: {np.isclose(params[5], expected_ty)}")
    print()

    return {
        'pdm': pdm,
        'params': params,
        'landmarks': landmarks,
        'mean_shape_bbox': (min_x, max_x, min_y, max_y),
        'model_width': model_width,
        'model_height': model_height,
    }


def analyze_openface_pdm():
    """Analyze OpenFace's PDM model directly from the model files."""
    print("=" * 80)
    print("ANALYZING OPENFACE PDM MODEL")
    print("=" * 80)
    print()

    # OpenFace PDM is in:
    # ~/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model/pdms/
    import os

    openface_model_dir = os.path.expanduser(
        "~/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model/pdms/"
    )

    # The main model is usually "In-the-wild_aligned_PDM_68.txt"
    model_file = os.path.join(openface_model_dir, "In-the-wild_aligned_PDM_68.txt")

    if not os.path.exists(model_file):
        print(f"ERROR: OpenFace PDM model not found at {model_file}")
        print(f"Available files in {openface_model_dir}:")
        if os.path.exists(openface_model_dir):
            for f in os.listdir(openface_model_dir):
                print(f"  {f}")
        return None

    print(f"Loading OpenFace PDM from: {model_file}")
    print()

    # Parse the OpenFace PDM file format
    # Format is:
    # n_points n_modes
    # mean_x mean_y mean_z (for each point)
    # eigenvalue_1
    # pc_1_for_all_points
    # eigenvalue_2
    # pc_2_for_all_points
    # ...

    with open(model_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # First line: n_points n_modes
    n_points, n_modes = map(int, lines[0].split())
    print(f"OpenFace PDM parameters:")
    print(f"  n_points: {n_points}")
    print(f"  n_modes: {n_modes}")
    print()

    # Next n_points lines: mean shape
    mean_shape = []
    for i in range(1, n_points + 1):
        parts = list(map(float, lines[i].split()))
        mean_shape.extend(parts)
    mean_shape = np.array(mean_shape, dtype=np.float32)

    print(f"  mean_shape: {mean_shape.shape}")

    # Reshape to (n_points, 3)
    mean_shape_3d = mean_shape.reshape(n_points, 3)

    # Get bounding box
    R = cv2.Rodrigues(np.array([0.0, 0.0, 0.0]))[0]
    rotated = (R @ mean_shape_3d.T)

    min_x, max_x = rotated[0, :].min(), rotated[0, :].max()
    min_y, max_y = rotated[1, :].min(), rotated[1, :].max()

    model_width = abs(max_x - min_x)
    model_height = abs(max_y - min_y)

    print(f"  model_width: {model_width:.2f}")
    print(f"  model_height: {model_height:.2f}")
    print(f"  aspect_ratio: {model_width/model_height:.4f}")
    print()

    # Compute initialization from bbox using OpenFace formula
    x, y, w, h = FACE_BBOX

    scaling = ((w / model_width) + (h / model_height)) / 2.0
    tx = x + w/2.0 - scaling * (min_x + max_x) / 2.0
    ty = y + h/2.0 - scaling * (min_y + max_y) / 2.0

    print(f"Initialization from bbox {FACE_BBOX}:")
    print(f"  Scale: {scaling:.6f}")
    print(f"  Translation: [{tx:.2f}, {ty:.2f}]")
    print(f"  Rotation: [0.0, 0.0, 0.0] (assumed)")
    print()

    # Compute initial landmarks (apply similarity transform)
    R = np.eye(3, dtype=np.float32)  # No rotation
    transformed_3d = scaling * (R @ mean_shape_3d.T)
    transformed_3d[0, :] += tx
    transformed_3d[1, :] += ty

    landmarks_2d = transformed_3d[:2, :].T

    print(f"Initial landmarks (2D):")
    print(f"  Shape: {landmarks_2d.shape}")
    print(f"  First 5 landmarks:")
    for i in range(5):
        print(f"    {i}: ({landmarks_2d[i, 0]:.2f}, {landmarks_2d[i, 1]:.2f})")
    print()

    return {
        'n_points': n_points,
        'n_modes': n_modes,
        'mean_shape': mean_shape,
        'landmarks': landmarks_2d,
        'mean_shape_bbox': (min_x, max_x, min_y, max_y),
        'model_width': model_width,
        'model_height': model_height,
        'scale': scaling,
        'translation': (tx, ty),
    }


def main():
    print()
    print("=" * 80)
    print("PDM MODEL COMPARISON")
    print("=" * 80)
    print()

    # Analyze both PDMs
    pyclnf_info = analyze_pyclnf_pdm()
    openface_info = analyze_openface_pdm()

    if openface_info is None:
        print("ERROR: Could not load OpenFace PDM")
        return

    # Compare
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()

    print("Model dimensions:")
    print(f"  PyCLNF  width: {pyclnf_info['model_width']:.2f}")
    print(f"  OpenFace width: {openface_info['model_width']:.2f}")
    print(f"  Difference: {abs(pyclnf_info['model_width'] - openface_info['model_width']):.4f}")
    print()

    print(f"  PyCLNF  height: {pyclnf_info['model_height']:.2f}")
    print(f"  OpenFace height: {openface_info['model_height']:.2f}")
    print(f"  Difference: {abs(pyclnf_info['model_height'] - openface_info['model_height']):.4f}")
    print()

    print("Initialization parameters:")
    print(f"  PyCLNF  scale: {pyclnf_info['params'][0]:.6f}")
    print(f"  OpenFace scale: {openface_info['scale']:.6f}")
    print(f"  Difference: {abs(pyclnf_info['params'][0] - openface_info['scale']):.6f}")
    print()

    print(f"  PyCLNF  tx: {pyclnf_info['params'][4]:.2f}")
    print(f"  OpenFace tx: {openface_info['translation'][0]:.2f}")
    print(f"  Difference: {abs(pyclnf_info['params'][4] - openface_info['translation'][0]):.2f} px")
    print()

    print(f"  PyCLNF  ty: {pyclnf_info['params'][5]:.2f}")
    print(f"  OpenFace ty: {openface_info['translation'][1]:.2f}")
    print(f"  Difference: {abs(pyclnf_info['params'][5] - openface_info['translation'][1]):.2f} px")
    print()

    # Compare initial landmarks
    py_lm = pyclnf_info['landmarks']
    of_lm = openface_info['landmarks']

    diff = py_lm - of_lm
    diff_mag = np.linalg.norm(diff, axis=1)

    print("Initial landmark differences:")
    print(f"  Mean error: {diff_mag.mean():.2f} px")
    print(f"  Median error: {np.median(diff_mag):.2f} px")
    print(f"  Max error: {diff_mag.max():.2f} px")
    print(f"  Min error: {diff_mag.min():.2f} px")
    print()

    if diff_mag.max() > 1.0:
        print("WARNING: Initialization differs by >1px!")
        print()
        worst_idx = np.argsort(-diff_mag)[:5]
        print("Worst 5 landmarks:")
        for rank, idx in enumerate(worst_idx, 1):
            print(f"  {rank}. Landmark {idx}: {diff_mag[idx]:.2f}px")
            print(f"     PyCLNF:  ({py_lm[idx, 0]:.2f}, {py_lm[idx, 1]:.2f})")
            print(f"     OpenFace: ({of_lm[idx, 0]:.2f}, {of_lm[idx, 1]:.2f})")
    else:
        print("✓ Initializations match within 1px!")

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()

    if diff_mag.mean() < 1.0:
        print("✓ PDM models are IDENTICAL")
        print("✓ Initialization formulas are IDENTICAL")
        print()
        print("The 224px error is EXPECTED - it's the distance from initial to converged.")
        print("The problem must be in the OPTIMIZATION, not initialization.")
    else:
        print("✗ PDM models or initialization formulas DIFFER!")
        print(f"  Mean initialization error: {diff_mag.mean():.2f}px")
        print()
        print("This explains the convergence issues.")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
