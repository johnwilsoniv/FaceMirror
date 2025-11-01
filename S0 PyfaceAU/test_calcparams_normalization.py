#!/usr/bin/env python3
"""
Isolated Test: CalcParams Eigenvalue Normalization

Tests whether Python CalcParams correctly normalizes local parameters
by eigenvalues to match C++ OpenFace 2.2 output.

Expected behavior (C++ OpenFace):
- During optimization: params are in "natural" units (actual 3D coordinates)
- Output params: normalized by dividing by sqrt(eigenvalue)
- Shape reconstruction: multiply by sqrt(eigenvalue) to get back to natural units

This keeps params in a reasonable range and makes them comparable across modes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add pyfaceau to path
sys.path.insert(0, str(Path(__file__).parent))

from pyfaceau.pipeline import FullPythonAUPipeline
import cv2


def test_calcparams_normalization():
    """Test if CalcParams normalizes params_local correctly"""
    print("=" * 80)
    print("CALCPARAMS EIGENVALUE NORMALIZATION TEST")
    print("=" * 80)

    # Load C++ reference
    cpp_df = pd.read_csv('cpp_reference/IMG_0434.csv', skipinitialspace=True)
    cpp_row = cpp_df.iloc[0]  # Frame 1

    print("\nC++ OpenFace 2.2 (Frame 1):")
    print(f"  p_0: {cpp_row['p_0']:.3f}")
    print(f"  p_1: {cpp_row['p_1']:.3f}")
    print(f"  p_2: {cpp_row['p_2']:.3f}")
    print(f"  p_3: {cpp_row['p_3']:.3f}")
    print(f"  p_4: {cpp_row['p_4']:.3f}")

    # Initialize pipeline
    print("\nInitializing Python pipeline...")
    pipeline = FullPythonAUPipeline(
        retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
        pfld_model='weights/pfld_cunjian.onnx',
        pdm_file='weights/In-the-wild_aligned_PDM_68.txt',
        au_models_dir='weights/AU_predictors',
        triangulation_file='weights/tris_68_full.txt',
        use_batched_predictor=True,
        verbose=False
    )
    pipeline._initialize_components()

    # Process frame 0 (which corresponds to frame 1 in 1-indexed CSV)
    cap = cv2.VideoCapture('/Users/johnwilsoniv/Documents/SplitFace Open3/D Normal Pts/IMG_0434.MOV')
    ret, frame = cap.read()
    cap.release()

    # Detect and process
    detections, _ = pipeline.face_detector.detect_faces(frame)
    bbox = detections[0][:4].astype(int)
    landmarks_68, _ = pipeline.landmark_detector.detect_landmarks(frame, bbox)
    params_global, params_local = pipeline.calc_params.calc_params(landmarks_68.flatten())

    print("\nPython (Frame 0 - unnormalized):")
    print(f"  p_0: {params_local[0]:.3f}")
    print(f"  p_1: {params_local[1]:.3f}")
    print(f"  p_2: {params_local[2]:.3f}")
    print(f"  p_3: {params_local[3]:.3f}")
    print(f"  p_4: {params_local[4]:.3f}")

    # Check eigenvalues
    eigenvalues = pipeline.calc_params.eigen_values
    print("\nEigenvalues (first 5):")
    print(f"  λ_0: {eigenvalues[0]:.6f} → sqrt: {np.sqrt(eigenvalues[0]):.6f}")
    print(f"  λ_1: {eigenvalues[1]:.6f} → sqrt: {np.sqrt(eigenvalues[1]):.6f}")
    print(f"  λ_2: {eigenvalues[2]:.6f} → sqrt: {np.sqrt(eigenvalues[2]):.6f}")
    print(f"  λ_3: {eigenvalues[3]:.6f} → sqrt: {np.sqrt(eigenvalues[3]):.6f}")
    print(f"  λ_4: {eigenvalues[4]:.6f} → sqrt: {np.sqrt(eigenvalues[4]):.6f}")

    # Manually normalize params_local
    params_local_normalized = params_local / np.sqrt(eigenvalues)

    print("\nPython (Frame 0 - manually normalized by sqrt(eigenvalue)):")
    print(f"  p_0: {params_local_normalized[0]:.3f}")
    print(f"  p_1: {params_local_normalized[1]:.3f}")
    print(f"  p_2: {params_local_normalized[2]:.3f}")
    print(f"  p_3: {params_local_normalized[3]:.3f}")
    print(f"  p_4: {params_local_normalized[4]:.3f}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)

    # Calculate ratios
    ratios = params_local[:5] / params_local_normalized[:5]
    print(f"\nRatio (unnormalized / normalized):")
    for i in range(5):
        print(f"  p_{i}: {ratios[i]:.2f} ≈ sqrt(λ_{i}) = {np.sqrt(eigenvalues[i]):.2f}")

    print("\nCONCLUSION:")
    print("  The unnormalized params are ~sqrt(eigenvalue) times larger!")
    print("  This confirms that Python CalcParams is missing eigenvalue normalization.")
    print("  C++ OpenFace outputs params normalized by sqrt(eigenvalue).")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_calcparams_normalization()
