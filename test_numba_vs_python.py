#!/usr/bin/env python3
"""
Test if Numba JIT is causing accuracy issues by comparing with pure Python.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
import pandas as pd

def load_openface_landmarks(csv_path: str, frame_num: int) -> np.ndarray:
    """Load OpenFace landmarks for a specific frame."""
    df = pd.read_csv(csv_path)
    row = df[df['frame'] == frame_num].iloc[0]

    landmarks = np.zeros((68, 2))
    for i in range(68):
        for x_col, y_col in [(f'x_{i}', f'y_{i}'), (f' x_{i}', f' y_{i}')]:
            if x_col in df.columns and y_col in df.columns:
                landmarks[i, 0] = row[x_col]
                landmarks[i, 1] = row[y_col]
                break
    return landmarks

def main():
    video_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/IMG_0942.MOV'
    csv_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/IMG_0942.csv'
    frame_num = 100

    # Load frame and C++ landmarks
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    cpp_lm = load_openface_landmarks(csv_path, frame_num)

    print("="*60)
    print("Testing Numba JIT vs Pure Python Accuracy")
    print("="*60)

    # Test 1: With Numba JIT (current implementation)
    print("\n1. With Numba JIT acceleration:")
    from pyclnf.clnf import CLNF
    clnf = CLNF('pyclnf/pyclnf/models', regularization=40)
    result = clnf.detect_and_fit(frame)
    py_lm = result[0]

    errors_jit = np.sqrt(np.sum((cpp_lm - py_lm)**2, axis=1))
    print(f"   Mean error: {np.mean(errors_jit):.2f}px")
    print(f"   Jaw (0-16): {np.mean(errors_jit[0:17]):.2f}px")
    print(f"   Inner face (27-67): {np.mean(errors_jit[27:68]):.2f}px")

    # Test 2: Disable Numba by patching
    print("\n2. Testing with Numba disabled (patching)...")

    # Check what's in the optimizer module
    from pyclnf.core import optimizer
    print(f"   USE_NUMBA = {optimizer.USE_NUMBA}")

    # If USE_NUMBA is True, the issue might be in the JIT functions
    # Let's check the response map values

    print("\n3. Checking response map computation...")
    # Get the same frame and check internal values
    from pyclnf.core.patch_expert import CCNFModel
    ccnf = CCNFModel('pyclnf/pyclnf/models')

    # Get patch expert for landmark 30 (nose tip)
    patch_experts_25 = ccnf.scale_models[0.25]['views'][0]['patches']
    expert = patch_experts_25[30]

    # Create a test patch from the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    test_patch = gray[500:511, 500:511].astype(np.float64)

    # Test the JIT vs pure Python
    from pyclnf.core.numba_accelerator import compute_patch_response_jit
    jit_response = compute_patch_response_jit(
        test_patch,
        expert._batched_weights,
        expert._batched_biases,
        expert._batched_alphas,
        expert._batched_norm_weights,
        expert.num_neurons
    )

    # Pure Python version
    python_response = expert.compute_response(test_patch)

    print(f"   JIT response: {jit_response:.6f}")
    print(f"   Python response: {python_response:.6f}")
    print(f"   Difference: {abs(jit_response - python_response):.10f}")

    # Test Jacobian
    print("\n4. Testing Jacobian computation...")
    from pyclnf.core.pdm import PDM
    pdm = PDM('pyclnf/pyclnf/models/exported_pdm')
    params = pdm.init_params((300, 700, 500, 500))

    # Compute with and without JIT
    J = pdm.compute_jacobian(params)
    print(f"   Jacobian shape: {J.shape}")
    print(f"   Jacobian norm: {np.linalg.norm(J):.4f}")

if __name__ == '__main__':
    main()
