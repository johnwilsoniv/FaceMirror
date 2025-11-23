#!/usr/bin/env python3
"""
Debug script to compare C++ and Python CCNF response computation
for the SAME landmark at the SAME position.
"""

import numpy as np
import cv2
import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

from pyclnf.clnf import CLNF

def main():
    # Use same test image as test_eye_refinement.py
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    frame_idx = 30

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load C++ landmarks from the test output
    import pandas as pd
    csv_path = '/tmp/openface_verify/shorty_frame_30.csv'
    try:
        df = pd.read_csv(csv_path)
        cpp_landmarks = np.zeros((68, 2))
        for i in range(68):
            # Try both formats (with and without space)
            for x_col, y_col in [(f' x_{i}', f' y_{i}'), (f'x_{i}', f'y_{i}')]:
                if x_col in df.columns and y_col in df.columns:
                    cpp_landmarks[i, 0] = df[x_col].iloc[0]
                    cpp_landmarks[i, 1] = df[y_col].iloc[0]
                    break
    except Exception as e:
        print(f"Error: Could not load C++ landmarks: {e}")
        return

    # Get eye landmark 36 position (outer right eye corner)
    lm36_pos = cpp_landmarks[36]
    print(f"Landmark 36 position: ({lm36_pos[0]:.2f}, {lm36_pos[1]:.2f})")

    # Extract patch at this position (like eye model would)
    # Eye model uses window_size=3, patch_size=11, so AOI=13
    ws = 3
    patch_size = 11
    aoi_size = ws + patch_size - 1  # 13
    half_aoi = (aoi_size - 1) / 2.0  # 6

    x, y = lm36_pos

    # Create transformation matrix (identity rotation like Python eye model)
    a1 = 1.0
    b1 = 0.0
    tx = x - a1 * half_aoi + b1 * half_aoi
    ty = y - a1 * half_aoi - b1 * half_aoi

    sim = np.array([[a1, -b1, tx],
                   [b1, a1, ty]], dtype=np.float32)

    # Extract area of interest
    area_of_interest = cv2.warpAffine(
        gray.astype(np.float32),
        sim,
        (aoi_size, aoi_size),
        flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR
    )

    print(f"\nExtracted AOI at landmark 36:")
    print(f"AOI shape: {area_of_interest.shape}")
    print(f"AOI range: {area_of_interest.min():.1f} to {area_of_interest.max():.1f}")
    print(f"\nFirst 5x5 pixels:")
    for row in range(5):
        values = [f"{area_of_interest[row, col]:.1f}" for col in range(5)]
        print(f"  {' '.join(values)}")

    # Now let's also check what C++ extracts
    # The C++ debug file shows the main model with window_size=11
    # But we need the eye model debug

    print("\n" + "="*60)
    print("COMPARISON WITH C++ EYE MODEL")
    print("="*60)

    # Check if there's C++ eye model debug
    try:
        with open('/tmp/cpp_eye_response_maps.txt', 'r') as f:
            content = f.read()
            print("\nC++ Eye Model Response Maps exist")
            # Parse to find landmark 8 (which maps to main 36)
            if 'Eye landmark 8' in content:
                print("Found Eye landmark 8 data")
    except:
        print("No C++ eye response maps file")

    # Load Python CLNF with eye model
    print("\n" + "="*60)
    print("PYTHON EYE MODEL PATCH EXTRACTION")
    print("="*60)

    clnf = CLNF(use_eye_refinement=True)

    if clnf.eye_model is None:
        print("Eye model not loaded!")
        return

    # Map main landmark 36 to eye landmark 8
    mapping = {36: 8, 37: 10, 38: 12, 39: 14, 40: 16, 41: 18}

    # Get the eye model's patch expert for landmark 8
    eye_ccnf = clnf.eye_model.ccnf['left']
    patch_experts = eye_ccnf.get_all_patch_experts(1.0)  # scale 1.0

    if 8 not in patch_experts:
        print("No patch expert for eye landmark 8!")
        print(f"Available landmarks: {list(patch_experts.keys())}")
        return

    patch_expert = patch_experts[8]
    print(f"\nEye landmark 8 patch expert:")
    print(f"  Size: {patch_expert.height}x{patch_expert.width}")
    print(f"  Neurons: {patch_expert.num_neurons}")

    # Check neuron parameters
    print(f"\nNeuron parameters:")
    for i, neuron in enumerate(patch_expert.neurons):
        print(f"  Neuron {i}: alpha={neuron['alpha']:.4f}, bias={neuron['bias']:.4f}, norm_w={neuron['norm_weights']:.4f}")

    # Extract center patch from AOI
    center_patch = area_of_interest[1:1+patch_size, 1:1+patch_size]
    print(f"\nCenter patch shape: {center_patch.shape}")
    print(f"Center patch range: {center_patch.min():.1f} to {center_patch.max():.1f}")

    # Compute response
    response = patch_expert.compute_response(
        center_patch.astype(np.uint8),
        debug_file='/tmp/python_eye_lm8_debug.txt'
    )

    print(f"\nComputed response: {response:.6f}")
    print(f"\nDetailed debug saved to /tmp/python_eye_lm8_debug.txt")

if __name__ == '__main__':
    main()
