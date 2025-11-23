#!/usr/bin/env python3
"""
Debug alignment of shapes between C++ and Python.
Focus on the mean-shift transformation matrix.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3')

from pyclnf.core.eye_patch_expert import align_shapes_with_scale
from pyclnf.core.eye_pdm import EyePDM

def main():
    print("=" * 70)
    print("DEBUG: ALIGN_SHAPES_WITH_SCALE vs C++ AlignShapesWithScale")
    print("=" * 70)

    # Load eye PDM
    model_dir = '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models/exported_eye_pdm_left'
    left_eye_pdm = EyePDM(model_dir)

    # Initial params from C++ debug
    params = np.zeros(left_eye_pdm.n_params)
    params[0] = 3.371204  # scale
    params[1] = -0.118319  # rx
    params[2] = 0.176098   # ry
    params[3] = -0.099366  # rz
    params[4] = 425.031629 # tx
    params[5] = 820.112023 # ty

    # Get image landmarks
    eye_landmarks = left_eye_pdm.params_to_landmarks_2d(params)

    # Get reference landmarks (scale=1, no rotation, no translation)
    ref_params = params.copy()
    ref_params[0] = 1.0  # patch_scale
    ref_params[1:4] = 0  # no rotation
    ref_params[4:6] = 0  # no translation
    reference_shape = left_eye_pdm.params_to_landmarks_2d(ref_params)

    print(f"\nImage shape (first 3 points):")
    for i in range(3):
        print(f"  Point {i}: ({eye_landmarks[i, 0]:.4f}, {eye_landmarks[i, 1]:.4f})")

    print(f"\nReference shape (first 3 points):")
    for i in range(3):
        print(f"  Point {i}: ({reference_shape[i, 0]:.4f}, {reference_shape[i, 1]:.4f})")

    # Compute transform
    sim_img_to_ref = align_shapes_with_scale(eye_landmarks, reference_shape)
    sim_ref_to_img = np.linalg.inv(sim_img_to_ref)

    print(f"\nPython sim_img_to_ref:")
    print(f"  [{sim_img_to_ref[0, 0]:10.6f}, {sim_img_to_ref[0, 1]:10.6f}]")
    print(f"  [{sim_img_to_ref[1, 0]:10.6f}, {sim_img_to_ref[1, 1]:10.6f}]")

    print(f"\nPython sim_ref_to_img:")
    print(f"  [{sim_ref_to_img[0, 0]:10.6f}, {sim_ref_to_img[0, 1]:10.6f}]")
    print(f"  [{sim_ref_to_img[1, 0]:10.6f}, {sim_ref_to_img[1, 1]:10.6f}]")

    # Expected C++ values from debug file
    print("\nExpected C++ sim_ref_to_img (from debug):")
    print("  Should be approximately [[3.254, 0.396], [-0.396, 3.254]]")

    # Extract a1, b1 like C++
    a1 = sim_ref_to_img[0, 0]
    b1_cpp_style = -sim_ref_to_img[0, 1]  # C++ uses -sim_ref_to_img(0,1)

    print(f"\nExtracted a1 = {a1:.6f}")
    print(f"Extracted b1 (C++ style -[0,1]) = {b1_cpp_style:.6f}")

    # Now test mean-shift transformation
    print("\n" + "=" * 70)
    print("TESTING MEAN-SHIFT TRANSFORMATION")
    print("=" * 70)

    # Test mean-shift: (1.0, -0.5) in reference space
    test_ms_ref = np.array([[1.0, -0.5]])

    # Python transformation: ms_img = ms_ref @ sim_ref_to_img.T
    ms_img_python = test_ms_ref @ sim_ref_to_img.T

    # C++ uses row vectors and right-multiplies:
    # mean_shifts_2D = mean_shifts_2D * cv::Mat(sim_ref_to_img).t()
    # In numpy that would be: ms_ref @ sim_ref_to_img.T (same as Python!)

    print(f"\nTest mean-shift in ref space: {test_ms_ref}")
    print(f"Transformed to image space: {ms_img_python}")

    # Alternative: column vector approach
    ms_ref_col = test_ms_ref.T  # (2, 1)
    ms_img_col = sim_ref_to_img @ ms_ref_col  # (2, 1)

    print(f"\nAlternative (column vector): {ms_img_col.T}")

    # Check if rotation matrix is proper
    print("\n" + "=" * 70)
    print("ROTATION MATRIX ANALYSIS")
    print("=" * 70)

    # A proper rotation matrix should have form [[cos, -sin], [sin, cos]]
    # With scale: [[s*cos, -s*sin], [s*sin, s*cos]]

    # From sim_ref_to_img
    r00 = sim_ref_to_img[0, 0]  # s*cos
    r01 = sim_ref_to_img[0, 1]  # should be -s*sin
    r10 = sim_ref_to_img[1, 0]  # should be s*sin
    r11 = sim_ref_to_img[1, 1]  # s*cos

    print(f"\nsim_ref_to_img analysis:")
    print(f"  [0,0] (s*cos): {r00:.6f}")
    print(f"  [0,1] (should be -s*sin): {r01:.6f}")
    print(f"  [1,0] (should be s*sin): {r10:.6f}")
    print(f"  [1,1] (s*cos): {r11:.6f}")

    # Check: [0,1] should be -[1,0]
    if abs(r01 + r10) < 0.001:
        print(f"\n✅ Matrix is proper rotation: [0,1]={r01:.4f}, -[1,0]={-r10:.4f}")
    else:
        print(f"\n❌ Matrix is NOT proper rotation: [0,1]={r01:.4f}, -[1,0]={-r10:.4f}")

    # Compute scale and angle
    scale = np.sqrt(r00**2 + r10**2)
    angle = np.arctan2(r10, r00)

    print(f"\nExtracted scale: {scale:.6f}")
    print(f"Extracted angle: {np.degrees(angle):.2f} degrees")

    # Verify with expected scale from params
    expected_scale = params[0]  # 3.371204
    print(f"Expected scale (from params): {expected_scale:.6f}")

    # The extracted scale should be close to expected
    if abs(scale - expected_scale) < 0.1:
        print(f"✅ Scale matches expected")
    else:
        print(f"❌ Scale differs: {scale:.4f} vs {expected_scale:.4f}")

    # Now check the actual issue: the mean-shift direction
    print("\n" + "=" * 70)
    print("ACTUAL MEAN-SHIFT ISSUE")
    print("=" * 70)

    # From debug: Python landmark 0 has mean-shift (0.47, -1.01) in ref space
    # But when transformed to image space, it goes wrong direction

    # The actual response map for landmark 0 shows peak at column 2-3 (right side)
    # This gives mean-shift pointing RIGHT in response map coordinates

    # But the transform is from REFERENCE space to IMAGE space
    # The response map is extracted in reference space orientation

    print("\nThe issue may be:")
    print("1. Response map is computed in reference space (rotated/scaled)")
    print("2. Mean-shift in reference space points toward response map peak")
    print("3. Transform to image space should map direction correctly")
    print("")
    print("Check: When a1=3.25, b1=0.40 (from -[0,1])")
    print("The warpAffine uses [[a1, -b1], [b1, a1]] = [[3.25, -0.40], [0.40, 3.25]]")
    print("This is the INVERSE mapping (image coord to AOI coord)")
    print("")
    print("But we computed sim_ref_to_img which maps ref to image.")
    print("The sign of [0,1] determines if it matches the warp or not.")

    # Compare with what C++ debug shows
    print("\n" + "=" * 70)
    print("C++ DEBUG VALUES COMPARISON")
    print("=" * 70)

    # From /tmp/cpp_eye_model_detailed.txt we should have C++ sim_ref_to_img values
    # Let's read them if available
    try:
        with open('/tmp/cpp_eye_model_detailed.txt', 'r') as f:
            content = f.read()

        import re
        match = re.search(r'sim_ref_to_img:\s*\n\s*\[([-\d.]+),\s*([-\d.]+)\]\s*\n\s*\[([-\d.]+),\s*([-\d.]+)\]', content)
        if match:
            cpp_r2i = np.array([
                [float(match.group(1)), float(match.group(2))],
                [float(match.group(3)), float(match.group(4))]
            ])
            print(f"\nC++ sim_ref_to_img:")
            print(f"  [{cpp_r2i[0, 0]:10.6f}, {cpp_r2i[0, 1]:10.6f}]")
            print(f"  [{cpp_r2i[1, 0]:10.6f}, {cpp_r2i[1, 1]:10.6f}]")

            print(f"\nDifference (Python - C++):")
            diff = sim_ref_to_img - cpp_r2i
            print(f"  [{diff[0, 0]:10.6f}, {diff[0, 1]:10.6f}]")
            print(f"  [{diff[1, 0]:10.6f}, {diff[1, 1]:10.6f}]")

            if np.max(np.abs(diff)) < 0.001:
                print("\n✅ Matrices match!")
            else:
                print(f"\n❌ Matrices differ by up to {np.max(np.abs(diff)):.4f}")
        else:
            print("\nCould not find sim_ref_to_img in C++ debug file")
    except Exception as e:
        print(f"\nCould not read C++ debug file: {e}")

if __name__ == '__main__':
    main()
