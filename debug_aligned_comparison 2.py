#!/usr/bin/env python3
"""
Aligned comparison of C++ and Python eye CCNF response computation.

This script:
1. Runs C++ OpenFace to get exact parameters at first eye iteration
2. Runs Python with identical parameters
3. Compares AOI extraction and response maps for the same landmark

Goal: Find why landmarks 0, 1 have opposite X mean-shifts but landmark 8 matches.
"""

import numpy as np
import cv2
import subprocess
import struct
import sys
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3')

from pyclnf.core.eye_patch_expert import HierarchicalEyeModel, align_shapes_with_scale, EyeCCNFPatchExpert
from pyclnf.core.eye_pdm import EyePDM

def run_cpp_and_get_params():
    """Run C++ OpenFace to get eye model parameters at first iteration."""
    # Run OpenFace with debug output
    video_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov'
    cmd = [
        '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction',
        '-f', video_path,
        '-out_dir', '/tmp',
        '-2Dfp',
        '-pose',
        '-aus',
        '-singleFrame', '0'
    ]

    print("Running C++ OpenFace...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse debug output from /tmp/cpp_eye_model_detailed.txt
    try:
        with open('/tmp/cpp_eye_model_detailed.txt', 'r') as f:
            content = f.read()

        # Extract initial global params
        import re
        match = re.search(r'Initial global params: scale=([\d.]+) rot=\(([-\d.]+),([-\d.]+),([-\d.]+)\) tx=([\d.]+) ty=([\d.]+)', content)
        if match:
            params = {
                'scale': float(match.group(1)),
                'rx': float(match.group(2)),
                'ry': float(match.group(3)),
                'rz': float(match.group(4)),
                'tx': float(match.group(5)),
                'ty': float(match.group(6))
            }
            return params
    except Exception as e:
        print(f"Error parsing C++ output: {e}")

    return None

def compare_all_landmarks():
    """Compare response maps for all 28 eye landmarks."""
    print("=" * 70)
    print("ALIGNED C++ vs PYTHON EYE CCNF COMPARISON")
    print("=" * 70)

    # Load image
    video_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov'
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    print(f"Image shape: {gray.shape}")

    # Load eye PDM
    model_dir = '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models/exported_eye_pdm_left'
    left_eye_pdm = EyePDM(model_dir)

    # Use exact C++ initial params
    # From debug: scale=3.371204 rot=(-0.118319,0.176098,-0.099366) tx=425.031629 ty=820.112023
    params = np.zeros(left_eye_pdm.n_params)
    params[0] = 3.371204  # scale
    params[1] = -0.118319  # rx
    params[2] = 0.176098   # ry
    params[3] = -0.099366  # rz
    params[4] = 425.031629 # tx
    params[5] = 820.112023 # ty

    # Get all 28 eye landmarks
    eye_landmarks = left_eye_pdm.params_to_landmarks_2d(params)

    print(f"\nInitial eye PDM params:")
    print(f"  scale={params[0]:.6f}")
    print(f"  rot=({params[1]:.6f}, {params[2]:.6f}, {params[3]:.6f})")
    print(f"  tx={params[4]:.6f}, ty={params[5]:.6f}")

    # Show landmark positions
    print(f"\nLandmark positions (first 5 and key landmarks):")
    for i in [0, 1, 2, 3, 4, 8, 10, 12, 14, 16, 18]:
        print(f"  Landmark {i}: ({eye_landmarks[i, 0]:.4f}, {eye_landmarks[i, 1]:.4f})")

    # Compute similarity transform
    ref_params = params.copy()
    ref_params[0] = 1.0  # patch_scale
    ref_params[1:4] = 0  # no rotation
    ref_params[4:6] = 0  # no translation
    reference_shape = left_eye_pdm.params_to_landmarks_2d(ref_params)

    sim_img_to_ref = align_shapes_with_scale(eye_landmarks, reference_shape)
    sim_ref_to_img = np.linalg.inv(sim_img_to_ref)

    a1 = sim_ref_to_img[0, 0]
    b1 = -sim_ref_to_img[0, 1]

    print(f"\nSimilarity transform:")
    print(f"  a1 = {a1:.6f}")
    print(f"  b1 = {b1:.6f}")

    # Load CCNF patch experts
    ccnf_dir = '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models/exported_eye_ccnf_left/scale_1.00/view_00'

    # Test multiple landmarks including problematic ones (0, 1) and working one (8)
    test_landmarks = [0, 1, 8]
    window_size = 3
    patch_size = 11
    aoi_size = window_size + patch_size - 1
    half_aoi = (aoi_size - 1) / 2.0

    print(f"\n{'=' * 70}")
    print(f"RESPONSE MAP COMPARISON (window_size={window_size})")
    print(f"{'=' * 70}")

    results = {}

    for lm_idx in test_landmarks:
        # Load patch expert
        patch_dir = f'{ccnf_dir}/patch_{lm_idx:02d}'
        try:
            patch_expert = EyeCCNFPatchExpert(patch_dir)
        except Exception as e:
            print(f"\nLandmark {lm_idx}: No patch expert - {e}")
            continue

        # Get landmark position
        x, y = eye_landmarks[lm_idx]

        # Compute warpAffine transform
        tx = x - a1 * half_aoi + b1 * half_aoi
        ty = y - a1 * half_aoi - b1 * half_aoi

        sim = np.array([[a1, -b1, tx],
                        [b1, a1, ty]], dtype=np.float32)

        # Extract AOI
        area_of_interest = cv2.warpAffine(
            gray, sim, (aoi_size, aoi_size),
            flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR
        )

        print(f"\n--- Landmark {lm_idx} ---")
        print(f"Position: ({x:.4f}, {y:.4f})")
        print(f"AOI stats: min={area_of_interest.min():.1f}, max={area_of_interest.max():.1f}, mean={area_of_interest.mean():.1f}")

        # Compute response map
        response_map = np.zeros((window_size, window_size), dtype=np.float32)

        for i in range(window_size):
            for j in range(window_size):
                patch = area_of_interest[i:i+patch_size, j:j+patch_size]
                response = patch_expert.compute_response(patch.astype(np.uint8))
                response_map[i, j] = response

        # Make non-negative
        min_val = response_map.min()
        if min_val < 0:
            response_map = response_map - min_val

        # Compute mean-shift
        sigma = 1.0
        a_kde = -0.5 / (sigma * sigma)
        center = (window_size - 1) / 2.0

        total_weight = 0.0
        mx = 0.0
        my = 0.0

        for ii in range(window_size):
            for jj in range(window_size):
                dist_sq = (ii - center)**2 + (jj - center)**2
                kde_weight = np.exp(a_kde * dist_sq)
                weight = response_map[ii, jj] * kde_weight
                total_weight += weight
                mx += weight * jj
                my += weight * ii

        if total_weight > 1e-10:
            ms_x = (mx / total_weight) - center
            ms_y = (my / total_weight) - center
        else:
            ms_x = 0.0
            ms_y = 0.0

        # Transform to image space
        ms_img_x = ms_x * sim_ref_to_img[0, 0] + ms_y * sim_ref_to_img[0, 1]
        ms_img_y = ms_x * sim_ref_to_img[1, 0] + ms_y * sim_ref_to_img[1, 1]

        print(f"\nResponse map ({window_size}x{window_size}):")
        for row in range(window_size):
            row_str = "  "
            for col in range(window_size):
                row_str += f"{response_map[row, col]:.4f} "
            print(row_str)

        print(f"\nMean-shift (ref space): ({ms_x:.6f}, {ms_y:.6f})")
        print(f"Mean-shift (img space): ({ms_img_x:.6f}, {ms_img_y:.6f})")

        results[lm_idx] = {
            'position': (x, y),
            'response_map': response_map,
            'mean_shift_ref': (ms_x, ms_y),
            'mean_shift_img': (ms_img_x, ms_img_y),
            'aoi': area_of_interest
        }

    # Summary comparison with C++ values
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH C++ VALUES")
    print(f"{'=' * 70}")

    # C++ values from debug (first iteration, window=3)
    cpp_mean_shifts = {
        0: (-0.47, -1.05),
        1: (-0.39, -1.21),
        8: (1.16, -0.22)
    }

    print("\n| Landmark | Python MS (img) | C++ MS | X Match? | Y Match? |")
    print("|----------|-----------------|--------|----------|----------|")

    for lm_idx in test_landmarks:
        if lm_idx in results and lm_idx in cpp_mean_shifts:
            py_ms = results[lm_idx]['mean_shift_img']
            cpp_ms = cpp_mean_shifts[lm_idx]

            # Check if signs match
            x_match = "✅" if (py_ms[0] * cpp_ms[0] > 0 or abs(py_ms[0]) < 0.1) else "❌"
            y_match = "✅" if (py_ms[1] * cpp_ms[1] > 0 or abs(py_ms[1]) < 0.1) else "❌"

            print(f"| {lm_idx:8d} | ({py_ms[0]:6.2f}, {py_ms[1]:6.2f}) | ({cpp_ms[0]:5.2f}, {cpp_ms[1]:5.2f}) | {x_match:8s} | {y_match:8s} |")

    # Save AOI for landmarks 0 and 8 for manual comparison
    if 0 in results:
        np.save('/tmp/python_aoi_lm0.npy', results[0]['aoi'])
        print(f"\nSaved AOI for landmark 0 to /tmp/python_aoi_lm0.npy")

    if 8 in results:
        np.save('/tmp/python_aoi_lm8.npy', results[8]['aoi'])
        print(f"Saved AOI for landmark 8 to /tmp/python_aoi_lm8.npy")

    return results

def check_iris_landmark_initialization():
    """Check how landmarks 0, 1 (iris) are initialized vs eyelid landmarks."""
    print(f"\n{'=' * 70}")
    print("IRIS LANDMARK INITIALIZATION CHECK")
    print(f"{'=' * 70}")

    model_dir = '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models/exported_eye_pdm_left'
    left_eye_pdm = EyePDM(model_dir)

    # Load mean shape
    mean_shape = np.load(f'{model_dir}/mean_shape.npy')

    print(f"\nMean shape (first 10 landmarks):")
    for i in range(10):
        print(f"  Landmark {i}: ({mean_shape[i, 0]:.4f}, {mean_shape[i, 1]:.4f})")

    print(f"\nEyelid landmarks in mean shape:")
    for i in [8, 10, 12, 14, 16, 18]:
        print(f"  Landmark {i}: ({mean_shape[i, 0]:.4f}, {mean_shape[i, 1]:.4f})")

    # Check the reference shape at scale=1
    params = np.zeros(left_eye_pdm.n_params)
    params[0] = 1.0  # scale = 1
    ref_landmarks = left_eye_pdm.params_to_landmarks_2d(params)

    print(f"\nReference shape at scale=1 (first 10 landmarks):")
    for i in range(10):
        print(f"  Landmark {i}: ({ref_landmarks[i, 0]:.4f}, {ref_landmarks[i, 1]:.4f})")

def verify_patch_expert_weights():
    """Verify that exported patch expert weights match expected values."""
    print(f"\n{'=' * 70}")
    print("PATCH EXPERT WEIGHT VERIFICATION")
    print(f"{'=' * 70}")

    # Load patch experts for landmarks 0, 1, 8
    ccnf_dir = '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models/exported_eye_ccnf_left/scale_1.00/view_00'

    for lm_idx in [0, 1, 8]:
        patch_dir = f'{ccnf_dir}/patch_{lm_idx:02d}'
        try:
            patch_expert = EyeCCNFPatchExpert(patch_dir)

            print(f"\nLandmark {lm_idx} patch expert:")
            print(f"  Size: {patch_expert.width}x{patch_expert.height}")
            print(f"  Num neurons: {patch_expert.num_neurons}")
            print(f"  Patch confidence: {patch_expert.patch_confidence:.6f}")

            # Check neuron details
            sum_alphas = 0
            for i, neuron in enumerate(patch_expert.neurons):
                if abs(neuron['alpha']) > 1e-4:
                    sum_alphas += neuron['alpha']
                    if i < 3:  # Show first 3 neurons
                        print(f"  Neuron {i}: alpha={neuron['alpha']:.4f}, bias={neuron['bias']:.4f}, norm_w={neuron['norm_weights']:.4f}")

            print(f"  Sum of alphas: {sum_alphas:.4f}")

        except Exception as e:
            print(f"\nLandmark {lm_idx}: Error loading - {e}")

if __name__ == '__main__':
    # Step 1 & 4: Compare response maps at aligned iteration
    results = compare_all_landmarks()

    # Step 2: Check iris landmark initialization
    check_iris_landmark_initialization()

    # Step 3: Verify patch expert weights
    verify_patch_expert_weights()
