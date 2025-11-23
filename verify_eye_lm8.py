#!/usr/bin/env python3
"""
Verify Python eye refinement matches C++ for landmark 8, window_size=3.

This script loads C++ debug output and compares Python computation at each stage:
1. AOI extraction (warpAffine)
2. CCNF response map computation
3. Mean-shift calculation
"""

import numpy as np
import cv2
import struct
from pathlib import Path

# Add project paths
import sys
sys.path.insert(0, str(Path(__file__).parent / "pyclnf"))

from pyclnf.core.eye_patch_expert import EyeCCNFPatchExpert


def load_cpp_binary(filepath):
    """Load binary file with rows, cols header followed by float32 data."""
    with open(filepath, 'rb') as f:
        rows = struct.unpack('i', f.read(4))[0]
        cols = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.float32)
        return data.reshape(rows, cols)


def load_cpp_params(filepath):
    """Parse the params text file."""
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key == 'landmark_8_pos':
                    # Parse "(x, y)"
                    value = value.strip('()')
                    x, y = map(float, value.split(','))
                    params['lm_x'] = x
                    params['lm_y'] = y
                elif key in ['a1', 'b1', 'n_landmarks', 'window_size']:
                    params[key] = float(value) if '.' in value else int(value)
                elif key == 'aoi_size':
                    w, h = value.split(' x ')
                    params['aoi_width'] = int(w)
                    params['aoi_height'] = int(h)
    return params


def load_cpp_meanshift(filepath):
    """Parse the mean-shift text file."""
    ms = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    ms[key] = float(value)
                except ValueError:
                    pass
    return ms


def main():
    print("=" * 60)
    print("Python vs C++ Eye Landmark 8 Verification")
    print("=" * 60)

    # Load C++ debug data
    cpp_aoi = load_cpp_binary('/tmp/cpp_eye_lm8_aoi.bin')
    cpp_response = load_cpp_binary('/tmp/cpp_eye_lm8_response.bin')
    cpp_params = load_cpp_params('/tmp/cpp_eye_lm8_params.txt')
    cpp_ms = load_cpp_meanshift('/tmp/cpp_eye_lm8_meanshift.txt')

    print("\n[C++ Reference Values]")
    print(f"  AOI shape: {cpp_aoi.shape}")
    print(f"  AOI stats: min={cpp_aoi.min():.4f}, max={cpp_aoi.max():.4f}, mean={cpp_aoi.mean():.4f}")
    print(f"  Response shape: {cpp_response.shape}")
    print(f"  Response stats: min={cpp_response.min():.6f}, max={cpp_response.max():.6f}, mean={cpp_response.mean():.6f}")
    print(f"  Mean-shift: msx={cpp_ms.get('msx (mx/sum - dx)', 0):.6f}, msy={cpp_ms.get('msy (my/sum - dy)', 0):.6f}")
    print(f"  Params: a1={cpp_params['a1']:.6f}, b1={cpp_params['b1']:.6f}")
    print(f"  Landmark pos: ({cpp_params['lm_x']:.6f}, {cpp_params['lm_y']:.6f})")

    # Load test image
    img_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_frame_0030.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Could not load image: {img_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    print(f"\n[Image]")
    print(f"  Shape: {gray.shape}")
    print(f"  Dtype: {gray.dtype}")

    # Step 1: Extract AOI using same transform as C++
    a1 = cpp_params['a1']
    b1 = cpp_params['b1']
    lm_x = cpp_params['lm_x']
    lm_y = cpp_params['lm_y']
    aoi_width = cpp_params['aoi_width']
    aoi_height = cpp_params['aoi_height']

    # Build the same affine transform as C++
    # IMPORTANT: C++ uses area_of_interest_width for ALL terms (not height!)
    # cv::Mat sim = (cv::Mat_<float>(2, 3) <<
    #     a1, -b1, lm_x - a1*(w-1)/2 + b1*(w-1)/2,
    #     b1, a1, lm_y - a1*(w-1)/2 - b1*(w-1)/2);
    half_w = (aoi_width - 1) / 2.0
    tx = lm_x - a1 * half_w + b1 * half_w
    ty = lm_y - a1 * half_w - b1 * half_w

    sim = np.array([
        [a1, -b1, tx],
        [b1, a1, ty]
    ], dtype=np.float32)

    print(f"\n  Transform matrix:")
    print(f"    [{a1:.6f}, {-b1:.6f}, {tx:.6f}]")
    print(f"    [{b1:.6f}, {a1:.6f}, {ty:.6f}]")

    # Extract AOI with inverse warp (like C++)
    # C++ uses CV_32F grayscale image
    py_aoi = cv2.warpAffine(gray, sim, (aoi_width, aoi_height),
                            flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR)

    # Also try with borderMode to match C++ defaults
    py_aoi_border = cv2.warpAffine(gray, sim, (aoi_width, aoi_height),
                                   flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)

    # Check if border mode makes a difference
    border_diff = np.abs(py_aoi - py_aoi_border).max()
    if border_diff > 0.001:
        print(f"  Note: Border mode difference = {border_diff:.6f}")

    print(f"\n[Step 1: AOI Extraction]")
    print(f"  Python AOI shape: {py_aoi.shape}")
    print(f"  Python AOI stats: min={py_aoi.min():.4f}, max={py_aoi.max():.4f}, mean={py_aoi.mean():.4f}")
    print(f"  C++ AOI stats:    min={cpp_aoi.min():.4f}, max={cpp_aoi.max():.4f}, mean={cpp_aoi.mean():.4f}")

    aoi_diff = np.abs(py_aoi - cpp_aoi)
    print(f"  Max difference: {aoi_diff.max():.6f}")
    print(f"  Mean difference: {aoi_diff.mean():.6f}")

    if aoi_diff.max() < 0.01:
        print("  ✓ AOI MATCHES!")
    else:
        print("  ✗ AOI MISMATCH - need to investigate")
        # Print first few values for debugging
        print(f"  Python first row: {py_aoi[0, :5]}")
        print(f"  C++ first row:    {cpp_aoi[0, :5]}")

        # Check if it's a consistent offset
        diff_map = cpp_aoi - py_aoi
        print(f"\n  Investigating AOI difference:")
        print(f"    Diff range: [{diff_map.min():.4f}, {diff_map.max():.4f}]")
        print(f"    Diff mean: {diff_map.mean():.4f}")
        print(f"    Diff std: {diff_map.std():.4f}")

        # Check corners and center
        print(f"\n    Corner/center diffs:")
        print(f"      Top-left [0,0]: {diff_map[0,0]:.4f}")
        print(f"      Top-right [0,-1]: {diff_map[0,-1]:.4f}")
        print(f"      Bottom-left [-1,0]: {diff_map[-1,0]:.4f}")
        print(f"      Bottom-right [-1,-1]: {diff_map[-1,-1]:.4f}")
        print(f"      Center [6,6]: {diff_map[6,6]:.4f}")

        # Check if the difference correlates with position
        # (would indicate transform parameter difference)
        row_means = diff_map.mean(axis=1)
        col_means = diff_map.mean(axis=0)
        print(f"\n    Row-wise mean diff: {row_means}")
        print(f"    Col-wise mean diff: {col_means}")

    # Step 2: Compute CCNF response using EyeCCNFPatchExpert
    print(f"\n[Step 2: CCNF Response]")

    # Load the right eye patch expert for landmark 8
    model_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models/exported_eye_ccnf_right/scale_1.00/view_00/patch_08")

    if not model_dir.exists():
        print(f"  ERROR: Model directory not found: {model_dir}")
        return

    # Load patch expert
    expert = EyeCCNFPatchExpert(model_dir)
    print(f"  Patch size: {expert.height}x{expert.width}")
    print(f"  Num neurons: {expert.num_neurons}")

    # Compute response map by sliding patch across AOI
    window_size = cpp_params['window_size']
    patch_h = expert.height
    patch_w = expert.width
    resp_h = aoi_height - patch_h + 1
    resp_w = aoi_width - patch_w + 1

    py_response = np.zeros((resp_h, resp_w), dtype=np.float32)

    for i in range(resp_h):
        for j in range(resp_w):
            patch = py_aoi[i:i+patch_h, j:j+patch_w]
            py_response[i, j] = expert.compute_response(patch.astype(np.float32))

    print(f"  Python response shape: {py_response.shape}")
    print(f"  Python response stats: min={py_response.min():.6f}, max={py_response.max():.6f}, mean={py_response.mean():.6f}")
    print(f"  C++ response stats:    min={cpp_response.min():.6f}, max={cpp_response.max():.6f}, mean={cpp_response.mean():.6f}")

    resp_diff = np.abs(py_response - cpp_response)
    print(f"  Max difference: {resp_diff.max():.6f}")
    print(f"  Mean difference: {resp_diff.mean():.6f}")

    if resp_diff.max() < 0.01:
        print("  ✓ RESPONSE MATCHES!")
    else:
        print("  ✗ RESPONSE MISMATCH - need to investigate")
        print(f"  Python response:\n{py_response}")
        print(f"  C++ response:\n{cpp_response}")

        # Check ratio
        ratio = py_response / (cpp_response + 1e-10)
        print(f"  Ratio (Py/C++): min={ratio.min():.4f}, max={ratio.max():.4f}, mean={ratio.mean():.4f}")

    # Step 3: Compute mean-shift WITH KDE weighting
    print(f"\n[Step 3: Mean-Shift with KDE]")

    # dx, dy are the fractional offset within the response window
    # For centered landmark, dx=dy=1 for a 3x3 window (center is at index 1)
    dx = cpp_ms.get('dx', 1.0)
    dy = cpp_ms.get('dy', 1.0)

    # Python uses sigma=1.0 for eye model
    sigma = 1.0
    a_kde = -0.5 / (sigma * sigma)

    # Compute mean-shift with KDE weighting
    mx = 0.0
    my = 0.0
    total = 0.0

    print(f"  KDE computation (dx={dx}, dy={dy}, sigma={sigma}):")
    for i in range(window_size):
        for j in range(window_size):
            # Distance from current position (dx, dy)
            dist_sq = (i - dy)**2 + (j - dx)**2

            # Gaussian KDE weight
            kde_weight = np.exp(a_kde * dist_sq)

            # Combined weight
            v = py_response[i, j] * kde_weight
            total += v
            mx += v * j
            my += v * i

            if i == 1 and j == 1:  # center
                print(f"    center: resp={py_response[i,j]:.6f}, kde={kde_weight:.6f}, v={v:.6f}")

    py_msx = mx / total - dx
    py_msy = my / total - dy

    cpp_msx = cpp_ms.get('msx (mx/sum - dx)', 0)
    cpp_msy = cpp_ms.get('msy (my/sum - dy)', 0)

    print(f"  Python mean-shift: msx={py_msx:.6f}, msy={py_msy:.6f}")
    print(f"  C++ mean-shift:    msx={cpp_msx:.6f}, msy={cpp_msy:.6f}")
    print(f"  Difference: msx={abs(py_msx - cpp_msx):.6f}, msy={abs(py_msy - cpp_msy):.6f}")

    if abs(py_msx - cpp_msx) < 0.01 and abs(py_msy - cpp_msy) < 0.01:
        print("  ✓ MEAN-SHIFT MATCHES!")
    else:
        print("  ✗ MEAN-SHIFT MISMATCH")
        print(f"  Python: mx={mx:.6f}, my={my:.6f}, sum={total:.6f}")
        print(f"  C++:    mx={cpp_ms.get('mx (weighted x sum)', 0):.6f}, my={cpp_ms.get('my (weighted y sum)', 0):.6f}, sum={cpp_ms.get('sum (total weight)', 0):.6f}")

    print("\n" + "=" * 60)
    print("Verification Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
