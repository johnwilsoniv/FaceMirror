#!/usr/bin/env python3
"""
Debug Python KDE mean-shift computation for landmark 36.
Output same format as C++ for direct comparison.
"""

import numpy as np
import cv2
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))

from pyclnf.core.pdm import PDM
from pyclnf.core.patch_expert import CCNFPatchExpert
from pyclnf.core.utils import align_shapes_with_scale

def main():
    print("=" * 80)
    print("PYTHON KDE MEAN-SHIFT DEBUG - Landmark 36")
    print("=" * 80)

    # Setup
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    frame_idx = 160

    # Extract frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame")
        return

    frame_path = "/tmp/debug_frame.png"
    cv2.imwrite(frame_path, frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run C++ first
    print("\n1. Running C++ to get debug output...")
    bin_path = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
    subprocess.run([bin_path, "-f", str(frame_path), "-out_dir", "/tmp/cpp_debug", "-2Dfp"], capture_output=True)

    # Print C++ KDE values
    cpp_kde = Path("/tmp/cpp_kde_lm36.txt")
    if cpp_kde.exists():
        print("\nC++ KDE values:")
        with open(cpp_kde) as f:
            print(f.read())

    # Load Python model
    MODEL_DIR = Path("pyclnf/models")
    pdm = PDM(str(MODEL_DIR / "exported_pdm"))

    # Get bbox
    bbox_file = Path("/tmp/cpp_init_bbox.txt")
    with open(bbox_file, 'r') as f:
        for line in f:
            if line.startswith('bbox:'):
                parts = line.split()[1:]
                bbox = np.array([float(p) for p in parts])
                break

    # Initialize params
    params = pdm.init_params(bbox)
    landmarks_2d = pdm.params_to_landmarks_2d(params)

    print("\n2. Python initialization...")
    print(f"  Bbox: {bbox}")
    print(f"  Landmark 36: ({landmarks_2d[36, 0]:.4f}, {landmarks_2d[36, 1]:.4f})")

    # Use CLNF to get response map
    from pyclnf import CLNF

    clnf = CLNF(
        model_dir=str(MODEL_DIR),
        detector=None,
        max_iterations=1,  # Just one iteration to capture debug data
        regularization=25,
        window_sizes=[11],  # Only first window
        debug_mode=True
    )

    # Run one iteration to get response map for landmark 36
    # The debug mode should print KDE values
    print("\n3. Running CLNF to get response map...")
    landmarks, info = clnf.fit(gray, bbox, return_params=True)

    # The response map is computed inside the optimizer
    # Let's manually compute the KDE to compare
    # Get the internal response maps from the optimizer history
    if 'iteration_history' in info and len(info['iteration_history']) > 0:
        print(f"  Iterations run: {len(info['iteration_history'])}")

    # Since we can't easily extract the response map, let's compute it manually
    # using the patch expert from CLNF
    window_size = 11
    scale_idx = 0  # 0.25 scale
    view_idx = 0

    # Get patch expert for landmark 36
    patch_experts = clnf._get_patch_experts(view_idx, 0.25)

    if 36 not in patch_experts:
        print("  Error: No patch expert for landmark 36")
        return

    response = patch_experts[36].compute_response(gray, landmarks_2d[36], window_size)

    print(f"\n3. Response map for landmark 36:")
    print(f"  Shape: {response.shape}")
    print(f"  Min: {response.min():.6f}")
    print(f"  Max: {response.max():.6f}")
    print(f"  Mean: {response.mean():.6f}")

    # Compute mean-shift using KDE
    # For first iteration, offset is 0 so dx=dy=center
    resp_size = response.shape[0]
    center = (resp_size - 1) / 2.0
    dx = center
    dy = center

    # KDE parameters
    sigma = 1.5
    a = -0.5 / (sigma * sigma)

    # Compute weighted mean
    mx = 0.0
    my = 0.0
    total_weight = 0.0

    print(f"\n4. KDE computation:")
    print(f"  dx: {dx:.8f}")
    print(f"  dy: {dy:.8f}")
    print(f"  a (Gaussian param): {a:.8f}")

    for ii in range(resp_size):
        for jj in range(resp_size):
            resp_val = response[ii, jj]

            # KDE weight
            dist_sq = (dy - ii)**2 + (dx - jj)**2
            kde_weight = np.exp(a * dist_sq)

            weight = resp_val * kde_weight
            total_weight += weight
            mx += weight * jj
            my += weight * ii

    # Compute mean-shift
    if total_weight > 1e-10:
        ms_x = (mx / total_weight) - dx
        ms_y = (my / total_weight) - dy
    else:
        ms_x = 0.0
        ms_y = 0.0

    print(f"\n5. Accumulation results:")
    print(f"  mx (weighted x sum): {mx:.8f}")
    print(f"  my (weighted y sum): {my:.8f}")
    print(f"  sum (total weight): {total_weight:.8f}")
    print(f"  mx/sum: {mx/total_weight:.8f}")
    print(f"  my/sum: {my/total_weight:.8f}")
    print(f"  msx (mx/sum - dx): {ms_x:.8f}")
    print(f"  msy (my/sum - dy): {ms_y:.8f}")

    # Save to file for comparison
    with open("/tmp/python_kde_lm36.txt", 'w') as f:
        f.write("Landmark 36 KDE Mean-Shift Computation\n")
        f.write(f"dx: {dx:.8f}\n")
        f.write(f"dy: {dy:.8f}\n")
        f.write(f"mx (weighted x sum): {mx:.8f}\n")
        f.write(f"my (weighted y sum): {my:.8f}\n")
        f.write(f"sum (total weight): {total_weight:.8f}\n")
        f.write(f"mx/sum: {mx/total_weight:.8f}\n")
        f.write(f"my/sum: {my/total_weight:.8f}\n")
        f.write(f"msx (mx/sum - dx): {ms_x:.8f}\n")
        f.write(f"msy (my/sum - dy): {ms_y:.8f}\n")

    # Compare with C++
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"{'Value':<25} {'C++':<15} {'Python':<15} {'Diff':<15}")
    print("-" * 70)

    # Parse C++ values
    cpp_vals = {}
    if cpp_kde.exists():
        with open(cpp_kde) as f:
            for line in f:
                if ':' in line:
                    parts = line.split(':')
                    key = parts[0].strip()
                    val = float(parts[1].strip())
                    cpp_vals[key] = val

    py_vals = {
        'dx': dx,
        'dy': dy,
        'mx (weighted x sum)': mx,
        'my (weighted y sum)': my,
        'sum (total weight)': total_weight,
        'mx/sum': mx/total_weight,
        'my/sum': my/total_weight,
        'msx (mx/sum - dx)': ms_x,
        'msy (my/sum - dy)': ms_y
    }

    for key in py_vals:
        cpp_val = cpp_vals.get(key, 0)
        py_val = py_vals[key]
        diff = py_val - cpp_val
        print(f"{key:<25} {cpp_val:<15.6f} {py_val:<15.6f} {diff:<15.6f}")

    # Check for significant differences
    if abs(py_vals['msx (mx/sum - dx)'] - cpp_vals.get('msx (mx/sum - dx)', 0)) > 0.1:
        print("\n⚠️  SIGNIFICANT difference in mean-shift X!")
    if abs(py_vals['msy (my/sum - dy)'] - cpp_vals.get('msy (my/sum - dy)', 0)) > 0.1:
        print("\n⚠️  SIGNIFICANT difference in mean-shift Y!")

if __name__ == "__main__":
    main()
