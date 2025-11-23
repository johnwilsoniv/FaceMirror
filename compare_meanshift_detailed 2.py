#!/usr/bin/env python3
"""
Compare mean-shift computation between C++ and Python in detail.

Outputs the same data format as C++ for direct comparison:
- Similarity transforms
- Per-landmark offsets, dx/dy, mean-shifts (before and after transform)
"""

import numpy as np
import cv2
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))

from pyclnf import CLNF
from pyclnf.core.pdm import PDM
from pyclnf.core.utils import align_shapes_with_scale
from pyclnf.core.optimizer import _kde_mean_shift_numba

def main():
    print("=" * 80)
    print("DETAILED MEAN-SHIFT COMPARISON")
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

    # Save frame for C++
    frame_path = "/tmp/debug_frame.png"
    cv2.imwrite(frame_path, frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run C++ first to get detailed output
    print("\n1. Running C++ OpenFace to generate detailed output...")
    bin_path = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
    cmd = [
        bin_path,
        "-f", str(frame_path),
        "-out_dir", "/tmp/cpp_debug",
        "-2Dfp",
        "-pose"
    ]
    subprocess.run(cmd, capture_output=True)

    # Read C++ output
    cpp_file = Path("/tmp/cpp_meanshift_detailed.txt")
    if cpp_file.exists():
        print(f"\nC++ output saved to: {cpp_file}")
        with open(cpp_file) as f:
            cpp_content = f.read()
        # Parse C++ similarity transforms
        for line in cpp_content.split('\n'):
            if 'sim_img_to_ref' in line or 'sim_ref_to_img' in line:
                print(f"  {line}")
    else:
        print("Warning: C++ detailed output not found")

    # Now run Python with detailed output
    print("\n2. Running Python pyclnf with detailed output...")

    MODEL_DIR = Path("pyclnf/models")
    pdm = PDM(str(MODEL_DIR / "exported_pdm"))

    # Get the same bbox that C++ used
    bbox_file = Path("/tmp/cpp_init_bbox.txt")
    bbox = None
    if bbox_file.exists():
        with open(bbox_file, 'r') as f:
            for line in f:
                if line.startswith('bbox:'):
                    parts = line.split()[1:]
                    bbox = np.array([float(p) for p in parts])
                    print(f"  Using C++ bbox: {bbox}")
                    break

    if bbox is None:
        print("Error: No bbox found")
        return

    # Initialize params from bbox
    params = pdm.init_params(bbox)
    landmarks_2d = pdm.params_to_landmarks_2d(params)

    print(f"  Initial params: scale={params[0]:.6f}, tx={params[4]:.2f}, ty={params[5]:.2f}")
    print(f"  Landmark 36: ({landmarks_2d[36, 0]:.4f}, {landmarks_2d[36, 1]:.4f})")

    # Compute similarity transform (image to reference)
    # Mean shape is stored as (204, 1) = [x0..x67, y0..y67, z0..z67]
    mean_shape = pdm.mean_shape.flatten()
    n = len(mean_shape) // 3  # 68
    ref_shape = np.column_stack([mean_shape[:n], mean_shape[n:2*n]])

    print(f"  ref_shape.shape: {ref_shape.shape}, landmarks_2d.shape: {landmarks_2d.shape}")

    sim_img_to_ref = align_shapes_with_scale(landmarks_2d, ref_shape)

    # Extract transform components
    a = sim_img_to_ref[0, 0]
    b = sim_img_to_ref[1, 0]
    det = a*a + b*b

    # Compute inverse
    sim_ref_to_img = np.array([
        [a/det, b/det],
        [-b/det, a/det]
    ], dtype=np.float32)

    # Save Python detailed output
    py_file = Path("/tmp/python_meanshift_detailed.txt")
    with open(py_file, 'w') as f:
        f.write("=== DETAILED MEAN-SHIFT COMPARISON DATA ===\n\n")

        # Similarity transforms
        f.write("sim_img_to_ref:\n")
        f.write(f"  [{sim_img_to_ref[0, 0]:.8f}, {sim_img_to_ref[0, 1]:.8f}]\n")
        f.write(f"  [{sim_img_to_ref[1, 0]:.8f}, {sim_img_to_ref[1, 1]:.8f}]\n")

        f.write("\nsim_ref_to_img:\n")
        f.write(f"  [{sim_ref_to_img[0, 0]:.8f}, {sim_ref_to_img[0, 1]:.8f}]\n")
        f.write(f"  [{sim_ref_to_img[1, 0]:.8f}, {sim_ref_to_img[1, 1]:.8f}]\n")

        # Gaussian param
        sigma = 1.5
        a_kde = -0.5 / (sigma * sigma)
        f.write(f"\na (Gaussian param): {a_kde:.8f}\n")
        f.write("resp_size: 11\n")

    print(f"\nPython output saved to: {py_file}")
    print(f"  sim_img_to_ref[0,0] (a): {sim_img_to_ref[0, 0]:.8f}")
    print(f"  sim_img_to_ref[1,0] (b): {sim_img_to_ref[1, 0]:.8f}")

    # Compare the transforms
    print("\n" + "=" * 80)
    print("COMPARISON: Similarity Transforms")
    print("=" * 80)

    if cpp_file.exists():
        # Parse C++ values
        cpp_lines = cpp_content.split('\n')
        cpp_sim_img = []
        cpp_sim_ref = []
        in_img = False
        in_ref = False

        for line in cpp_lines:
            if 'sim_img_to_ref:' in line:
                in_img = True
                in_ref = False
            elif 'sim_ref_to_img:' in line:
                in_img = False
                in_ref = True
            elif line.strip().startswith('['):
                vals = [float(x) for x in line.strip().strip('[]').split(',')]
                if in_img:
                    cpp_sim_img.append(vals)
                elif in_ref:
                    cpp_sim_ref.append(vals)

        if cpp_sim_img:
            print("\nsim_img_to_ref comparison:")
            print(f"  C++:    [{cpp_sim_img[0][0]:.8f}, {cpp_sim_img[0][1]:.8f}]")
            print(f"          [{cpp_sim_img[1][0]:.8f}, {cpp_sim_img[1][1]:.8f}]")
            print(f"  Python: [{sim_img_to_ref[0, 0]:.8f}, {sim_img_to_ref[0, 1]:.8f}]")
            print(f"          [{sim_img_to_ref[1, 0]:.8f}, {sim_img_to_ref[1, 1]:.8f}]")

            # Check differences
            diff_a = abs(cpp_sim_img[0][0] - sim_img_to_ref[0, 0])
            diff_b = abs(cpp_sim_img[1][0] - sim_img_to_ref[1, 0])
            print(f"\n  Difference in 'a': {diff_a:.8f}")
            print(f"  Difference in 'b': {diff_b:.8f}")

            if diff_a > 0.0001 or diff_b > 0.0001:
                print("\n  ⚠️  SIGNIFICANT DIFFERENCE in similarity transforms!")
            else:
                print("\n  ✓ Transforms match!")

    # Now parse and compare per-landmark data
    print("\n" + "=" * 80)
    print("COMPARISON: Per-Landmark Mean-Shifts")
    print("=" * 80)

    # Parse C++ per-landmark data
    cpp_before = {}
    cpp_after = {}
    if cpp_file.exists():
        in_before = False
        in_after = False
        for line in cpp_lines:
            if 'BEFORE transform' in line:
                in_before = True
                in_after = False
            elif 'AFTER transform' in line:
                in_before = False
                in_after = True
            elif line.strip() and line[0].isdigit():
                parts = [x.strip() for x in line.split(',')]
                if len(parts) >= 4:
                    idx = int(parts[0])
                    if in_before and len(parts) >= 7:
                        cpp_before[idx] = {
                            'offset_x': float(parts[1]),
                            'offset_y': float(parts[2]),
                            'dx': float(parts[3]),
                            'dy': float(parts[4]),
                            'ms_ref_x': float(parts[5]),
                            'ms_ref_y': float(parts[6])
                        }
                    elif in_after:
                        cpp_after[idx] = {
                            'ms_x': float(parts[1]),
                            'ms_y': float(parts[2]),
                            'mag': float(parts[3])
                        }

    # Compare for key landmarks
    key_landmarks = [36, 48, 30, 8]

    print("\nKey landmarks (AFTER transform to image coords):")
    print(f"{'Idx':<5} {'C++ ms_x':>12} {'C++ ms_y':>12} {'Py ms_x':>12} {'Py ms_y':>12}")
    print("-" * 55)

    for lm_idx in key_landmarks:
        if lm_idx in cpp_after:
            cpp = cpp_after[lm_idx]
            # Note: We don't have Python mean-shifts yet since we need to run the full optimizer
            # This script shows the comparison framework
            print(f"{lm_idx:<5} {cpp['ms_x']:>12.4f} {cpp['ms_y']:>12.4f} {'--':>12} {'--':>12}")

    print("\nNote: Run full CLNF.fit() to get Python mean-shifts for comparison.")
    print("The key is to check if similarity transforms match first.")

if __name__ == "__main__":
    main()
