#!/usr/bin/env python3
"""
Diagnose why Python iteration updates are less effective than C++.

Compares:
1. Mean-shift magnitudes
2. Update magnitudes
3. Parameter changes
4. Damping factors
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))

from pyclnf.core.pdm import PDM

# Load C++ trace
def load_cpp_trace(trace_file):
    """Load C++ iteration trace."""
    iterations = []

    with open(trace_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()

            if len(parts) < 45:
                continue

            local_params = [float(p) for p in parts[11:11+34]]
            if len(local_params) != 34:
                continue

            iteration_data = {
                'iteration': int(parts[0]),
                'phase': parts[1],
                'window_size': int(parts[2]),
                'mean_shift_norm': float(parts[3]),
                'update_magnitude': float(parts[4]),
                'params': np.array([
                    float(parts[5]),   # scale
                    float(parts[6]),   # rot_x
                    float(parts[7]),   # rot_y
                    float(parts[8]),   # rot_z
                    float(parts[9]),   # tx
                    float(parts[10]),  # ty
                ] + local_params)
            }
            iterations.append(iteration_data)

    return iterations

def main():
    print("=" * 100)
    print("DIAGNOSTIC: Python vs C++ Update Effectiveness")
    print("=" * 100)

    # Load PDM
    pdm = PDM(str(Path("pyclnf/models/exported_pdm")))

    # Load C++ trace
    cpp_trace = load_cpp_trace("/tmp/clnf_iteration_traces/cpp_trace.txt")

    if not cpp_trace:
        print("Error: No C++ trace found")
        return

    # Get initial params (from bbox file)
    bbox_file = Path("/tmp/cpp_init_bbox.txt")
    init_params = None
    if bbox_file.exists():
        with open(bbox_file, 'r') as f:
            for line in f:
                if line.startswith('params:'):
                    parts = line.split()[1:]
                    init_params = np.zeros(40)
                    init_params[0] = float(parts[0])  # scale
                    init_params[1] = float(parts[3])  # rot_x (was 0)
                    init_params[2] = float(parts[4])  # rot_y (was 0)
                    init_params[3] = float(parts[5])  # rot_z (was 0)
                    init_params[4] = float(parts[1])  # tx
                    init_params[5] = float(parts[2])  # ty
                    break

    if init_params is None:
        print("Error: Could not load initial params")
        return

    print(f"\nInitial params (from bbox):")
    print(f"  scale = {init_params[0]:.6f}")
    print(f"  tx = {init_params[4]:.6f}, ty = {init_params[5]:.6f}")
    print(f"  rot = ({init_params[1]:.6f}, {init_params[2]:.6f}, {init_params[3]:.6f})")

    # Analyze first few iterations
    print("\n" + "=" * 100)
    print("C++ ITERATION ANALYSIS (first 10 iterations)")
    print("=" * 100)

    print(f"\n{'Iter':<5} {'Phase':<10} {'WS':<4} {'MS Norm':<12} {'Update Mag':<12} {'Δscale':<10} {'Δtx':<10} {'Δty':<10}")
    print("-" * 85)

    prev_params = init_params.copy()
    for i, iter_data in enumerate(cpp_trace[:10]):
        curr_params = iter_data['params']

        # Compute parameter changes
        delta_scale = curr_params[0] - prev_params[0]
        delta_tx = curr_params[4] - prev_params[4]
        delta_ty = curr_params[5] - prev_params[5]

        print(f"{iter_data['iteration']:<5} {iter_data['phase']:<10} {iter_data['window_size']:<4} "
              f"{iter_data['mean_shift_norm']:<12.2f} {iter_data['update_magnitude']:<12.2f} "
              f"{delta_scale:<10.4f} {delta_tx:<10.2f} {delta_ty:<10.2f}")

        prev_params = curr_params.copy()

    # Compute effective damping
    print("\n" + "=" * 100)
    print("DAMPING ANALYSIS")
    print("=" * 100)

    # For C++ iteration 0:
    # mean_shift_norm = 213.882
    # update_magnitude = 23.108
    # Effective damping ≈ update_magnitude / mean_shift_norm

    iter0 = cpp_trace[0]
    cpp_damping = iter0['update_magnitude'] / iter0['mean_shift_norm'] if iter0['mean_shift_norm'] > 0 else 0

    print(f"\nC++ Iteration 0:")
    print(f"  Mean-shift norm: {iter0['mean_shift_norm']:.2f}")
    print(f"  Update magnitude: {iter0['update_magnitude']:.2f}")
    print(f"  Ratio (update/mean_shift): {cpp_damping:.4f}")

    # The actual damping factor in C++ is 0.75 (applied to delta_p)
    # But the relationship between mean_shift and delta_p depends on the Jacobian

    print("\n" + "=" * 100)
    print("KEY OBSERVATIONS")
    print("=" * 100)

    # Compute total parameter change in first iteration
    iter0_params = cpp_trace[0]['params']
    total_param_change = np.linalg.norm(iter0_params - init_params)

    print(f"\nC++ first iteration total param change: {total_param_change:.4f}")
    print(f"C++ first iteration tx change: {iter0_params[4] - init_params[4]:.2f} pixels")
    print(f"C++ first iteration ty change: {iter0_params[5] - init_params[5]:.2f} pixels")

    # Check rotation changes
    rot_change = np.linalg.norm(iter0_params[1:4] - init_params[1:4])
    print(f"C++ first iteration rotation change: {rot_change:.4f} radians")

    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)

    print("""
To match C++ update effectiveness, check:

1. DAMPING FACTOR
   - C++ uses 0.75 damping (PDM.cpp line 677)
   - Python currently uses 0.5 damping
   - Try increasing to 0.75

2. MEAN-SHIFT MAGNITUDE
   - C++ iter 0 mean_shift_norm = {:.2f}
   - If Python's is much smaller, check response map quality

3. JACOBIAN COMPUTATION
   - Verify analytical Jacobian matches numerical
   - Check parameter order consistency

4. REGULARIZATION
   - C++ might use different regularization per phase
   - Check if regularization is too strong in Python
""".format(iter0['mean_shift_norm']))

    # Now let's check what Python produces
    print("\n" + "=" * 100)
    print("TESTING PYTHON OPTIMIZATION")
    print("=" * 100)

    import cv2
    from pyclnf import CLNF

    # Load test image
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 160)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame")
        return

    # Get bbox
    bbox = np.array([305.65, 695.87, 448.2, 476.76])

    # Initialize CLNF with debug mode to see mean-shift values
    MODEL_DIR = Path("pyclnf/models")
    clnf = CLNF(
        model_dir=str(MODEL_DIR),
        detector=None,
        max_iterations=40,
        convergence_threshold=0.5,
        regularization=20,
        window_sizes=[11, 9, 7, 5],
        debug_mode=True  # Enable debug output
    )

    print("\nRunning Python CLNF with debug mode...")
    print("(Looking for mean-shift and update values)\n")

    # Run optimization
    landmarks, info = clnf.fit(frame, bbox, return_params=True)

    print(f"\nPython completed {info['iterations']} iterations")
    print(f"Final update magnitude: {info['final_update']:.6f}")

    # Compare iteration histories if available
    if 'iteration_history' in info and len(info['iteration_history']) > 0:
        print("\n" + "=" * 100)
        print("PYTHON ITERATION ANALYSIS")
        print("=" * 100)

        print(f"\n{'Iter':<5} {'Phase':<10} {'WS':<4} {'MS Norm':<12} {'Update Mag':<12}")
        print("-" * 50)

        for i, iter_data in enumerate(info['iteration_history'][:10]):
            print(f"{i:<5} {iter_data['phase']:<10} {iter_data['window_size']:<4} "
                  f"{iter_data['mean_shift_norm']:<12.2f} {iter_data['update_magnitude']:<12.2f}")

        # Compare first iteration
        py_iter0 = info['iteration_history'][0]
        cpp_iter0 = cpp_trace[0]

        print("\n" + "=" * 100)
        print("FIRST ITERATION COMPARISON")
        print("=" * 100)

        print(f"\n{'Metric':<25} {'C++':<15} {'Python':<15} {'Ratio':<10}")
        print("-" * 65)

        ms_ratio = py_iter0['mean_shift_norm'] / cpp_iter0['mean_shift_norm'] if cpp_iter0['mean_shift_norm'] > 0 else 0
        up_ratio = py_iter0['update_magnitude'] / cpp_iter0['update_magnitude'] if cpp_iter0['update_magnitude'] > 0 else 0

        print(f"{'Mean-shift norm':<25} {cpp_iter0['mean_shift_norm']:<15.2f} {py_iter0['mean_shift_norm']:<15.2f} {ms_ratio:<10.2f}")
        print(f"{'Update magnitude':<25} {cpp_iter0['update_magnitude']:<15.2f} {py_iter0['update_magnitude']:<15.2f} {up_ratio:<10.2f}")

        if ms_ratio < 0.5:
            print("\n⚠️  Python mean-shift is significantly smaller than C++!")
            print("    This suggests response maps or KDE computation differs.")

        if up_ratio < 0.5:
            print("\n⚠️  Python update magnitude is significantly smaller than C++!")
            print("    This suggests damping factor or Jacobian computation differs.")

if __name__ == "__main__":
    main()
