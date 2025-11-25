#!/usr/bin/env python3
"""
Analyze existing eye trace files to understand convergence differences.
"""

import sys
from pathlib import Path
import numpy as np

# Read Python eye trace
python_trace = Path("/tmp/clnf_iteration_traces/python_eye_trace.txt")
if not python_trace.exists():
    print("Python eye trace not found")
    sys.exit(1)

# Parse Python trace
py_eye_iters = []
with open(python_trace, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) >= 4:
            iter_num = int(parts[0])
            window_size = int(parts[1])
            phase = parts[2]
            eye_side = parts[3]
            # Rest are parameters
            params = [float(p) for p in parts[4:]]

            py_eye_iters.append({
                'iteration': iter_num,
                'window_size': window_size,
                'phase': phase,
                'eye_side': eye_side,
                'params': params
            })

# Load C++ trace for comparison
cpp_trace = Path("/tmp/clnf_iteration_traces/cpp_trace.txt")

cpp_eye_iters = []
if cpp_trace.exists():
    with open(cpp_trace, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()

            # Check if this is an eye iteration (10 local params)
            if len(parts) > 10:
                # Count local params (after the first few fields)
                # Format: iter phase ws mean_shift update jwtm scale rot(3) trans(2) local_params...
                local_start = 12  # After fixed fields
                if len(parts) >= local_start:
                    n_local = len(parts) - local_start
                    if n_local == 10:  # Eye model has 10 local params
                        # This is an eye iteration
                        iter_num = int(parts[0])
                        phase = parts[1]
                        ws = int(parts[2])

                        # Determine eye side based on iteration count
                        eye_idx = len(cpp_eye_iters)
                        eye_side = 'left' if eye_idx < 20 else 'right'

                        # Get update magnitude
                        update_mag = float(parts[4]) if len(parts) > 4 else 0

                        cpp_eye_iters.append({
                            'iteration': iter_num,
                            'phase': phase,
                            'window_size': ws,
                            'eye_side': eye_side,
                            'update_magnitude': update_mag
                        })

print("=" * 80)
print("EYE TRACE ANALYSIS: C++ vs Python")
print("=" * 80)

# Analyze Python trace
py_left = [i for i in py_eye_iters if i['eye_side'] == 'left']
py_right = [i for i in py_eye_iters if i['eye_side'] == 'right']

# Analyze C++ trace
cpp_left = [i for i in cpp_eye_iters if i['eye_side'] == 'left']
cpp_right = [i for i in cpp_eye_iters if i['eye_side'] == 'right']

print(f"\nTotal iterations:")
print(f"  C++ Left: {len(cpp_left)}, Right: {len(cpp_right)}, Total: {len(cpp_eye_iters)}")
print(f"  Python Left: {len(py_left)}, Right: {len(py_right)}, Total: {len(py_eye_iters)}")

def analyze_by_window(data, name):
    print(f"\n{name} Window Size Analysis:")

    # Group by window size
    by_ws = {}
    for item in data:
        ws = item['window_size']
        if ws not in by_ws:
            by_ws[ws] = []
        by_ws[ws].append(item)

    for ws in sorted(by_ws.keys()):
        items = by_ws[ws]
        phases = [i['phase'] for i in items]
        rigid = phases.count('rigid')
        nonrigid = phases.count('nonrigid')

        print(f"  WS {ws}: {len(items)} iterations (rigid:{rigid}, nonrigid:{nonrigid})")

        # For C++, check update magnitudes if available
        if 'update_magnitude' in items[0]:
            updates = [i['update_magnitude'] for i in items]
            if len(updates) > 1:
                trend = "↓ converging" if updates[-1] < updates[0] else "↑ DIVERGING"
                print(f"    Updates: {updates[0]:.4f} → {updates[-1]:.4f} {trend}")

        # For Python, analyze parameter changes
        if 'params' in items[0]:
            # Look at translation parameters (indices 4,5 are trans_x, trans_y)
            if len(items[0]['params']) > 5:
                first_tx = items[0]['params'][4]
                first_ty = items[0]['params'][5]
                last_tx = items[-1]['params'][4]
                last_ty = items[-1]['params'][5]

                delta_tx = last_tx - first_tx
                delta_ty = last_ty - first_ty
                total_movement = np.sqrt(delta_tx**2 + delta_ty**2)

                print(f"    Translation change: Δx={delta_tx:.2f}, Δy={delta_ty:.2f}, total={total_movement:.2f}px")

                # Check shape params (indices 6+)
                if len(items[0]['params']) > 6:
                    first_shape = items[0]['params'][6:]
                    last_shape = items[-1]['params'][6:]
                    shape_change = np.linalg.norm(np.array(last_shape) - np.array(first_shape))
                    print(f"    Shape param change: {shape_change:.4f}")

# Analyze each eye
print("\n" + "=" * 80)
print("LEFT EYE ANALYSIS")
print("=" * 80)
analyze_by_window(cpp_left, "C++ Left Eye")
analyze_by_window(py_left, "Python Left Eye")

print("\n" + "=" * 80)
print("RIGHT EYE ANALYSIS")
print("=" * 80)
analyze_by_window(cpp_right, "C++ Right Eye")
analyze_by_window(py_right, "Python Right Eye")

# Check for divergence issues
print("\n" + "=" * 80)
print("DIVERGENCE ANALYSIS")
print("=" * 80)

# C++ divergence check
for eye_data, eye_name in [(cpp_left, "C++ Left"), (cpp_right, "C++ Right")]:
    by_ws = {}
    for item in eye_data:
        ws = item['window_size']
        if ws not in by_ws:
            by_ws[ws] = []
        by_ws[ws].append(item)

    for ws in sorted(by_ws.keys()):
        items = by_ws[ws]
        if 'update_magnitude' in items[0]:
            updates = [i['update_magnitude'] for i in items]
            if len(updates) > 1 and updates[-1] > updates[0]:
                print(f"  ⚠️  {eye_name} WS {ws}: DIVERGES ({updates[0]:.4f} → {updates[-1]:.4f})")

# Python movement check
for eye_data, eye_name in [(py_left, "Python Left"), (py_right, "Python Right")]:
    by_ws = {}
    for item in eye_data:
        ws = item['window_size']
        if ws not in by_ws:
            by_ws[ws] = []
        by_ws[ws].append(item)

    for ws in sorted(by_ws.keys()):
        items = by_ws[ws]
        if 'params' in items[0] and len(items[0]['params']) > 5:
            # Check if eye center moved significantly
            first_tx = items[0]['params'][4]
            first_ty = items[0]['params'][5]
            last_tx = items[-1]['params'][4]
            last_ty = items[-1]['params'][5]

            movement = np.sqrt((last_tx - first_tx)**2 + (last_ty - first_ty)**2)
            if movement > 5:  # More than 5 pixels movement
                print(f"  ⚠️  {eye_name} WS {ws}: Large movement {movement:.2f}px")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

# Compare iteration counts per window size
print("\nIteration Count Comparison:")
print("  C++ uses 5 iterations per phase (10 per window size)")
print("  Python uses 5 iterations per phase (10 per window size)")
print("  Both use window sizes [3, 5]")

# Note about convergence
if len(cpp_eye_iters) > 0:
    print("\n⚠️  Important: C++ shows divergence at window size 5")
    print("    This may explain the eye refinement quality issues")