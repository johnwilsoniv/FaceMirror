#!/usr/bin/env python3
"""
Analyze C++ eye convergence patterns in detail.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))

from analyze_convergence import load_cpp_trace

# Load C++ trace
cpp_trace = Path("/tmp/clnf_iteration_traces/cpp_trace.txt")

if not cpp_trace.exists():
    print(f"Error: C++ trace not found at {cpp_trace}")
    print("Run analyze_convergence.py first")
    sys.exit(1)

face_iters, eye_iters = load_cpp_trace(cpp_trace, include_eyes=True)

print("C++ Eye Convergence Analysis")
print("=" * 80)

# Separate by eye
left_eye = [i for i in eye_iters if i.get('eye_side') == 'left']
right_eye = [i for i in eye_iters if i.get('eye_side') == 'right']

print(f"Total eye iterations: {len(eye_iters)}")
print(f"  Left eye: {len(left_eye)}")
print(f"  Right eye: {len(right_eye)}")

# Analyze convergence by window size
def analyze_window_convergence(eye_data, eye_name):
    print(f"\n{eye_name} Eye Convergence by Window Size:")

    # Group by window size
    ws_groups = {}
    for iter_data in eye_data:
        ws = iter_data['window_size']
        if ws not in ws_groups:
            ws_groups[ws] = []
        ws_groups[ws].append(iter_data)

    for ws in sorted(ws_groups.keys()):
        iters = ws_groups[ws]
        print(f"  Window size {ws}: {len(iters)} iterations")

        # Check phases
        phases = [i['phase'] for i in iters]
        rigid_count = phases.count('rigid')
        nonrigid_count = phases.count('nonrigid')
        print(f"    Rigid: {rigid_count}, Nonrigid: {nonrigid_count}")

        # Check update magnitudes
        updates = [i.get('update_magnitude', 0) for i in iters]
        if updates:
            print(f"    Update magnitudes: min={min(updates):.6f}, max={max(updates):.6f}, mean={np.mean(updates):.6f}")

        # Check convergence
        if len(iters) > 1:
            first_update = iters[0].get('update_magnitude', 0)
            last_update = iters[-1].get('update_magnitude', 0)
            print(f"    Convergence: {first_update:.6f} -> {last_update:.6f} (reduction: {(first_update-last_update)/first_update*100:.1f}%)")

analyze_window_convergence(left_eye, "Left")
analyze_window_convergence(right_eye, "Right")

# Plot convergence curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left eye
ax1 = axes[0]
if left_eye:
    updates = [i.get('update_magnitude', 0) for i in left_eye]
    ax1.plot(updates, 'g-', linewidth=2)
    ax1.set_title('C++ Left Eye Convergence')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Update Magnitude')
    ax1.grid(True, alpha=0.3)

    # Mark window size transitions
    prev_ws = None
    for i, iter_data in enumerate(left_eye):
        if iter_data['window_size'] != prev_ws and prev_ws is not None:
            ax1.axvline(x=i, color='orange', linestyle='--', alpha=0.5)
            ax1.text(i, ax1.get_ylim()[1]*0.9, f"ws={iter_data['window_size']}", rotation=90)
        prev_ws = iter_data['window_size']

# Right eye
ax2 = axes[1]
if right_eye:
    updates = [i.get('update_magnitude', 0) for i in right_eye]
    ax2.plot(updates, 'm-', linewidth=2)
    ax2.set_title('C++ Right Eye Convergence')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Update Magnitude')
    ax2.grid(True, alpha=0.3)

    # Mark window size transitions
    prev_ws = None
    for i, iter_data in enumerate(right_eye):
        if iter_data['window_size'] != prev_ws and prev_ws is not None:
            ax2.axvline(x=i, color='orange', linestyle='--', alpha=0.5)
            ax2.text(i, ax2.get_ylim()[1]*0.9, f"ws={iter_data['window_size']}", rotation=90)
        prev_ws = iter_data['window_size']

plt.suptitle('C++ Eye Model Convergence Analysis')
plt.tight_layout()
plt.savefig('/Users/johnwilsoniv/Documents/SplitFace Open3/convergence_analysis/cpp_eye_convergence.png', dpi=150)
print(f"\nPlot saved to convergence_analysis/cpp_eye_convergence.png")
plt.show()

# Analyze parameter changes
print("\n" + "=" * 80)
print("Parameter Evolution Analysis:")

def analyze_param_changes(eye_data, eye_name):
    print(f"\n{eye_name} Eye Parameter Changes:")

    if not eye_data:
        return

    # Extract parameters over iterations
    params_list = []
    for iter_data in eye_data:
        if 'params' in iter_data:
            global_params = iter_data['params'].get('global', [])
            local_params = iter_data['params'].get('local', [])
            all_params = global_params + local_params
            params_list.append(all_params)

    if not params_list:
        return

    params_array = np.array(params_list)

    # Analyze global parameters (first 6)
    if params_array.shape[1] >= 6:
        print(f"  Global parameters (scale, rot, trans):")
        for i, name in enumerate(['scale', 'rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y']):
            if i < params_array.shape[1]:
                values = params_array[:, i]
                change = values[-1] - values[0]
                print(f"    {name}: initial={values[0]:.4f}, final={values[-1]:.4f}, change={change:.4f}")

    # Analyze local parameters
    if params_array.shape[1] > 6:
        n_local = params_array.shape[1] - 6
        print(f"  Local parameters ({n_local} shape modes):")
        for i in range(min(4, n_local)):  # Show first 4 shape params
            values = params_array[:, 6 + i]
            change = values[-1] - values[0]
            print(f"    mode_{i}: initial={values[0]:.4f}, final={values[-1]:.4f}, change={change:.4f}")

analyze_param_changes(left_eye, "Left")
analyze_param_changes(right_eye, "Right")