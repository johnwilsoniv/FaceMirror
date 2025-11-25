#!/usr/bin/env python3
"""
Compare C++ and Python eye convergence patterns.
"""

import sys
from pathlib import Path
import numpy as np
import json
sys.path.insert(0, str(Path(__file__).parent))

from analyze_convergence import load_cpp_trace, run_python_on_frame, analyze_video
import cv2
from pyclnf import CLNF

# Load test frame
test_frame_path = Path("/tmp/clnf_iteration_traces/frame_0.jpg")
if not test_frame_path.exists():
    print("Error: Test frame not found. Run analyze_convergence.py first")
    sys.exit(1)

image = cv2.imread(str(test_frame_path))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get bbox from C++ trace (use same bbox for fair comparison)
cpp_trace = Path("/tmp/clnf_iteration_traces/cpp_trace.txt")
if cpp_trace.exists():
    face_iters, cpp_eye_iters = load_cpp_trace(cpp_trace, include_eyes=True)

    # Use standard bbox for testing
    h, w = gray.shape
    bbox = np.array([297.13986403, 698.72593072, 482.76186432, 471.92386217])  # From C++ output
else:
    print("Error: C++ trace not found")
    sys.exit(1)

# Run Python CLNF with iteration tracking
print("Running Python CLNF with eye tracking...")
clnf = CLNF(use_eye_refinement=True)
landmarks, face_iters, py_eye_iters = run_python_on_frame(gray, bbox, clnf)

print("\n" + "=" * 80)
print("EYE CONVERGENCE COMPARISON: C++ vs Python")
print("=" * 80)

# Separate by eye
cpp_left = [i for i in cpp_eye_iters if i.get('eye_side') == 'left']
cpp_right = [i for i in cpp_eye_iters if i.get('eye_side') == 'right']

py_left = [i for i in py_eye_iters if i.get('eye_side') == 'left']
py_right = [i for i in py_eye_iters if i.get('eye_side') == 'right']

def compare_convergence(cpp_data, py_data, eye_name):
    print(f"\n{eye_name} Eye Comparison:")
    print(f"  C++ iterations: {len(cpp_data)}")
    print(f"  Python iterations: {len(py_data)}")

    # Group by window size
    cpp_ws = {}
    py_ws = {}

    for i in cpp_data:
        ws = i['window_size']
        if ws not in cpp_ws:
            cpp_ws[ws] = []
        cpp_ws[ws].append(i)

    for i in py_data:
        ws = i['window_size']
        if ws not in py_ws:
            py_ws[ws] = []
        py_ws[ws].append(i)

    # Compare each window size
    all_ws = sorted(set(list(cpp_ws.keys()) + list(py_ws.keys())))

    for ws in all_ws:
        print(f"\n  Window Size {ws}:")

        cpp_iters = cpp_ws.get(ws, [])
        py_iters = py_ws.get(ws, [])

        print(f"    C++ iterations: {len(cpp_iters)}")
        print(f"    Python iterations: {len(py_iters)}")

        if cpp_iters:
            cpp_updates = [i.get('update_magnitude', 0) for i in cpp_iters]
            print(f"    C++ updates: first={cpp_updates[0]:.6f}, last={cpp_updates[-1]:.6f}")
            print(f"                 mean={np.mean(cpp_updates):.6f}, convergence={(cpp_updates[0]-cpp_updates[-1])/cpp_updates[0]*100:.1f}%")

        if py_iters:
            py_updates = [i.get('update_magnitude', 0) for i in py_iters]
            print(f"    Py updates:  first={py_updates[0]:.6f}, last={py_updates[-1]:.6f}")
            print(f"                 mean={np.mean(py_updates):.6f}, convergence={(py_updates[0]-py_updates[-1])/py_updates[0]*100:.1f}%")

        # Compare phases
        if cpp_iters and py_iters:
            cpp_phases = [i['phase'] for i in cpp_iters]
            py_phases = [i['phase'] for i in py_iters]

            print(f"    C++ phases: rigid={cpp_phases.count('rigid')}, nonrigid={cpp_phases.count('nonrigid')}")
            print(f"    Py phases:  rigid={py_phases.count('rigid')}, nonrigid={py_phases.count('nonrigid')}")

            # Check if updates are decreasing (good) or increasing (bad)
            if len(cpp_updates) > 1:
                cpp_trend = "CONVERGING" if cpp_updates[-1] < cpp_updates[0] else "DIVERGING"
            else:
                cpp_trend = "SINGLE"

            if len(py_updates) > 1:
                py_trend = "CONVERGING" if py_updates[-1] < py_updates[0] else "DIVERGING"
            else:
                py_trend = "SINGLE"

            print(f"    Trends: C++ {cpp_trend}, Python {py_trend}")

            # Flag issues
            if cpp_trend != py_trend:
                print(f"    ⚠️  DIFFERENT CONVERGENCE BEHAVIOR!")

            if cpp_trend == "DIVERGING" or py_trend == "DIVERGING":
                print(f"    ⚠️  DIVERGENCE DETECTED!")

compare_convergence(cpp_left, py_left, "LEFT")
compare_convergence(cpp_right, py_right, "RIGHT")

# Compare parameters
print("\n" + "=" * 80)
print("PARAMETER COMPARISON")
print("=" * 80)

def compare_params(cpp_data, py_data, eye_name):
    if not cpp_data or not py_data:
        return

    print(f"\n{eye_name} Eye Final Parameters:")

    # Get final parameters
    cpp_final = cpp_data[-1].get('params', {})
    py_final = py_data[-1].get('params', {})

    if cpp_final and py_final:
        cpp_global = cpp_final.get('global', [])
        py_global = py_final.get('global', [])

        if cpp_global and py_global:
            print("  Global params (scale, rot, trans):")
            names = ['scale', 'rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y']
            for i, name in enumerate(names):
                if i < len(cpp_global) and i < len(py_global):
                    diff = py_global[i] - cpp_global[i]
                    print(f"    {name}: C++={cpp_global[i]:.4f}, Py={py_global[i]:.4f}, diff={diff:.4f}")

compare_params(cpp_left, py_left, "LEFT")
compare_params(cpp_right, py_right, "RIGHT")