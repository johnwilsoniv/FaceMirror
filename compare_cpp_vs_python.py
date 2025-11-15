#!/usr/bin/env python3
"""
Compare C++ OpenFace vs Python CLNF landmarks for the same image.
"""

import pandas as pd
import json
import numpy as np

# Read C++ output
cpp_csv = 'validation_output/cpp_baseline/patient1_frame1.csv'
cpp_df = pd.read_csv(cpp_csv)

# Read Python debug output
with open('debug_output/clnf_debug_info.json', 'r') as f:
    py_debug = json.load(f)

# Extract tracked landmarks
tracked_lms = [36, 48, 30, 8]

print("="*80)
print("C++ OpenFace vs Python CLNF - Final Landmark Comparison")
print("="*80)
print(f"\nImage: patient1_frame1.jpg")
print(f"BBox: {py_debug['bbox']}")
print(f"\nFinal Landmark Positions:")
print(f"{'LM':<4} {'C++ X':>10} {'C++ Y':>10} {'Py X':>10} {'Py Y':>10} {'ΔX':>8} {'ΔY':>8} {'Distance':>10}")
print("-"*80)

py_final = py_debug['final']['landmarks']
total_error = 0
for lm_idx in tracked_lms:
    cpp_x = cpp_df[f'x_{lm_idx}'].values[0]
    cpp_y = cpp_df[f'y_{lm_idx}'].values[0]
    py_x = py_final[lm_idx][0]
    py_y = py_final[lm_idx][1]

    dx = py_x - cpp_x
    dy = py_y - cpp_y
    dist = np.sqrt(dx**2 + dy**2)
    total_error += dist

    print(f"{lm_idx:<4} {cpp_x:>10.2f} {cpp_y:>10.2f} {py_x:>10.2f} {py_y:>10.2f} {dx:>8.2f} {dy:>8.2f} {dist:>10.2f}")

avg_error = total_error / len(tracked_lms)
print("-"*80)
print(f"Average error across tracked landmarks: {avg_error:.2f} pixels\n")

# Check initialization
print("\n" + "="*80)
print("Initialization Comparison")
print("="*80)
print(f"\nPython Initial Landmarks:")
py_init = py_debug['initialization']['tracked_landmarks']
for lm_idx in tracked_lms:
    pos = py_init[str(lm_idx)]
    print(f"  LM{lm_idx}: ({pos[0]:.2f}, {pos[1]:.2f})")

print(f"\nC++ has final landmarks only (no initialization in CSV)")
print("Note: C++ likely uses similar PDM initialization from bbox")

# Show convergence details
print("\n" + "="*80)
print("Python Convergence Details")
print("="*80)
for stage in py_debug['window_stages']:
    ws = stage['window_size']
    iters = len(stage['optimization']['iterations'])
    converged = stage['info']['converged']
    print(f"\nWindow {ws}: {iters} iterations, converged={converged}")

    # Show landmark drift
    first_iter = stage['optimization']['iterations'][0]
    last_iter = stage['optimization']['iterations'][-1]

    for lm_idx in [36, 48]:  # Just show 2 landmarks
        first_pos = first_iter['tracked_landmarks'][str(lm_idx)]
        last_pos = last_iter['tracked_landmarks'][str(lm_idx)]
        dx = last_pos[0] - first_pos[0]
        dy = last_pos[1] - first_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        print(f"  LM{lm_idx}: ({first_pos[0]:.1f}, {first_pos[1]:.1f}) -> ({last_pos[0]:.1f}, {last_pos[1]:.1f}) moved {dist:.1f}px")

print("\n" + "="*80)
