#!/usr/bin/env python3
"""
Compare C++ and Python parameter updates to identify divergence source.
"""
import numpy as np

print("="*80)
print("PARAMETER UPDATE COMPARISON: C++ vs Python")
print("="*80)

# Parse C++ data
cpp_rigid = {
    'J_w_t_m': [-5335.17871094, 11881.63378906, 14084.36621094, -29703.65625000, -747.70251465, -55.43494415],
    'hessian_diag': [201627.14062500, 274987.90625000, 274987.90625000, 1598694.00000000, 68.00000000, 68.00000000],
    'param_update': [-0.02497768, 0.04437888, 0.05069129, -0.01905321, -10.99562550, -0.81521988],
    'scale_before': 2.81584167,
    'scale_after': 2.79086399,
}

cpp_nonrigid = {
    'J_w_t_m': [31.46384621, 8.54759407, 26.33841324, -97.31203461, -0.03433178, -0.04117662],
    'hessian_diag': [200647.09375000, 278150.65625000, 278415.12500000, 1564024.50000000, 68.00000000, 68.00000000],
    'param_update': [-0.01071294, -0.02572130, -0.01237332, -0.00076114, -0.00050536, -0.00060795],
    'scale_before': 2.79233718,
    'scale_after': 2.77505063,  # From iteration landmarks
}

# Parse Python data (combined rigid+nonrigid)
python = {
    'J_w_t_m': [8007.35104857, 4170.16234147, 16304.62792338, 4947.33278118, -459.02213924, -93.81524631],
    'hessian_diag': [201627.09867445, 274987.48171215, 274987.48171215, 1598691.81328413, 68.00000000, 68.00000000],
    'param_update': [0.03518528, -0.00786299, 0.03772783, 0.00231032, -5.06274443, -1.03472796],
    'scale_before': 2.81583968,
    'scale_after': 2.85102497,
}

print("\n1. HESSIAN DIAGONAL COMPARISON (Rigid params only)")
print("-" * 80)
print("These should be IDENTICAL since they depend on Jacobian and regularization")
print()
print("Parameter  C++ Rigid       Python          Difference      % Diff")
print("-" * 80)
params = ['scale', 'rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y']
for i, param in enumerate(params):
    cpp_val = cpp_rigid['hessian_diag'][i]
    py_val = python['hessian_diag'][i]
    diff = py_val - cpp_val
    pct = 100 * diff / cpp_val if cpp_val != 0 else 0
    print(f"{param:10s} {cpp_val:14.2f}  {py_val:14.2f}  {diff:12.2f}  {pct:8.4f}%")

print("\n2. J_w_t_m COMPARISON (Projected mean-shifts)")
print("-" * 80)
print("C++ RIGID vs Python - These are VERY different!")
print()
print("Parameter  C++ Rigid       Python          Difference      Ratio")
print("-" * 80)
for i, param in enumerate(params):
    cpp_val = cpp_rigid['J_w_t_m'][i]
    py_val = python['J_w_t_m'][i]
    diff = py_val - cpp_val
    ratio = py_val / cpp_val if cpp_val != 0 else float('inf')
    print(f"{param:10s} {cpp_val:14.2f}  {py_val:14.2f}  {diff:12.2f}  {ratio:8.2f}x")

print("\n3. PARAMETER UPDATE COMPARISON")
print("-" * 80)
print("C++ RIGID vs Python")
print()
print("Parameter  C++ Rigid       Python          Difference      Sign Match")
print("-" * 80)
for i, param in enumerate(params):
    cpp_val = cpp_rigid['param_update'][i]
    py_val = python['param_update'][i]
    diff = py_val - cpp_val
    sign_match = "✓" if (cpp_val * py_val) >= 0 else "✗ OPPOSITE"
    print(f"{param:10s} {cpp_val:14.8f}  {py_val:14.8f}  {diff:12.8f}  {sign_match}")

print("\n4. SCALE PARAMETER EVOLUTION")
print("-" * 80)
print(f"Initial:         {cpp_rigid['scale_before']:.8f}")
print(f"C++ after RIGID: {cpp_rigid['scale_after']:.8f} (delta: {cpp_rigid['scale_after'] - cpp_rigid['scale_before']:.8f})")
print(f"Python after:    {python['scale_after']:.8f} (delta: {python['scale_after'] - python['scale_before']:.8f})")
print()
print(f"C++ moved scale DOWN by {cpp_rigid['scale_before'] - cpp_rigid['scale_after']:.8f}")
print(f"Python moved scale UP by {python['scale_after'] - python['scale_before']:.8f}")
print()
print("This is OPPOSITE direction!")

print("\n5. KEY FINDINGS")
print("="*80)
print("""
CRITICAL DISCOVERY: C++ and Python solve DIFFERENT optimization problems!

C++ Approach (two-phase):
  1. RIGID optimization: Only update global params (scale, rotation, translation)
     - Keeps local params (shape) at 0
     - Solves: minimize ||v - J_global·Δp_global||²

  2. NON-RIGID optimization: Update both global and local params
     - Uses result from RIGID as starting point
     - Solves: minimize ||v - J_all·Δp_all||² + λ||Λ^(-1/2)·Δp_local||²

Python Approach (single-phase):
  - Updates ALL params (global + local) simultaneously
  - Solves: minimize ||v - J_all·Δp_all||² + λ||Λ^(-1/2)·Δp_all||²

Impact:
  - Different Jacobian dimensions (6 vs 40 params)
  - Different mean-shift projections (J_w_t_m)
  - Different Hessian structure
  - Different parameter updates
  - Results in ~9px divergence in final landmarks

Root Cause: Python is missing the two-phase optimization!
""")

print("\n6. RECOMMENDED FIX")
print("="*80)
print("""
Implement two-phase optimization in Python:

Phase 1 (RIGID):
  - Create Jacobian for ONLY global params (6 params)
  - Solve for Δp_global while keeping Δp_local = 0
  - Update global params only

Phase 2 (NON-RIGID):
  - Create full Jacobian (6 global + 34 local params)
  - Solve for Δp_all with regularization on local params
  - Update both global and local params

This matches OpenFace's NU_RLMS implementation:
  LandmarkDetectorModel.cpp line 844: rigid=true
  LandmarkDetectorModel.cpp line 868: rigid=false
""")
