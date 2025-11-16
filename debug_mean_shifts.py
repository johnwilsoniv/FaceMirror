#!/usr/bin/env python3
"""
Debug mean-shift computation to find why J_w_t_m doesn't match C++.
"""
import numpy as np

# From debug files:
print("="*80)
print("MEAN-SHIFT ANALYSIS")
print("="*80)

# C++ NONRIGID J_w_t_m (first 6 elements, which are for global params)
cpp_j_w_t_m = np.array([31.46384621, 8.54759407, 26.33841324, -97.31203461, -0.03433178, -0.04117662])

# Python NONRIGID J_w_t_m (first 6 elements)
py_j_w_t_m = np.array([27.05518459, 6219.13920158, 20383.88088840, 5129.11785471, -279.57335087, -60.32688874])

# C++ NONRIGID Hessian diagonal (first 6)
cpp_hessian = np.array([200647.09375000, 278150.65625000, 278415.12500000, 1564024.50000000, 68.00000000, 68.00000000])

# Python NONRIGID Hessian diagonal (first 6)
py_hessian = np.array([201695.95711916, 283034.61928106, 280541.83749307, 1632282.57105087, 68.00000000, 68.00000000])

print("\n1. HESSIAN COMPARISON (should match)")
print("-" * 80)
print(f"{'Param':<10} {'C++':<15} {'Python':<15} {'Diff':<12} {'% Diff':<10}")
print("-" * 80)
params = ['scale', 'rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y']
for i, param in enumerate(params):
    diff = py_hessian[i] - cpp_hessian[i]
    pct = 100 * diff / cpp_hessian[i]
    print(f"{param:<10} {cpp_hessian[i]:<15.2f} {py_hessian[i]:<15.2f} {diff:<12.2f} {pct:<10.4f}%")

print("\n2. J_w_t_m COMPARISON (should match)")
print("-" * 80)
print(f"{'Param':<10} {'C++':<15} {'Python':<15} {'Diff':<15} {'Ratio':<10}")
print("-" * 80)
for i, param in enumerate(params):
    diff = py_j_w_t_m[i] - cpp_j_w_t_m[i]
    ratio = py_j_w_t_m[i] / cpp_j_w_t_m[i] if cpp_j_w_t_m[i] != 0 else float('inf')
    print(f"{param:<10} {cpp_j_w_t_m[i]:<15.2f} {py_j_w_t_m[i]:<15.2f} {diff:<15.2f} {ratio:<10.2f}x")

print("\n3. ANALYSIS")
print("="*80)
print(f"""
Observations:
1. Hessian diagonal is close (within ~3%) → Jacobian is approximately correct
2. J_w_t_m values are WILDLY different for elements 1-5
3. Element 0 (scale) is reasonably close: 31.5 vs 27.1

J_w_t_m = J_rigid^T @ W @ mean_shift

Since Hessian = J^T @ W @ J is close, the issue must be in mean_shift!

Specifically:
- J_w_t_m[0] = scale row of J^T weighted by mean_shifts → close
- J_w_t_m[1-5] = rotation/translation rows → WAY OFF

This suggests mean_shift vector values are incorrect, likely due to:
1. Different offsets calculation
2. Different transform application
3. Bug in mean-shift computation from response maps
""")
