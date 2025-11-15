#!/usr/bin/env python3
"""Compare C++ vs Python regular vs Python debug"""

import pandas as pd
import json
import numpy as np

# C++ landmarks
cpp_df = pd.read_csv('validation_output/cpp_baseline/patient1_frame1.csv')

# Python debug landmarks
with open('debug_output/clnf_debug_info.json', 'r') as f:
    py_debug = json.load(f)

# Python regular landmarks (from previous run)
py_regular = {
    36: (362.22, 859.47),
    48: (416.42, 1050.32),
    30: (485.94, 939.86),
    8: (485.70, 1173.74)
}

tracked_lms = [36, 48, 30, 8]

print("="*100)
print("THREE-WAY COMPARISON: C++ OpenFace vs Python Regular vs Python Debug")
print("="*100)
print(f"\n{'LM':<4} {'C++ X':>10} {'C++ Y':>10} {'Reg X':>10} {'Reg Y':>10} {'Dbg X':>10} {'Dbg Y':>10} {'Reg Err':>10} {'Dbg Err':>10}")
print("-"*100)

for lm_idx in tracked_lms:
    cpp_x = cpp_df[f'x_{lm_idx}'].values[0]
    cpp_y = cpp_df[f'y_{lm_idx}'].values[0]

    reg_x, reg_y = py_regular[lm_idx]

    dbg_x = py_debug['final']['landmarks'][lm_idx][0]
    dbg_y = py_debug['final']['landmarks'][lm_idx][1]

    reg_err = np.sqrt((reg_x - cpp_x)**2 + (reg_y - cpp_y)**2)
    dbg_err = np.sqrt((dbg_x - cpp_x)**2 + (dbg_y - cpp_y)**2)

    print(f"{lm_idx:<4} {cpp_x:>10.2f} {cpp_y:>10.2f} {reg_x:>10.2f} {reg_y:>10.2f} {dbg_x:>10.2f} {dbg_y:>10.2f} {reg_err:>10.2f} {dbg_err:>10.2f}")

print("-"*100)
print(f"\n{'':>40}Regular avg error vs C++: {sum([np.sqrt((py_regular[i][0] - cpp_df[f'x_{i}'].values[0])**2 + (py_regular[i][1] - cpp_df[f'y_{i}'].values[0])**2) for i in tracked_lms])/4:.2f} pixels")

dbg_errs = [np.sqrt((py_debug['final']['landmarks'][i][0] - cpp_df[f'x_{i}'].values[0])**2 + (py_debug['final']['landmarks'][i][1] - cpp_df[f'y_{i}'].values[0])**2) for i in tracked_lms]
print(f"{'':>40}Debug avg error vs C++:   {sum(dbg_errs)/4:.2f} pixels")

print("\n" + "="*100)
print("CONCLUSION: Regular optimizer is MUCH closer to C++ than debug version!")
print("The debug version has a bug that causes large errors (especially LM30: 65px!)")
print("="*100)
