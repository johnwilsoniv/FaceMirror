#!/usr/bin/env python3
"""
Debug script to check what data is in eye iterations.
"""

import sys
from pathlib import Path
import json
sys.path.insert(0, str(Path(__file__).parent))

# Check the saved Python eye trace
trace_file = Path("/tmp/clnf_iteration_traces/python_eye_trace.txt")

if trace_file.exists():
    print("Python Eye Trace File Contents:")
    print("=" * 80)

    with open(trace_file, 'r') as f:
        lines = f.readlines()

    # Skip header
    data_lines = [l for l in lines if not l.startswith('#')]

    print(f"Total eye iterations: {len(data_lines)}")

    if data_lines:
        # Parse first few lines to understand structure
        print("\nFirst 5 iterations:")
        for i, line in enumerate(data_lines[:5]):
            parts = line.strip().split()
            if len(parts) >= 4:
                iter_num = parts[0]
                window_size = parts[1]
                phase = parts[2]
                eye_side = parts[3]
                num_params = len(parts) - 4
                print(f"  {i}: iter={iter_num}, ws={window_size}, phase={phase}, eye={eye_side}, params={num_params}")

        # Count by eye and phase
        left_rigid = sum(1 for l in data_lines if 'rigid left' in l)
        left_nonrigid = sum(1 for l in data_lines if 'nonrigid left' in l)
        right_rigid = sum(1 for l in data_lines if 'rigid right' in l)
        right_nonrigid = sum(1 for l in data_lines if 'nonrigid right' in l)

        print(f"\nBreakdown:")
        print(f"  Left eye:  {left_rigid} rigid + {left_nonrigid} nonrigid = {left_rigid + left_nonrigid} total")
        print(f"  Right eye: {right_rigid} rigid + {right_nonrigid} nonrigid = {right_rigid + right_nonrigid} total")

        # Check parameter count (should be 10 for eye model: 6 global + 4 local)
        if data_lines:
            sample = data_lines[0].strip().split()
            param_count = len(sample) - 4  # subtract iter, ws, phase, side
            print(f"\nParameters per iteration: {param_count} (expected: 10)")
            if param_count == 10:
                print("  ✓ Correct parameter count for eye model")
            else:
                print("  ⚠️ Unexpected parameter count")
else:
    print(f"Error: Trace file not found at {trace_file}")
    print("Run analyze_convergence.py first")

# Also check if we can read the data back
print("\n" + "=" * 80)
print("Testing load_cpp_trace with eye iterations:")

from analyze_convergence import load_cpp_trace

cpp_trace = Path("/tmp/clnf_iteration_traces/cpp_trace.txt")
if cpp_trace.exists():
    face_iters, eye_iters = load_cpp_trace(cpp_trace, include_eyes=True)
    print(f"C++ face iterations: {len(face_iters)}")
    print(f"C++ eye iterations: {len(eye_iters)}")

    if eye_iters:
        # Check structure
        sample = eye_iters[0]
        print(f"\nSample eye iteration structure:")
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Window size: {sample.get('window_size')}")
        print(f"  Phase: {sample.get('phase')}")
        print(f"  Eye side: {sample.get('eye_side')}")
        print(f"  Global params: {len(sample['params']['global'])}")
        print(f"  Local params: {len(sample['params']['local'])}")