#!/usr/bin/env python3
"""
Test script to verify eye iteration parsing from C++ trace.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from analyze_convergence import load_cpp_trace

# Test with existing trace file
trace_file = Path("/tmp/clnf_iteration_traces/cpp_trace.txt")

if trace_file.exists():
    print("Testing eye iteration parsing...")
    print("=" * 80)

    # Load with eyes included
    face_iters, eye_iters = load_cpp_trace(trace_file, include_eyes=True)

    print(f"Face iterations loaded: {len(face_iters)}")
    print(f"Eye iterations loaded: {len(eye_iters)}")
    print()

    # Analyze face iterations
    if face_iters:
        print("Face model iterations:")
        print(f"  Window sizes used: {sorted(set(i['window_size'] for i in face_iters))}")
        print(f"  Phases: {sorted(set(i['phase'] for i in face_iters))}")
        print(f"  Iteration range: {face_iters[0]['iteration']} - {face_iters[-1]['iteration']}")
        print(f"  Local params per iter: {len(face_iters[0]['params']['local'])}")
        print()

    # Analyze eye iterations
    if eye_iters:
        print("Eye model iterations:")
        print(f"  Window sizes used: {sorted(set(i['window_size'] for i in eye_iters))}")
        print(f"  Phases: {sorted(set(i['phase'] for i in eye_iters))}")
        print(f"  Eye sides: {sorted(set(i.get('eye_side', 'unknown') for i in eye_iters))}")

        # Count iterations per eye
        left_count = sum(1 for i in eye_iters if i.get('eye_side') == 'left')
        right_count = sum(1 for i in eye_iters if i.get('eye_side') == 'right')
        print(f"  Left eye iterations: {left_count}")
        print(f"  Right eye iterations: {right_count}")

        if eye_iters:
            print(f"  Iteration range: {eye_iters[0]['iteration']} - {eye_iters[-1]['iteration']}")
            print(f"  Local params per iter: {len(eye_iters[0]['params']['local'])}")

        # Show sample eye iterations
        print("\nSample eye iterations (first 5):")
        for i, iter_data in enumerate(eye_iters[:5]):
            print(f"    Iter {iter_data['iteration']}: {iter_data['eye_side']} eye, "
                  f"ws={iter_data['window_size']}, phase={iter_data['phase']}, "
                  f"update={iter_data['update_magnitude']:.6f}")

        print("\nWindow size transitions in eye refinement:")
        prev_ws = None
        for iter_data in eye_iters:
            if iter_data['window_size'] != prev_ws:
                print(f"    Iter {iter_data['iteration']}: Window size changed to {iter_data['window_size']}")
                prev_ws = iter_data['window_size']

    # Verify parsing correctness
    print("\nVerification:")
    total_iters_in_file = sum(1 for line in open(trace_file) if not line.startswith('#'))
    total_parsed = len(face_iters) + len(eye_iters)
    print(f"  Total iterations in file: {total_iters_in_file}")
    print(f"  Total iterations parsed: {total_parsed}")
    print(f"  Unaccounted iterations: {total_iters_in_file - total_parsed}")

else:
    print(f"Trace file not found: {trace_file}")
    print("Run analyze_convergence.py first to generate trace data.")