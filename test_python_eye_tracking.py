#!/usr/bin/env python3
"""
Test script to verify Python eye iteration tracking is working.
"""

import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))

from analyze_convergence import run_python_on_frame, load_cpp_trace, TRACE_DIR
import cv2
from pyclnf import CLNF

def test_eye_tracking():
    """Test that eye iteration tracking works in Python CLNF."""

    # Load test image
    test_image_path = TRACE_DIR / "frame_0.jpg"
    if not test_image_path.exists():
        print(f"Error: Test image not found at {test_image_path}")
        print("Run analyze_convergence.py first to generate test frame")
        return

    image = cv2.imread(str(test_image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize CLNF with eye refinement
    clnf = CLNF(use_eye_refinement=True)

    # Use a test bounding box (can be adjusted based on test image)
    # This is a typical face bbox
    h, w = gray.shape
    bbox = np.array([w*0.25, h*0.15, w*0.5, h*0.7])

    print("Testing Python eye iteration tracking...")
    print("=" * 80)

    # Run Python CLNF with iteration tracking
    landmarks, face_iters, eye_iters = run_python_on_frame(gray, bbox, clnf)

    if landmarks is None:
        print("Error: Python CLNF fit failed")
        return

    print(f"\nResults:")
    print(f"  Face iterations: {len(face_iters)}")
    print(f"  Eye iterations: {len(eye_iters)}")

    # Analyze eye iterations
    if eye_iters:
        # Count by eye side
        left_count = sum(1 for i in eye_iters if i.get('eye_side') == 'left')
        right_count = sum(1 for i in eye_iters if i.get('eye_side') == 'right')

        print(f"\nEye iteration breakdown:")
        print(f"  Left eye: {left_count} iterations")
        print(f"  Right eye: {right_count} iterations")

        # Get unique window sizes
        window_sizes = sorted(set(i['window_size'] for i in eye_iters))
        print(f"  Window sizes: {window_sizes}")

        # Get phases
        phases = sorted(set(i['phase'] for i in eye_iters))
        print(f"  Phases: {phases}")

        # Show sample iterations
        print("\nSample eye iterations (first 5):")
        for i, iter_data in enumerate(eye_iters[:5]):
            print(f"    Iter {iter_data['iteration']}: {iter_data['eye_side']} eye, "
                  f"ws={iter_data['window_size']}, phase={iter_data['phase']}, "
                  f"update={iter_data['update_magnitude']:.6f}")

        # Check parameter dimensions
        if eye_iters:
            sample = eye_iters[0]
            print(f"\nParameter dimensions:")
            print(f"  Global params: {len(sample['params']['global'])} (should be 6)")
            print(f"  Local params: {len(sample['params']['local'])} (should be 4 for eye model)")
            print(f"  Landmarks: {len(sample['landmarks'])} points (should be 28)")

        # Save to file for inspection
        trace_file = TRACE_DIR / "test_python_eye_trace.txt"
        with open(trace_file, 'w') as f:
            f.write("# Test Python Eye Iterations\n")
            for iter_data in eye_iters:
                f.write(f"{iter_data['iteration']} {iter_data['eye_side']} "
                       f"{iter_data['window_size']} {iter_data['phase']} "
                       f"update={iter_data['update_magnitude']:.6f}\n")
        print(f"\nSaved eye trace to: {trace_file}")
    else:
        print("\nNo eye iterations captured - check if eye refinement is enabled")

    # Compare with C++ if available
    cpp_trace_file = TRACE_DIR / "cpp_trace.txt"
    if cpp_trace_file.exists():
        print("\nComparing with C++ eye iterations...")
        face_iters_cpp, eye_iters_cpp = load_cpp_trace(cpp_trace_file, include_eyes=True)

        print(f"  C++ eye iterations: {len(eye_iters_cpp)}")
        print(f"  Python eye iterations: {len(eye_iters)}")

        if eye_iters_cpp:
            cpp_left = sum(1 for i in eye_iters_cpp if i.get('eye_side') == 'left')
            cpp_right = sum(1 for i in eye_iters_cpp if i.get('eye_side') == 'right')
            print(f"  C++ left eye: {cpp_left}, Python left eye: {left_count}")
            print(f"  C++ right eye: {cpp_right}, Python right eye: {right_count}")

if __name__ == "__main__":
    test_eye_tracking()