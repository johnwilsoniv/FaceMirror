#!/usr/bin/env python3
"""
Compare Python eye refinement with C++ using identical input landmarks.

This script feeds C++ pre-refinement landmarks into Python's eye model
and compares response maps and mean-shifts iteration by iteration.
"""

import numpy as np
import cv2
from pathlib import Path

# Add paths
import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

from pyclnf.core.eye_patch_expert import HierarchicalEyeModel, align_shapes_with_scale
from pyclnf.core.eye_patch_expert import LEFT_EYE_MAPPING, RIGHT_EYE_MAPPING

# C++ pre-refinement landmarks (from /tmp/cpp_eye_model_debug.txt)
CPP_PRE_LANDMARKS = {
    36: (391.1343, 827.5673),
    37: (409.2716, 807.8469),
    38: (436.2549, 806.1482),
    39: (461.1137, 823.6552),
    40: (437.5308, 830.7004),
    41: (410.2859, 833.9506),
    42: (557.6241, 816.1306),
    43: (580.6108, 794.6487),
    44: (608.5191, 792.8670),
    45: (632.3162, 808.2702),
    46: (612.4580, 817.1681),
    47: (584.8137, 819.5327),
}

# C++ post-refinement landmarks
CPP_POST_LANDMARKS = {
    36: (391.5277, 830.1721),
    37: (409.6088, 811.0718),
    38: (436.0404, 808.9324),
    39: (460.0428, 824.4908),
    40: (437.0960, 832.4396),
    41: (410.6767, 835.8009),
    42: (558.8107, 816.5471),
    43: (580.9568, 797.1573),
    44: (608.3911, 795.6616),
    45: (631.6840, 810.1879),
    46: (611.9465, 818.6151),
    47: (585.0833, 820.6722),
}

def load_test_image():
    """Load the same test image used by C++."""
    # Use the Shorty video (same as used in other tests)
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame from {video_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def create_68_landmarks_from_cpp():
    """Create 68-point landmark array from C++ eye landmarks."""
    landmarks = np.zeros((68, 2), dtype=np.float32)

    # Fill in eye landmarks from C++
    for idx, (x, y) in CPP_PRE_LANDMARKS.items():
        landmarks[idx] = [x, y]

    # Fill in other landmarks with reasonable values (won't be used)
    # Just need face center for scale estimation
    face_center_x = (landmarks[36, 0] + landmarks[45, 0]) / 2
    face_center_y = (landmarks[36, 1] + landmarks[45, 1]) / 2

    # Nose tip (30)
    landmarks[30] = [face_center_x, face_center_y + 70]

    # Chin (8)
    landmarks[8] = [face_center_x, face_center_y + 200]

    return landmarks

def compare_eye_refinement():
    """Run Python eye refinement with C++ landmarks and compare results."""
    print("=" * 70)
    print("COMPARE C++ vs PYTHON EYE REFINEMENT")
    print("=" * 70)

    # Load image and model
    print("\nLoading test image and eye model...")
    image = load_test_image()

    model_dir = "/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models"
    eye_model = HierarchicalEyeModel(model_dir)

    # Create landmarks from C++ values
    cpp_landmarks = create_68_landmarks_from_cpp()

    print("\nC++ Input Landmarks (left eye):")
    for idx in [36, 37, 38, 39, 40, 41]:
        x, y = cpp_landmarks[idx]
        print(f"  {idx}: ({x:.4f}, {y:.4f})")

    # Run Python refinement with C++ landmarks
    print("\nRunning Python eye refinement with C++ input landmarks...")

    # We need main_rotation - estimate from eye positions
    # C++ had: rot=(-0.127467, 0.164196, -0.097800)
    main_rotation = np.array([-0.127, 0.164, -0.098])

    # Refine left eye
    refined = eye_model.refine_eye_landmarks(
        image, cpp_landmarks, 'left',
        main_rotation=main_rotation,
        main_scale=1.0
    )

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON: C++ vs Python Refinement")
    print("=" * 70)

    print("\nLeft Eye Landmarks (36-41):")
    print(f"{'Idx':<4} {'C++ Pre':<20} {'C++ Post':<20} {'Python Post':<20} {'Diff':<10}")
    print("-" * 80)

    total_cpp_change = 0
    total_py_change = 0
    total_diff = 0

    for idx in [36, 37, 38, 39, 40, 41]:
        cpp_pre = CPP_PRE_LANDMARKS[idx]
        cpp_post = CPP_POST_LANDMARKS[idx]
        py_post = refined[idx]

        cpp_change = np.sqrt((cpp_post[0] - cpp_pre[0])**2 + (cpp_post[1] - cpp_pre[1])**2)
        py_change = np.sqrt((py_post[0] - cpp_pre[0])**2 + (py_post[1] - cpp_pre[1])**2)
        diff = np.sqrt((py_post[0] - cpp_post[0])**2 + (py_post[1] - cpp_post[1])**2)

        total_cpp_change += cpp_change
        total_py_change += py_change
        total_diff += diff

        print(f"{idx:<4} ({cpp_pre[0]:.1f},{cpp_pre[1]:.1f})"
              f"  ({cpp_post[0]:.1f},{cpp_post[1]:.1f})"
              f"  ({py_post[0]:.1f},{py_post[1]:.1f})"
              f"  {diff:.2f}px")

    print("-" * 80)
    print(f"Mean C++ movement: {total_cpp_change/6:.2f}px")
    print(f"Mean Python movement: {total_py_change/6:.2f}px")
    print(f"Mean difference from C++: {total_diff/6:.2f}px")

    # Load C++ detailed debug for comparison
    print("\n" + "=" * 70)
    print("DETAILED ITERATION COMPARISON")
    print("=" * 70)

    # Read Python debug output
    py_debug_path = "/tmp/python_eye_model_detailed.txt"
    cpp_debug_path = "/tmp/cpp_eye_model_detailed.txt"

    print("\nPython debug saved to:", py_debug_path)
    print("C++ debug available at:", cpp_debug_path)

    # Parse and compare first iteration mean-shifts
    print("\n--- First Iteration Mean-Shift Comparison (Landmark 8 = Main 36) ---")

    # Read Python mean-shifts from debug file
    try:
        with open(py_debug_path, 'r') as f:
            content = f.read()

        # Find sim_ref_to_img values
        if "sim_ref_to_img:" in content:
            for line in content.split('\n'):
                if "sim_ref_to_img: a1=" in line:
                    print(f"\nPython {line.strip()}")
                    break

        # Find first iteration mean-shifts for landmark 8
        lines = content.split('\n')
        in_iter0 = False
        for i, line in enumerate(lines):
            if "Iteration 0 (NONRIGID" in line:
                in_iter0 = True
            elif in_iter0 and "8: ms=" in line:
                print(f"Python Iter 0 Eye_8: {line.strip()}")
                break
            elif in_iter0 and "Iteration 1" in line:
                break

    except Exception as e:
        print(f"Could not parse Python debug: {e}")

    # Parse C++ mean-shifts
    try:
        with open(cpp_debug_path, 'r') as f:
            cpp_content = f.read()

        # Find first iteration for left eye (model 0)
        lines = cpp_content.split('\n')
        found_model0 = False
        in_iter0 = False

        for i, line in enumerate(lines):
            if "Iteration 0 (RIGID):" in line and not found_model0:
                found_model0 = True
                in_iter0 = True
            elif found_model0 and in_iter0 and "Eye_8:" in line:
                print(f"C++ Iter 0 Eye_8: {line.strip()}")
                break

    except Exception as e:
        print(f"Could not parse C++ debug: {e}")

    # Also compare response maps
    print("\n--- Response Map Comparison ---")

    py_response_path = "/tmp/python_eye_response_maps.txt"
    cpp_response_path = "/tmp/cpp_eye_response_maps.txt"

    try:
        with open(py_response_path, 'r') as f:
            py_resp = f.read()

        # Find Eye_8 response map stats
        for line in py_resp.split('\n'):
            if "Eye landmark 8" in line or ("min:" in line and "max:" in line):
                print(f"Python: {line.strip()}")
                if "mean:" in line:
                    break

    except Exception as e:
        print(f"Could not read Python response: {e}")

    try:
        with open(cpp_response_path, 'r') as f:
            cpp_resp = f.read()

        # Find Eye_8 response map
        for line in cpp_resp.split('\n'):
            if "Eye_8" in line or "Landmark 8" in line:
                print(f"C++: {line.strip()}")
                break

    except Exception as e:
        print(f"Could not read C++ response: {e}")

    print("\n" + "=" * 70)
    print("Done! Check debug files for detailed iteration comparison.")
    print("=" * 70)

if __name__ == "__main__":
    compare_eye_refinement()
