#!/usr/bin/env python3
"""
Debug eye error progression to understand negative convergence.
"""

import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))

from analyze_convergence import load_cpp_trace, run_python_on_frame, analyze_eye_convergence
import cv2
from pyclnf import CLNF

# Load test frame
test_frame = Path("/tmp/clnf_iteration_traces/frame_0.jpg")
if not test_frame.exists():
    print("Test frame not found")
    sys.exit(1)

image = cv2.imread(str(test_frame))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load C++ results
cpp_trace = Path("/tmp/clnf_iteration_traces/cpp_trace.txt")
face_iters, cpp_eye_iters = load_cpp_trace(cpp_trace, include_eyes=True)

# Get C++ final landmarks
# Read the C++ output file for final landmarks
cpp_output = Path("/tmp/clnf_iteration_traces/cpp_output_0/frame_0.csv")
if cpp_output.exists():
    # Read landmarks from CSV
    import csv
    with open(cpp_output, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = next(reader)  # First row has landmarks

    # Extract x,y coordinates (columns are x_0, y_0, x_1, y_1, ...)
    cpp_final_landmarks = []
    for i in range(68):
        x_col = f"x_{i}"
        y_col = f"y_{i}"
        x_idx = header.index(x_col)
        y_idx = header.index(y_col)
        cpp_final_landmarks.append([float(data[x_idx]), float(data[y_idx])])
    cpp_final_landmarks = np.array(cpp_final_landmarks)
else:
    print("No C++ landmarks file found")
    sys.exit(1)

# Run Python with eye tracking
bbox = np.array([297.14, 698.73, 482.76, 471.92])
clnf = CLNF(use_eye_refinement=True)
py_landmarks, py_face_iters, py_eye_iters = run_python_on_frame(gray, bbox, clnf)

if not py_eye_iters:
    print("No Python eye iterations")
    sys.exit(1)

# Analyze eye errors
print("Eye Error Progression Analysis")
print("=" * 80)

# Left eye landmarks
LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))

# Correct eye model mapping
EYE_TO_MAIN_MAP = {8: 0, 10: 1, 12: 2, 14: 3, 16: 4, 18: 5}

# Separate by eye
py_left = [i for i in py_eye_iters if i['eye_side'] == 'left']
py_right = [i for i in py_eye_iters if i['eye_side'] == 'right']

def analyze_eye_errors(eye_iters, cpp_eye_landmarks, eye_name):
    print(f"\n{eye_name} Eye Error Progression:")
    print(f"  Comparing to C++ final eye landmarks")

    if not eye_iters:
        return

    errors = []
    for i, iter_data in enumerate(eye_iters):
        if 'landmarks' in iter_data:
            eye_lms = np.array(iter_data['landmarks'])
            # Extract the 6 mapped eye points
            eye_points = np.array([eye_lms[idx] for idx in EYE_TO_MAIN_MAP.keys()])
            # Compare to C++ final
            error = np.mean(np.linalg.norm(eye_points - cpp_eye_landmarks, axis=1))
            errors.append(error)

            if i == 0 or i == len(eye_iters) - 1:
                print(f"    Iter {i}: error = {error:.3f}px (ws={iter_data['window_size']}, {iter_data['phase']})")

    if len(errors) > 1:
        change = errors[-1] - errors[0]
        pct_change = (change / errors[0]) * 100 if errors[0] > 0 else 0
        print(f"  Error change: {errors[0]:.3f} → {errors[-1]:.3f} ({change:+.3f}px, {pct_change:+.1f}%)")

        if pct_change > 0:
            print(f"  ⚠️  Error INCREASED during refinement")
        else:
            print(f"  ✓ Error decreased during refinement")

    # Check error by window size
    ws_errors = {}
    for iter_data in eye_iters:
        if 'landmarks' in iter_data:
            ws = iter_data['window_size']
            if ws not in ws_errors:
                ws_errors[ws] = []
            eye_lms = np.array(iter_data['landmarks'])
            eye_points = np.array([eye_lms[idx] for idx in EYE_TO_MAIN_MAP.keys()])
            error = np.mean(np.linalg.norm(eye_points - cpp_eye_landmarks, axis=1))
            ws_errors[ws].append(error)

    print(f"\n  Error by window size:")
    for ws in sorted(ws_errors.keys()):
        errors = ws_errors[ws]
        if errors:
            print(f"    WS {ws}: {errors[0]:.3f} → {errors[-1]:.3f} ({(errors[-1]-errors[0]):+.3f}px)")

# Analyze each eye
analyze_eye_errors(py_left, cpp_final_landmarks[LEFT_EYE_IDX], "Left")
analyze_eye_errors(py_right, cpp_final_landmarks[RIGHT_EYE_IDX], "Right")

# Check if initial main model eye landmarks are actually better
print("\n" + "=" * 80)
print("Main Model Eye Landmarks vs C++:")

if py_face_iters and py_face_iters[-1].get('landmarks'):
    main_final = np.array(py_face_iters[-1]['landmarks'])

    left_main_error = np.mean(np.linalg.norm(main_final[LEFT_EYE_IDX] - cpp_final_landmarks[LEFT_EYE_IDX], axis=1))
    right_main_error = np.mean(np.linalg.norm(main_final[RIGHT_EYE_IDX] - cpp_final_landmarks[RIGHT_EYE_IDX], axis=1))

    print(f"  Main model left eye error:  {left_main_error:.3f}px")
    print(f"  Main model right eye error: {right_main_error:.3f}px")

    # Compare to eye model final
    if py_left and py_left[-1].get('landmarks'):
        eye_lms = np.array(py_left[-1]['landmarks'])
        eye_points = np.array([eye_lms[idx] for idx in EYE_TO_MAIN_MAP.keys()])
        eye_error = np.mean(np.linalg.norm(eye_points - cpp_final_landmarks[LEFT_EYE_IDX], axis=1))
        print(f"  Eye model left eye error:   {eye_error:.3f}px")
        if eye_error > left_main_error:
            print(f"  ⚠️  Eye refinement made left eye WORSE by {eye_error - left_main_error:.3f}px")

    if py_right and py_right[-1].get('landmarks'):
        eye_lms = np.array(py_right[-1]['landmarks'])
        eye_points = np.array([eye_lms[idx] for idx in EYE_TO_MAIN_MAP.keys()])
        eye_error = np.mean(np.linalg.norm(eye_points - cpp_final_landmarks[RIGHT_EYE_IDX], axis=1))
        print(f"  Eye model right eye error:  {eye_error:.3f}px")
        if eye_error > right_main_error:
            print(f"  ⚠️  Eye refinement made right eye WORSE by {eye_error - right_main_error:.3f}px")