#!/usr/bin/env python3
"""
Visualize C++ vs Python eye landmark positions on the same image.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))

def parse_cpp_landmarks(filepath):
    """Parse C++ eye init debug file."""
    landmarks = {}
    current_model = None

    with open(filepath, 'r') as f:
        for line in f:
            if 'Model name:' in line:
                current_model = line.split(':')[1].strip()
                landmarks[current_model] = {}
            elif current_model and line.strip().startswith(tuple(str(i) + ':' for i in range(28))):
                parts = line.strip().split(':')
                idx = int(parts[0])
                coords = parts[1].strip().strip('()').split(',')
                x, y = float(coords[0]), float(coords[1])
                landmarks[current_model][idx] = (x, y)

    return landmarks

def parse_python_landmarks(filepath):
    """Parse Python eye model detailed debug file."""
    landmarks = {}
    in_landmarks = False

    with open(filepath, 'r') as f:
        for line in f:
            if 'Initial eye landmarks (28 points):' in line:
                in_landmarks = True
                continue
            if in_landmarks:
                if line.strip().startswith(tuple(str(i) + ':' for i in range(28))):
                    parts = line.strip().split(':')
                    idx = int(parts[0])
                    coords = parts[1].strip().strip('()').split(',')
                    x, y = float(coords[0]), float(coords[1])
                    landmarks[idx] = (x, y)
                elif '===' in line or not line.strip():
                    break

    return landmarks

def main():
    # Load test frame
    img = cv2.imread('/tmp/test_frame.jpg')
    if img is None:
        print("Error: Could not load /tmp/test_frame.jpg")
        return

    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # Parse C++ landmarks
    cpp_landmarks = parse_cpp_landmarks('/tmp/cpp_eye_init_debug.txt')
    cpp_left = cpp_landmarks.get('left_eye_28', {})
    cpp_right = cpp_landmarks.get('right_eye_28', {})

    # Parse Python landmarks
    python_left = parse_python_landmarks('/tmp/python_eye_model_detailed.txt')

    print(f"\nC++ left eye: {len(cpp_left)} landmarks")
    print(f"Python left eye: {len(python_left)} landmarks")

    # Compare key landmarks
    print("\n=== Landmark Position Comparison (Left Eye) ===")
    key_landmarks = [0, 8, 10, 12, 14, 16, 18]  # 0=pupil, rest=eyelids
    for idx in key_landmarks:
        cpp_pos = cpp_left.get(idx, (0, 0))
        py_pos = python_left.get(idx, (0, 0))
        dx = cpp_pos[0] - py_pos[0]
        dy = cpp_pos[1] - py_pos[1]
        dist = np.sqrt(dx*dx + dy*dy)
        name = "pupil" if idx == 0 else f"eyelid"
        print(f"  {idx} ({name}): C++({cpp_pos[0]:.1f}, {cpp_pos[1]:.1f}) vs Py({py_pos[0]:.1f}, {py_pos[1]:.1f}) = {dist:.2f}px diff")

    # Create visualization
    vis = img.copy()

    # Draw C++ landmarks in RED
    for idx, (x, y) in cpp_left.items():
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)  # Red
        if idx in [0, 8, 10, 12, 14, 16, 18]:
            cv2.putText(vis, f"C{idx}", (int(x)+5, int(y)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # Draw Python landmarks in GREEN
    for idx, (x, y) in python_left.items():
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)  # Green
        if idx in [0, 8, 10, 12, 14, 16, 18]:
            cv2.putText(vis, f"P{idx}", (int(x)+5, int(y)+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # Draw lines between corresponding landmarks
    for idx in python_left.keys():
        if idx in cpp_left:
            cpp_pos = cpp_left[idx]
            py_pos = python_left[idx]
            cv2.line(vis, (int(cpp_pos[0]), int(cpp_pos[1])),
                    (int(py_pos[0]), int(py_pos[1])), (255, 255, 0), 1)

    # Crop to left eye region
    if python_left:
        all_x = [p[0] for p in python_left.values()]
        all_y = [p[1] for p in python_left.values()]
        min_x, max_x = int(min(all_x)) - 50, int(max(all_x)) + 50
        min_y, max_y = int(min(all_y)) - 50, int(max(all_y)) + 50
        eye_crop = vis[max(0, min_y):max_y, max(0, min_x):max_x]

        # Scale up for better viewing
        scale = 3
        eye_crop_large = cv2.resize(eye_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # Add legend
        cv2.putText(eye_crop_large, "RED = C++", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(eye_crop_large, "GREEN = Python", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save
        out_path = '/tmp/eye_landmarks_comparison.jpg'
        cv2.imwrite(out_path, eye_crop_large)
        print(f"\nVisualization saved to: {out_path}")

    # Also show patch extraction locations
    print("\n=== Patch Extraction Analysis ===")
    print("C++ and Python extract 11x11 patches centered on landmark 0:")
    cpp_l0 = cpp_left.get(0, (0, 0))
    py_l0 = python_left.get(0, (0, 0))
    print(f"  C++ center: ({cpp_l0[0]:.1f}, {cpp_l0[1]:.1f})")
    print(f"  Python center: ({py_l0[0]:.1f}, {py_l0[1]:.1f})")

    # Extract actual patches to compare
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # C++ patch location
    cx, cy = int(cpp_l0[0]), int(cpp_l0[1])
    cpp_patch = gray[cy-5:cy+6, cx-5:cx+6]

    # Python patch location
    px, py = int(py_l0[0]), int(py_l0[1])
    py_patch = gray[py-5:py+6, px-5:px+6]

    if cpp_patch.size > 0 and py_patch.size > 0:
        print(f"\n  C++ patch values (5x5 center): mean={cpp_patch.mean():.1f}, range=[{cpp_patch.min()}, {cpp_patch.max()}]")
        print(f"  Python patch values (5x5 center): mean={py_patch.mean():.1f}, range=[{py_patch.min()}, {py_patch.max()}]")

        # Show the actual pixel difference
        print(f"\n  Patch intensity difference: {abs(cpp_patch.mean() - py_patch.mean()):.1f}")

if __name__ == "__main__":
    main()
