#!/usr/bin/env python3
"""
Trace bbox coordinate transformations step-by-step to match C++ exactly.
Compare Python vs expected C++ behavior at each step.
"""

import cv2
import numpy as np

print("="*80)
print("BBOX COORDINATE TRANSFORMATION TRACE")
print("="*80)

# Test the exact transformations from the debug output
print("\n1. C++ BBOX GENERATION (generate_bounding_boxes)")
print("-"*80)
print("C++ code (FaceDetectorMTCNN.cpp:771-777):")
print("""
    float min_x = int((stride * x + 1) / scale);
    float max_x = int((stride * x + face_support) / scale);
    float min_y = int((stride * y + 1) / scale);
    float max_y = int((stride * y + face_support) / scale);
    o_bounding_boxes.push_back(cv::Rect_<float>(min_x, min_y, max_x - min_x, max_y - min_y));
""")

# Simulate a detection at scale 7 (0.027) which generates the face
scale = 0.027017483878958408
stride = 2
face_support = 12

# Find the heatmap position that corresponds to face region ~(259, 703)
# Working backwards: min_x = int((stride * x + 1) / scale)
# 259 ≈ int((2*x + 1) / 0.027)
# 259 * 0.027 ≈ 2*x + 1
# x ≈ (259 * 0.027 - 1) / 2
x = int((259 * scale - 1) / stride)
y = int((703 * scale - 1) / stride)

print(f"\nExample: Heatmap position (x={x}, y={y}) at scale={scale:.6f}")

# C++ calculation
cpp_min_x = int((stride * x + 1) / scale)
cpp_max_x = int((stride * x + face_support) / scale)
cpp_min_y = int((stride * y + 1) / scale)
cpp_max_y = int((stride * y + face_support) / scale)
cpp_w = cpp_max_x - cpp_min_x
cpp_h = cpp_max_y - cpp_min_y

print(f"C++ bbox: x={cpp_min_x}, y={cpp_min_y}, w={cpp_w}, h={cpp_h}")

# Python calculation (typical implementation)
py_min_x = np.floor((stride * x + 1) / scale).astype(np.int32)
py_max_x = np.floor((stride * x + face_support) / scale).astype(np.int32)
py_min_y = np.floor((stride * y + 1) / scale).astype(np.int32)
py_max_y = np.floor((stride * y + face_support) / scale).astype(np.int32)
py_w = py_max_x - py_min_x
py_h = py_max_y - py_min_y

print(f"Python bbox: x={py_min_x}, y={py_min_y}, w={py_w}, h={py_h}")
print(f"Difference: Δx={py_min_x-cpp_min_x}, Δy={py_min_y-cpp_min_y}, Δw={py_w-cpp_w}, Δh={py_h-cpp_h}")

print("\n2. BBOX REGRESSION (apply_correction)")
print("-"*80)
print("C++ code (FaceDetectorMTCNN.cpp:828-832):")
print("""
    float new_min_x = curr_box.x + corrections[i].x * curr_box.width;
    float new_min_y = curr_box.y + corrections[i].y * curr_box.height;
    float new_max_x = curr_box.x + curr_box.width + curr_box.width * corrections[i].width;
    float new_max_y = curr_box.y + curr_box.height + curr_box.height * corrections[i].height;
    total_bboxes[i] = cv::Rect_<float>(new_min_x, new_min_y, new_max_x - new_min_x, new_max_y - new_min_y);
""")

# Simulate regression coefficients (typical values)
corr_x, corr_y, corr_w, corr_h = 0.0, 0.0, 0.0, 0.0  # Assuming no regression for simplicity

# C++ with add1=false (FaceDetectorMTCNN.cpp:977)
cpp_x = cpp_min_x
cpp_y = cpp_min_y
cpp_x2 = cpp_max_x
cpp_y2 = cpp_max_y

print(f"\nC++ after regression: x={cpp_x}, y={cpp_y}, x2={cpp_x2}, y2={cpp_y2}")

# Python
py_x = py_min_x
py_y = py_min_y
py_x2 = py_max_x
py_y2 = py_max_y

print(f"Python after regression: x={py_x}, y={py_y}, x2={py_x2}, y2={py_y2}")

print("\n3. RECTIFY / SQUARE BBOX")
print("-"*80)
print("C++ rectify (FaceDetectorMTCNN.cpp:799-811):")
print("""
    float height = total_bboxes[i].height;
    float width = total_bboxes[i].width;
    float max_side = std::max(width, height);
    float new_min_x = total_bboxes[i].x + 0.5 * (width - max_side);
    float new_min_y = total_bboxes[i].y + 0.5 * (height - max_side);
    total_bboxes[i].x = (int)new_min_x;
    total_bboxes[i].y = (int)new_min_y;
    total_bboxes[i].width = (int)max_side;
    total_bboxes[i].height = (int)max_side;
""")

# C++ rectify
cpp_w_rect = cpp_x2 - cpp_x
cpp_h_rect = cpp_y2 - cpp_y
cpp_max_side = max(cpp_w_rect, cpp_h_rect)
cpp_new_x = cpp_x + 0.5 * (cpp_w_rect - cpp_max_side)
cpp_new_y = cpp_y + 0.5 * (cpp_h_rect - cpp_max_side)
cpp_x_rect = int(cpp_new_x)
cpp_y_rect = int(cpp_new_y)
cpp_w_rect = int(cpp_max_side)
cpp_h_rect = int(cpp_max_side)

print(f"\nC++ after rectify: x={cpp_x_rect}, y={cpp_y_rect}, w={cpp_w_rect}, h={cpp_h_rect}")

# Python _square_bbox (typical implementation)
py_w_rect = py_x2 - py_x
py_h_rect = py_y2 - py_y
py_max_side = max(py_w_rect, py_h_rect)
py_new_x = py_x + 0.5 * (py_w_rect - py_max_side)
py_new_y = py_y + 0.5 * (py_h_rect - py_max_side)
py_x_rect = int(py_new_x)
py_y_rect = int(py_new_y)
py_w_rect = int(py_max_side)
py_h_rect = int(py_max_side)

print(f"Python after square: x={py_x_rect}, y={py_y_rect}, w={py_w_rect}, h={py_h_rect}")
print(f"Difference: Δx={py_x_rect-cpp_x_rect}, Δy={py_y_rect-cpp_y_rect}, Δw={py_w_rect-cpp_w_rect}, Δh={py_h_rect-cpp_h_rect}")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)

print(f"""
Expected C++ RNet input: x={cpp_x_rect}, y={cpp_y_rect}, w={cpp_w_rect}, h={cpp_h_rect}
Python RNet input:       x={py_x_rect}, y={py_y_rect}, w={py_w_rect}, h={py_h_rect}

Known C++ RNet input from logs: x=243, y=662, w=499, h=499
Known Python RNet input:        x=221, y=675, w=505, h=505

The difference arises from:
1. int() casting in C++ vs np.floor() + astype(int32) in Python
2. Floating point precision in intermediate calculations
3. Order of operations (C++ casts to int at different stages)
""")

print("\nKEY INSIGHT:")
print("-"*80)
print("""
C++ uses int() cast which truncates towards zero:
  int(407.5) = 407
  int(-407.5) = -407

Python's np.floor() always rounds down:
  np.floor(407.5) = 407
  np.floor(-407.5) = -408

For positive coordinates this matches, but the difference compounds through
multiple casting operations and floating point operations.
""")
