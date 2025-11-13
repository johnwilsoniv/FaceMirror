#!/usr/bin/env python3
"""
Compare C++ vs Python MTCNN intermediate stages to find divergence point.
"""

import cv2
import numpy as np
from cpp_mtcnn_detector import CPPMTCNNDetector

print("="*80)
print("MTCNN STAGE-BY-STAGE COMPARISON")
print("="*80)

# Load test image
img = cv2.imread('cpp_mtcnn_test.jpg')
print(f"\nTest image shape: {img.shape}")

# Run Python MTCNN
print("\n" + "="*80)
print("Running Python MTCNN...")
print("="*80)
detector = CPPMTCNNDetector()
py_bboxes, py_landmarks = detector.detect(img)

print(f"\n{'='*80}")
print(f"PYTHON RESULTS:")
print(f"{'='*80}")
print(f"Detected {len(py_bboxes)} face(s)")
for i, bbox in enumerate(py_bboxes):
    print(f"  Face {i+1}: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}x{bbox[3]:.1f})")

# Compare debug outputs
print(f"\n{'='*80}")
print(f"STAGE-BY-STAGE COMPARISON:")
print(f"{'='*80}")

# ===== RNet Comparison =====
print("\n----- RNet Stage Comparison -----\n")

try:
    with open('/tmp/cpp_rnet_debug.txt', 'r') as f:
        cpp_rnet = f.read()
    with open('/tmp/python_rnet_debug.txt', 'r') as f:
        py_rnet = f.read()

    print("C++ RNet Debug:")
    print(cpp_rnet)

    print("\nPython RNet Debug:")
    print(py_rnet)

    # Parse and compare RNet outputs
    cpp_lines = [l for l in cpp_rnet.split('\n') if 'Input bbox:' in l or 'RNet output:' in l]
    py_lines = [l for l in py_rnet.split('\n') if 'Input bbox:' in l or 'RNet output:' in l]

    print("\n" + "="*80)
    print("RNet Input/Output Comparison:")
    print("="*80)

    for i in range(min(len(cpp_lines)//2, len(py_lines)//2, 3)):
        print(f"\n--- Detection {i} ---")
        print(f"C++:    {cpp_lines[i*2]}")
        print(f"Python: {py_lines[i*2]}")
        print(f"C++:    {cpp_lines[i*2+1]}")
        print(f"Python: {py_lines[i*2+1]}")

except Exception as e:
    print(f"⚠ Error comparing RNet: {e}")

# ===== ONet Comparison =====
print("\n\n----- ONet Stage Comparison -----\n")

try:
    with open('/tmp/cpp_onet_debug.txt', 'r') as f:
        cpp_onet = f.read()

    print("C++ ONet Debug:")
    print(cpp_onet)

    # Python ONet debug should be in the console output
    print("\n(Python ONet debug should be in console output above)")

except Exception as e:
    print(f"⚠ Error reading ONet: {e}")

# ===== Final Bbox Comparison =====
print("\n\n" + "="*80)
print("FINAL BBOX COMPARISON:")
print("="*80)

try:
    with open('/tmp/cpp_mtcnn_final_bbox.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                x, y, w, h = map(float, parts[:4])
                print(f"\nC++ MTCNN:  ({x:.1f}, {y:.1f}, {w:.1f}x{h:.1f})")
except:
    print("\n⚠ No C++ bbox found")

if len(py_bboxes) > 0:
    py_bbox = py_bboxes[0]
    print(f"PyMTCNN:    ({py_bbox[0]:.1f}, {py_bbox[1]:.1f}, {py_bbox[2]:.1f}x{py_bbox[3]:.1f})")

    # Calculate difference
    if 'x' in locals():
        cpp_bbox = np.array([x, y, w, h])
        diff = np.abs(cpp_bbox - py_bbox)
        print(f"Difference: ({diff[0]:.1f}, {diff[1]:.1f}, {diff[2]:.1f}x{diff[3]:.1f})")
        print(f"Max diff:   {diff.max():.1f} pixels")

print(f"\n{'='*80}")
print("SUMMARY:")
print("="*80)
print("""
To investigate further:
1. Check PNet outputs: Are detection proposals similar?
2. Check RNet inputs: Are the input bboxes aligned?
3. Check RNet outputs: Do probabilities match?
4. Check ONet inputs: Are the final proposal boxes aligned?
5. Check NMS thresholds: Are NMS implementations identical?
""")
