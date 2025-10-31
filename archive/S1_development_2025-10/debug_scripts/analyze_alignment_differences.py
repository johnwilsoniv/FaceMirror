#!/usr/bin/env python3
"""
Detailed analysis of alignment differences between Python and C++ OpenFace
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from openface22_face_aligner import OpenFace22FaceAligner

# Initialize aligner
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")

# Load validation CSV
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
row = df.iloc[0]

# Extract landmarks and pose
x_cols = [f'x_{i}' for i in range(68)]
y_cols = [f'y_{i}' for i in range(68)]
x = row[x_cols].values.astype(np.float32)
y = row[y_cols].values.astype(np.float32)
landmarks_68 = np.stack([x, y], axis=1)
pose_tx = row['p_tx']
pose_ty = row['p_ty']

# Load frame
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read frame")
    exit(1)

# Align with Python
python_aligned = aligner.align_face(frame, landmarks_68, pose_tx, pose_ty)

# Load C++ aligned
# Search for existing aligned face in validation output
cpp_aligned_paths = list(Path("alignment_validation_output").glob("frame_0001_comparison.png"))
if cpp_aligned_paths:
    comparison = cv2.imread(str(cpp_aligned_paths[0]))
    # Extract C++ aligned (center panel of comparison image)
    cpp_aligned = comparison[:, 112:224]
else:
    print("Error: No C++ aligned face found in validation_output")
    exit(1)

print("=" * 70)
print("Detailed Alignment Difference Analysis")
print("=" * 70)

# 1. Overall statistics
diff = cv2.absdiff(cpp_aligned, python_aligned)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

print("\n[1] Overall Pixel Differences:")
print(f"  Mean difference: {diff_gray.mean():.2f}")
print(f"  Median difference: {np.median(diff_gray):.2f}")
print(f"  Std dev: {diff_gray.std():.2f}")
print(f"  Max difference: {diff_gray.max()}")
print(f"  Pixels with diff > 10: {(diff_gray > 10).sum()} ({(diff_gray > 10).sum() / diff_gray.size * 100:.1f}%)")
print(f"  Pixels with diff > 50: {(diff_gray > 50).sum()} ({(diff_gray > 50).sum() / diff_gray.size * 100:.1f}%)")
print(f"  Pixels with diff > 100: {(diff_gray > 100).sum()} ({(diff_gray > 100).sum() / diff_gray.size * 100:.1f}%)")

# 2. Spatial distribution of errors
print("\n[2] Spatial Distribution:")
# Divide into 4 quadrants
h, w = diff_gray.shape
q1 = diff_gray[:h//2, :w//2]  # Top-left
q2 = diff_gray[:h//2, w//2:]  # Top-right
q3 = diff_gray[h//2:, :w//2]  # Bottom-left
q4 = diff_gray[h//2:, w//2:]  # Bottom-right

print(f"  Top-left quadrant mean diff: {q1.mean():.2f}")
print(f"  Top-right quadrant mean diff: {q2.mean():.2f}")
print(f"  Bottom-left quadrant mean diff: {q3.mean():.2f}")
print(f"  Bottom-right quadrant mean diff: {q4.mean():.2f}")

# 3. Check for systematic shift
print("\n[3] Checking for Systematic Translation:")
# Cross-correlation to detect shifts
cpp_gray = cv2.cvtColor(cpp_aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
py_gray = cv2.cvtColor(python_aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)

# Try small shifts
best_corr = 0
best_shift = (0, 0)
for dy in range(-3, 4):
    for dx in range(-3, 4):
        if dy == 0 and dx == 0:
            shifted = py_gray
        else:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(py_gray, M, (w, h))

        corr = np.corrcoef(cpp_gray.flatten(), shifted.flatten())[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_shift = (dx, dy)

print(f"  Best correlation: {best_corr:.6f} at shift {best_shift}")
if best_shift != (0, 0):
    print(f"  ⚠ Python alignment is shifted by {best_shift} pixels!")
else:
    print(f"  ✓ No significant translation detected")

# 4. Check background vs foreground differences
print("\n[4] Background vs Foreground Differences:")
# Use brightness threshold to separate face from background
cpp_bright = cpp_gray > 30
py_bright = py_gray > 30

# Face region (bright pixels)
face_mask = cpp_bright & py_bright
face_diff = diff_gray[face_mask]
if len(face_diff) > 0:
    print(f"  Face region mean diff: {face_diff.mean():.2f}")
    print(f"  Face region max diff: {face_diff.max()}")

# Background region (dark pixels)
bg_mask = ~cpp_bright & ~py_bright
bg_diff = diff_gray[bg_mask]
if len(bg_diff) > 0:
    print(f"  Background region mean diff: {bg_diff.mean():.2f}")
    print(f"  Background region max diff: {bg_diff.max()}")

# Edge region (transition pixels)
edge_mask = cpp_bright != py_bright
edge_diff = diff_gray[edge_mask]
if len(edge_diff) > 0:
    print(f"  Edge region mean diff: {edge_diff.mean():.2f}")
    print(f"  Edge pixels: {edge_mask.sum()} ({edge_mask.sum() / edge_mask.size * 100:.1f}%)")

# 5. Visualize difference heatmap
print("\n[5] Creating visualization...")
diff_colored = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)

# Create comprehensive comparison
vis = np.zeros((112 * 2, 112 * 3, 3), dtype=np.uint8)
vis[:112, :112] = cpp_aligned
vis[:112, 112:224] = python_aligned
vis[:112, 224:336] = diff_colored
vis[112:, :112] = cv2.cvtColor(cpp_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
vis[112:, 112:224] = cv2.cvtColor(py_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
# Histogram
hist_img = np.zeros((112, 112, 3), dtype=np.uint8)
hist = cv2.calcHist([diff_gray], [0], None, [112], [0, 256])
hist = hist / hist.max() * 112
for i in range(112):
    cv2.line(hist_img, (i, 112), (i, 112 - int(hist[i])), (255, 255, 255), 1)
vis[112:, 224:336] = hist_img

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(vis, "C++", (5, 15), font, 0.4, (255, 255, 255), 1)
cv2.putText(vis, "Python", (117, 15), font, 0.4, (255, 255, 255), 1)
cv2.putText(vis, "Diff (JET)", (229, 15), font, 0.4, (255, 255, 255), 1)
cv2.putText(vis, "C++ Gray", (5, 127), font, 0.4, (255, 255, 255), 1)
cv2.putText(vis, "Py Gray", (117, 127), font, 0.4, (255, 255, 255), 1)
cv2.putText(vis, "Diff Hist", (229, 127), font, 0.4, (255, 255, 255), 1)

cv2.imwrite("alignment_detailed_analysis.png", vis)
print("✓ Saved alignment_detailed_analysis.png")

# Large version
large = cv2.resize(vis, (112 * 6, 112 * 4), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("alignment_detailed_analysis_large.png", large)
print("✓ Saved alignment_detailed_analysis_large.png")

print("\n" + "=" * 70)
print("Analysis complete")
print("=" * 70)
