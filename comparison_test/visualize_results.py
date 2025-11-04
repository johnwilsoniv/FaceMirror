#!/usr/bin/env python3
"""Create visualizations of the comparison results."""

import numpy as np
import cv2
from pathlib import Path

print("="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Load data
TEST_IMAGE = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/frames/IMG_8401.jpg"
RESULTS_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results")

image = cv2.imread(TEST_IMAGE)
h, w = image.shape[:2]

# Load results
cpp_landmarks = np.load(RESULTS_DIR / "cpp_landmarks.npy")
pyfacelm_data = np.load(RESULTS_DIR / "pyfacelm_test_results.npz")
pyfaceau_data = np.load(RESULTS_DIR / "pyfaceau_clnf_test_results.npz")

pyfacelm_landmarks = pyfacelm_data['pyfacelm_landmarks']
pyfaceau_landmarks = pyfaceau_data['pyfaceau_landmarks']

print(f"Loaded landmarks:")
print(f"  C++ OpenFace: {len(cpp_landmarks)} points")
print(f"  PyfaceLM: {len(pyfacelm_landmarks)} points")
print(f"  pyfaceau: {len(pyfaceau_landmarks)} points")

# Create comparison visualization
vis_width = w * 3
vis = np.zeros((h, vis_width, 3), dtype=np.uint8)

# Column 1: C++ OpenFace (GREEN - ground truth)
vis[:, :w] = image.copy()
for i, (x, y) in enumerate(cpp_landmarks):
    cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
    if i % 10 == 0:  # Label every 10th landmark
        cv2.putText(vis, str(i), (int(x)+5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
cv2.putText(vis, "C++ OpenFace (Ground Truth)", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
cv2.putText(vis, "Confidence: 0.98", (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Column 2: PyfaceLM (BLUE)
vis[:, w:2*w] = image.copy()
for i, (x, y) in enumerate(pyfacelm_landmarks):
    cv2.circle(vis, (int(x + w), int(y)), 3, (255, 0, 0), -1)
    if i % 10 == 0:
        cv2.putText(vis, str(i), (int(x+w)+5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
cv2.putText(vis, "PyfaceLM", (w + 10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
cv2.putText(vis, f"Mean Error: {pyfacelm_data['mean_error']:.1f}px", (w + 10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
cv2.putText(vis, f"Converged: {pyfacelm_data['converged']}", (w + 10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Column 3: pyfaceau (RED)
vis[:, 2*w:] = image.copy()
for i, (x, y) in enumerate(pyfaceau_landmarks):
    cv2.circle(vis, (int(x + 2*w), int(y)), 3, (0, 0, 255), -1)
    if i % 10 == 0:
        cv2.putText(vis, str(i), (int(x+2*w)+5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
cv2.putText(vis, "pyfaceau CLNF", (2*w + 10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
cv2.putText(vis, f"Mean Error: {pyfaceau_data['mean_error']:.1f}px", (2*w + 10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(vis, f"Converged: {pyfaceau_data['converged']}", (2*w + 10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Save visualization
vis_file = RESULTS_DIR / "IMG_8401_comparison.jpg"
cv2.imwrite(str(vis_file), vis)
print(f"\n✓ Saved 3-way comparison: {vis_file}")

# Create error heatmap overlay for PyfaceLM
error_vis = image.copy()
errors = pyfacelm_data['errors']
max_error = np.max(errors)

for i, ((x_cpp, y_cpp), (x_py, y_py)) in enumerate(zip(cpp_landmarks, pyfacelm_landmarks)):
    error = errors[i]
    # Color from green (low error) to red (high error)
    color_ratio = min(error / 100.0, 1.0)  # Normalize to 100px max for color scale
    color = (0, int(255 * (1 - color_ratio)), int(255 * color_ratio))

    # Draw line from C++ to Python landmark
    cv2.line(error_vis, (int(x_cpp), int(y_cpp)), (int(x_py), int(y_py)), color, 2)
    cv2.circle(error_vis, (int(x_cpp), int(y_cpp)), 3, (0, 255, 0), -1)  # C++ in green
    cv2.circle(error_vis, (int(x_py), int(y_py)), 3, (0, 0, 255), -1)    # Python in red

cv2.putText(error_vis, "PyfaceLM Error Visualization", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
cv2.putText(error_vis, f"Green=C++ OpenFace, Red=PyfaceLM", (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(error_vis, f"Mean Error: {pyfacelm_data['mean_error']:.1f}px", (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

error_vis_file = RESULTS_DIR / "IMG_8401_error_heatmap.jpg"
cv2.imwrite(str(error_vis_file), error_vis)
print(f"✓ Saved error heatmap: {error_vis_file}")

# Print top 10 worst errors
print(f"\nTop 10 worst landmark errors (PyfaceLM):")
worst_indices = np.argsort(errors)[-10:][::-1]
for idx in worst_indices:
    print(f"  Landmark {idx:2d}: {errors[idx]:7.2f} pixels")

print("\n" + "="*80)
print("VISUALIZATIONS COMPLETE")
print("="*80)
print(f"\nView results:")
print(f"  {vis_file}")
print(f"  {error_vis_file}")
