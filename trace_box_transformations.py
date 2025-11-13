#!/usr/bin/env python3
"""
Trace the problematic box through each transformation:
1. Raw PNet output (before any transformations)
2. After PNet regression
3. After squaring

This will show us EXACTLY which operation introduces the bad framing.
"""

import cv2
import numpy as np
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2

print("=" * 80)
print("TRACING BOX TRANSFORMATIONS")
print("=" * 80)

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
img_h, img_w = img.shape[:2]
img_float = img.astype(np.float32)

print(f"\nTest image: {img_w}×{img_h}")
print(f"C++ Gold Standard: x=331, y=753, w=368, h=423")

# Create detector
detector = PurePythonMTCNN_V2()

# Run PNet to get raw boxes
min_face_size = 40
factor = 0.709
m = 12.0 / min_face_size
min_l = min(img_h, img_w) * m

scales = []
scale = m
while min_l >= 12:
    scales.append(scale)
    scale *= factor
    min_l *= factor

print(f"\nImage pyramid: {len(scales)} scales")

# Run PNet on all scales
total_boxes_raw = []
for i, scale in enumerate(scales):
    hs = int(np.ceil(img_h * scale))
    ws = int(np.ceil(img_w * scale))

    img_scaled = cv2.resize(img_float, (ws, hs))
    img_data = detector._preprocess(img_scaled)

    output = detector._run_pnet(img_data)
    output = output[0].transpose(1, 2, 0)

    logit_not_face = output[:, :, 0]
    logit_face = output[:, :, 1]
    prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

    score_map = np.stack([1.0 - prob_face, prob_face], axis=2)
    reg_map = output[:, :, 2:6]

    boxes = detector._generate_bboxes(score_map, reg_map, scale, 0.6)

    if boxes.shape[0] > 0:
        keep = detector._nms(boxes, 0.5, 'Union')
        boxes = boxes[keep]
        total_boxes_raw.append(boxes)

total_boxes_raw = np.vstack(total_boxes_raw)
print(f"PNet raw: {total_boxes_raw.shape[0]} boxes before cross-scale NMS")

# Cross-scale NMS
keep = detector._nms(total_boxes_raw, 0.7, 'Union')
total_boxes_raw = total_boxes_raw[keep]
print(f"PNet raw: {total_boxes_raw.shape[0]} boxes after cross-scale NMS")

# Find large boxes (the problematic 452x451 box from Phase 1)
gold_center_x = 331 + 368/2  # 515
gold_center_y = 753 + 423/2  # 964.5

print("\nSearching for large boxes (>400px) near face region:")
large_boxes = []

for i in range(total_boxes_raw.shape[0]):
    x1, y1, x2, y2 = total_boxes_raw[i, 0:4]
    w, h = x2 - x1, y2 - y1
    size = max(w, h)

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    dist = np.sqrt((center_x - gold_center_x)**2 + (center_y - gold_center_y)**2)

    if size > 400:  # Look for large boxes
        large_boxes.append((i, size, dist, x1, y1, w, h))
        print(f"  Box #{i}: {w:.0f}×{h:.0f}px at ({x1:.0f}, {y1:.0f}), distance from gold: {dist:.0f}px")

if not large_boxes:
    print("  No boxes >400px found! Trying >300px...")
    for i in range(total_boxes_raw.shape[0]):
        x1, y1, x2, y2 = total_boxes_raw[i, 0:4]
        w, h = x2 - x1, y2 - y1
        size = max(w, h)

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        dist = np.sqrt((center_x - gold_center_x)**2 + (center_y - gold_center_y)**2)

        if size > 300:
            large_boxes.append((i, size, dist, x1, y1, w, h))
            print(f"  Box #{i}: {w:.0f}×{h:.0f}px at ({x1:.0f}, {y1:.0f}), distance from gold: {dist:.0f}px")

# Pick the box closest to gold standard (not just largest)
large_boxes.sort(key=lambda x: (x[2], -x[1]))  # Sort by distance (asc), then size (desc)
best_idx = large_boxes[0][0] if large_boxes else 0

print(f"\nSelected box #{best_idx} for tracing (closest to gold standard)")

# Extract this box at each stage
box_raw = total_boxes_raw[best_idx].copy()

print("\n" + "=" * 80)
print("STAGE 1: RAW PNET OUTPUT (before regression)")
print("=" * 80)

x1, y1, x2, y2 = box_raw[0:4]
w, h = x2 - x1, y2 - y1
print(f"Position: ({x1:.1f}, {y1:.1f})")
print(f"Size: {w:.1f}×{h:.1f}")
print(f"Regression values: {box_raw[5:9]}")

# Apply regression
boxes_after_regression = np.array([box_raw])
boxes_after_regression = detector._apply_bbox_regression(boxes_after_regression)
box_after_regression = boxes_after_regression[0]

print("\n" + "=" * 80)
print("STAGE 2: AFTER PNET REGRESSION")
print("=" * 80)

x1, y1, x2, y2 = box_after_regression[0:4]
w, h = x2 - x1, y2 - y1
print(f"Position: ({x1:.1f}, {y1:.1f})")
print(f"Size: {w:.1f}×{h:.1f}")

# Apply squaring
boxes_after_squaring = detector._square_bbox(boxes_after_regression)
box_after_squaring = boxes_after_squaring[0]

print("\n" + "=" * 80)
print("STAGE 3: AFTER SQUARING")
print("=" * 80)

x1, y1, x2, y2 = box_after_squaring[0:4]
w, h = x2 - x1, y2 - y1
print(f"Position: ({x1:.1f}, {y1:.1f})")
print(f"Size: {w:.1f}×{h:.1f}")

# Visualize all stages
img_vis = img.copy()

# Draw gold standard (red)
cv2.rectangle(img_vis, (331, 753), (699, 1176), (0, 0, 255), 3)
cv2.putText(img_vis, 'GOLD', (331, 740), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# Draw raw box (blue)
x1, y1, x2, y2 = box_raw[0:4].astype(int)
cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
cv2.putText(img_vis, 'STAGE 1: Raw PNet', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Draw after regression (magenta)
x1, y1, x2, y2 = box_after_regression[0:4].astype(int)
cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
cv2.putText(img_vis, 'STAGE 2: After regression', (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

# Draw after squaring (green)
x1, y1, x2, y2 = box_after_squaring[0:4].astype(int)
cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.putText(img_vis, 'STAGE 3: After squaring', (x1, y1-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite('box_transformation_stages.jpg', img_vis)
print(f"\nSaved visualization: box_transformation_stages.jpg")

# Calculate deltas at each stage
print("\n" + "=" * 80)
print("DELTA ANALYSIS")
print("=" * 80)

# Stage 1 → 2 (regression)
raw_x1, raw_y1, raw_x2, raw_y2 = box_raw[0:4]
reg_x1, reg_y1, reg_x2, reg_y2 = box_after_regression[0:4]

print("\nRegression changes:")
print(f"  x1: {raw_x1:.1f} → {reg_x1:.1f} (Δ{reg_x1-raw_x1:+.1f})")
print(f"  y1: {raw_y1:.1f} → {reg_y1:.1f} (Δ{reg_y1-raw_y1:+.1f})")
print(f"  x2: {raw_x2:.1f} → {reg_x2:.1f} (Δ{reg_x2-raw_x2:+.1f})")
print(f"  y2: {raw_y2:.1f} → {reg_y2:.1f} (Δ{reg_y2-raw_y2:+.1f})")
print(f"  width: {raw_x2-raw_x1:.1f} → {reg_x2-reg_x1:.1f} (Δ{(reg_x2-reg_x1)-(raw_x2-raw_x1):+.1f})")
print(f"  height: {raw_y2-raw_y1:.1f} → {reg_y2-reg_y1:.1f} (Δ{(reg_y2-reg_y1)-(raw_y2-raw_y1):+.1f})")

# Stage 2 → 3 (squaring)
sq_x1, sq_y1, sq_x2, sq_y2 = box_after_squaring[0:4]

print("\nSquaring changes:")
print(f"  x1: {reg_x1:.1f} → {sq_x1:.1f} (Δ{sq_x1-reg_x1:+.1f})")
print(f"  y1: {reg_y1:.1f} → {sq_y1:.1f} (Δ{sq_y1-reg_y1:+.1f})")
print(f"  x2: {reg_x2:.1f} → {sq_x2:.1f} (Δ{sq_x2-reg_x2:+.1f})")
print(f"  y2: {reg_y2:.1f} → {sq_y2:.1f} (Δ{sq_y2-reg_y2:+.1f})")
print(f"  width: {reg_x2-reg_x1:.1f} → {sq_x2-sq_x1:.1f} (Δ{(sq_x2-sq_x1)-(reg_x2-reg_x1):+.1f})")
print(f"  height: {reg_y2-reg_y1:.1f} → {sq_y2-sq_y1:.1f} (Δ{(sq_y2-sq_y1)-(reg_y2-reg_y1):+.1f})")

# Compare to gold
print("\nComparison to C++ gold standard:")
print(f"  Final box: ({sq_x1:.0f}, {sq_y1:.0f}) {sq_x2-sq_x1:.0f}×{sq_y2-sq_y1:.0f}")
print(f"  Gold:      (331, 753) 368×423")
print(f"  Offset:    x={sq_x1-331:.0f}px, y={sq_y1-753:.0f}px")
print(f"  Size diff: w={sq_x2-sq_x1-368:.0f}px, h={sq_y2-sq_y1-423:.0f}px")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print("""
Check the visualization (box_transformation_stages.jpg) to see:
- Red box: C++ gold standard (correct)
- Blue box: Raw PNet output before any transforms
- Magenta box: After PNet regression
- Green box: Final after squaring (this is what gets rejected)

Which transformation introduces the bad framing?
""")
