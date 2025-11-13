#!/usr/bin/env python3
"""
Trace bbox generation through the pyramid to see where large-scale detections are lost.
"""

import cv2
import numpy as np
from cpp_cnn_loader import CPPCNN
import os

def preprocess(img):
    img_norm = (img.astype(np.float32) - 127.5) * 0.0078125
    img_chw = np.transpose(img_norm, (2, 0, 1))
    return img_chw

def generate_bboxes(score_map, reg_map, scale, threshold):
    """Generate bounding boxes from PNet output."""
    stride = 2
    cellsize = 12

    t_index = np.where(score_map[:, :, 1] > threshold)

    if t_index[0].size == 0:
        return np.array([])

    dx1, dy1, dx2, dy2 = [reg_map[t_index[0], t_index[1], i] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2])
    score = score_map[t_index[0], t_index[1], 1]
    boundingbox = np.vstack([
        np.round((stride * t_index[1] + 1) / scale),
        np.round((stride * t_index[0] + 1) / scale),
        np.round((stride * t_index[1] + 1 + cellsize) / scale),
        np.round((stride * t_index[0] + 1 + cellsize) / scale),
        score,
        reg
    ])

    return boundingbox.T

def nms(boxes, threshold, method):
    """Non-Maximum Suppression."""
    if boxes.shape[0] == 0:
        return np.array([])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_s = np.argsort(s)

    pick = []
    while sorted_s.shape[0] > 0:
        i = sorted_s[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[sorted_s[:-1]])
        yy1 = np.maximum(y1[i], y1[sorted_s[:-1]])
        xx2 = np.minimum(x2[i], x2[sorted_s[:-1]])
        yy2 = np.minimum(y2[i], y2[sorted_s[:-1]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h

        if method == 'Min':
            o = inter / np.minimum(area[i], area[sorted_s[:-1]])
        else:
            o = inter / (area[i] + area[sorted_s[:-1]] - inter)

        sorted_s = sorted_s[np.where(o <= threshold)[0]]

    return pick

print("=" * 80)
print("TRACING BBOX GENERATION THROUGH PYRAMID")
print("=" * 80)

# Load PNet
model_dir = os.path.expanduser(
    "~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/"
    "face_detection/mtcnn/convert_to_cpp/"
)
pnet = CPPCNN(os.path.join(model_dir, "PNet.dat"))

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
img_h, img_w = img.shape[:2]
img_float = img.astype(np.float32)

print(f"\nTest image: {img_w}×{img_h}")
print(f"C++ Gold Standard: x=331.6, y=753.5, w=367.9, h=422.8")

# Build pyramid
min_face_size = 40
factor = 0.709
pnet_threshold = 0.6

m = 12.0 / min_face_size
min_l = min(img_h, img_w) * m

scales = []
scale = m
while min_l >= 12:
    scales.append(scale)
    scale *= factor
    min_l *= factor

print(f"\nImage pyramid: {len(scales)} scales")

total_boxes_all = []

for i, scale in enumerate(scales):
    hs = int(np.ceil(img_h * scale))
    ws = int(np.ceil(img_w * scale))

    face_size = min_face_size / scale

    img_scaled = cv2.resize(img_float, (ws, hs))
    img_data = preprocess(img_scaled)

    # Run PNet
    output = pnet(img_data)
    output = output[-1]
    output = output[np.newaxis, :, :, :]
    output = output[0].transpose(1, 2, 0)

    logit_not_face = output[:, :, 0]
    logit_face = output[:, :, 1]
    prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

    score_map = np.stack([1.0 - prob_face, prob_face], axis=2)
    reg_map = output[:, :, 2:6]

    boxes = generate_bboxes(score_map, reg_map, scale, pnet_threshold)

    if boxes.shape[0] > 0:
        # Calculate box sizes
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

        print(f"\n--- Scale {i}: m={scale:.4f}, detects faces ≥{face_size:.0f}px ---")
        print(f"  Scaled image: {ws}×{hs}")
        print(f"  Prob map: {prob_face.shape}")
        print(f"  Max prob: {prob_face.max():.4f}")
        print(f"  Pixels > {pnet_threshold}: {(prob_face > pnet_threshold).sum()}")
        print(f"  Boxes generated: {boxes.shape[0]}")
        print(f"  Box sizes: {widths.min():.0f}-{widths.max():.0f}px (w), {heights.min():.0f}-{heights.max():.0f}px (h)")

        # Show top 5 boxes by score
        sorted_idx = np.argsort(boxes[:, 4])[::-1][:5]
        print(f"  Top 5 boxes:")
        for j, idx in enumerate(sorted_idx):
            x1, y1, x2, y2, score = boxes[idx, :5]
            w = x2 - x1
            h = y2 - y1
            print(f"    #{j+1}: ({x1:.0f}, {y1:.0f}) {w:.0f}×{h:.0f}, score={score:.4f}")

        # Apply per-scale NMS
        keep = nms(boxes, 0.5, 'Union')
        boxes = boxes[keep]
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

        print(f"  After per-scale NMS (0.5): {boxes.shape[0]} boxes")
        print(f"  Box sizes after NMS: {widths.min():.0f}-{widths.max():.0f}px (w), {heights.min():.0f}-{heights.max():.0f}px (h)")

        total_boxes_all.append(boxes)

# Merge all scales
print("\n" + "=" * 80)
print("MERGING ALL SCALES")
print("=" * 80)

if len(total_boxes_all) == 0:
    print("\n⚠️  No boxes from any scale!")
else:
    total_boxes = np.vstack(total_boxes_all)
    widths = total_boxes[:, 2] - total_boxes[:, 0]
    heights = total_boxes[:, 3] - total_boxes[:, 1]

    print(f"\nTotal boxes before cross-scale NMS: {total_boxes.shape[0]}")
    print(f"  Size range: {widths.min():.0f}-{widths.max():.0f}px (w), {heights.min():.0f}-{heights.max():.0f}px (h)")

    # Show size distribution
    size_bins = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 1000)]
    print(f"\n  Size distribution:")
    for min_size, max_size in size_bins:
        count = ((widths >= min_size) & (widths < max_size)).sum()
        print(f"    {min_size}-{max_size}px: {count} boxes")

    # Cross-scale NMS
    keep = nms(total_boxes, 0.7, 'Union')
    total_boxes = total_boxes[keep]
    widths = total_boxes[:, 2] - total_boxes[:, 0]
    heights = total_boxes[:, 3] - total_boxes[:, 1]

    print(f"\nAfter cross-scale NMS (0.7): {total_boxes.shape[0]} boxes")
    print(f"  Size range: {widths.min():.0f}-{widths.max():.0f}px (w), {heights.min():.0f}-{heights.max():.0f}px (h)")

    # Show size distribution after NMS
    print(f"\n  Size distribution after NMS:")
    for min_size, max_size in size_bins:
        count = ((widths >= min_size) & (widths < max_size)).sum()
        print(f"    {min_size}-{max_size}px: {count} boxes")

    # Show boxes near expected face location
    cpp_x, cpp_y, cpp_w, cpp_h = 331.6, 753.5, 367.9, 422.8
    cpp_center_x = cpp_x + cpp_w / 2
    cpp_center_y = cpp_y + cpp_h / 2

    print(f"\n  Boxes near expected face region ({cpp_center_x:.0f}, {cpp_center_y:.0f}):")
    for i in range(total_boxes.shape[0]):
        x1, y1, x2, y2, score = total_boxes[i, :5]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        dist = np.sqrt((center_x - cpp_center_x)**2 + (center_y - cpp_center_y)**2)

        if dist < 300:  # Within 300px of expected face center
            w = x2 - x1
            h = y2 - y1
            print(f"    Box: ({x1:.0f}, {y1:.0f}) {w:.0f}×{h:.0f}, score={score:.4f}, dist={dist:.0f}px")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print("\nKey Questions:")
print("1. Are large boxes (300-400px) being generated at Scale 2-3?")
print("2. Are they being filtered out by per-scale NMS?")
print("3. Are they being filtered out by cross-scale NMS?")
print("4. Or are only small boxes being generated in the first place?")
