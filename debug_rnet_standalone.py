#!/usr/bin/env python3
"""
Debug Pure Python RNet on real face crops - standalone version.
Manually run PNet stage and capture RNet inputs.
"""

import cv2
import numpy as np
from cpp_cnn_loader import CPPCNN
import os

def preprocess(img: np.ndarray) -> np.ndarray:
    """Preprocess image for CNN."""
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

def square_bbox(bboxes):
    """Convert bounding boxes to squares."""
    square_bboxes = bboxes.copy()
    h = bboxes[:, 3] - bboxes[:, 1]
    w = bboxes[:, 2] - bboxes[:, 0]
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = bboxes[:, 0] + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = bboxes[:, 1] + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side
    return square_bboxes


print("=" * 80)
print("RNET DEBUG ON REAL FACE CROPS - STANDALONE")
print("=" * 80)

# Load models
model_dir = os.path.expanduser(
    "~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/"
    "face_detection/mtcnn/convert_to_cpp/"
)

print("\nLoading networks...")
pnet = CPPCNN(os.path.join(model_dir, "PNet.dat"))
rnet = CPPCNN(os.path.join(model_dir, "RNet.dat"))

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
img_h, img_w = img.shape[:2]
img_float = img.astype(np.float32)

print(f"\nTest image: {img.shape}")

# Run PNet stage to get candidate boxes
print("\n" + "=" * 80)
print("STAGE 1: Run PNet to get candidate faces")
print("=" * 80)

min_face_size = 40
factor = 0.709
pnet_threshold = 0.6

# Build pyramid
m = 12.0 / min_face_size
min_l = min(img_h, img_w) * m

scales = []
scale = m
while min_l >= 12:
    scales.append(scale)
    scale *= factor
    min_l *= factor

print(f"\nImage pyramid: {len(scales)} scales")

total_boxes = []

for i, scale in enumerate(scales):
    hs = int(np.ceil(img_h * scale))
    ws = int(np.ceil(img_w * scale))

    img_scaled = cv2.resize(img_float, (ws, hs))
    img_data = preprocess(img_scaled)

    # Run PNet
    output = pnet(img_data)
    output = output[-1]  # Get final output
    output = output[np.newaxis, :, :, :]  # Add batch dimension
    output = output[0].transpose(1, 2, 0)  # (H, W, 6)

    logit_not_face = output[:, :, 0]
    logit_face = output[:, :, 1]
    prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

    score_map = np.stack([1.0 - prob_face, prob_face], axis=2)
    reg_map = output[:, :, 2:6]

    boxes = generate_bboxes(score_map, reg_map, scale, pnet_threshold)

    if boxes.shape[0] > 0:
        keep = nms(boxes, 0.5, 'Union')
        boxes = boxes[keep]
        total_boxes.append(boxes)

if len(total_boxes) == 0:
    print("\n⚠️  PNet found no faces!")
    exit(1)

total_boxes = np.vstack(total_boxes)
keep = nms(total_boxes, 0.7, 'Union')
total_boxes = total_boxes[keep]

print(f"\nPNet detected {len(total_boxes)} candidate faces")

# Prepare RNet inputs
print("\n" + "=" * 80)
print("STAGE 2: Extract and prepare RNet inputs")
print("=" * 80)

total_boxes = square_bbox(total_boxes)

rnet_inputs = []
rnet_input_indices = []

for i in range(total_boxes.shape[0]):
    # Simplified: clip bbox to image bounds and resize
    x1 = int(max(0, total_boxes[i, 0]))
    y1 = int(max(0, total_boxes[i, 1]))
    x2 = int(min(img_w, total_boxes[i, 2]))
    y2 = int(min(img_h, total_boxes[i, 3]))

    if x2 <= x1 or y2 <= y1:
        continue

    # Crop face region
    face = img_float[y1:y2, x1:x2]

    # Resize to 24x24
    face = cv2.resize(face, (24, 24))
    rnet_inputs.append(preprocess(face))
    rnet_input_indices.append(i)

total_boxes = total_boxes[rnet_input_indices]

print(f"Prepared {len(rnet_inputs)} face crops for RNet (24x24 each)")

# Run RNet and analyze outputs
print("\n" + "=" * 80)
print("STAGE 3: Run RNet and analyze outputs")
print("=" * 80)

rnet_outputs = []
for face_data in rnet_inputs:
    output = rnet(face_data)
    rnet_outputs.append(output[-1])

# Analyze each face
for i, (face_crop, rnet_output) in enumerate(zip(rnet_inputs, rnet_outputs)):
    print(f"\n--- Face {i+1}/{len(rnet_inputs)} ---")

    # Input stats
    print(f"Input shape: {face_crop.shape}")
    print(f"Input range: [{face_crop.min():.4f}, {face_crop.max():.4f}]")
    print(f"Input mean: {face_crop.mean():.4f}")
    print(f"Input std: {face_crop.std():.4f}")

    # Output stats
    logit_not_face = rnet_output[0]
    logit_face = rnet_output[1]
    score = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

    print(f"\nRNet output:")
    print(f"  Logit not-face: {logit_not_face:.4f}")
    print(f"  Logit face: {logit_face:.4f}")
    print(f"  Score: {score:.4f} {'✓ PASS' if score > 0.7 else '✗ FAIL' if score > 0.6 else '✗✗ FAIL'}")
    print(f"  Regression: [{rnet_output[2]:.4f}, {rnet_output[3]:.4f}, {rnet_output[4]:.4f}, {rnet_output[5]:.4f}]")

    # Save visualization (first 5 faces only)
    if i < 5:
        face_vis = face_crop.copy()
        face_vis = (face_vis / 0.0078125 + 127.5).clip(0, 255).astype(np.uint8)
        face_vis = np.transpose(face_vis, (1, 2, 0))
        face_vis = cv2.cvtColor(face_vis, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f'debug_rnet_face_{i+1}_score{score:.3f}.jpg', face_vis)
        print(f"  💾 Saved: debug_rnet_face_{i+1}_score{score:.3f}.jpg")

# Summary
scores = []
for output in rnet_outputs:
    score = 1.0 / (1.0 + np.exp(output[0] - output[1]))
    scores.append(score)

best_score = max(scores)
best_idx = np.argmax(scores)

print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

print(f"\nRNet Statistics:")
print(f"  Total faces tested: {len(scores)}")
print(f"  Best score: {best_score:.4f}")
print(f"  Scores > 0.7 (official threshold): {sum(s > 0.7 for s in scores)}")
print(f"  Scores > 0.6 (lowered threshold): {sum(s > 0.6 for s in scores)}")

if best_score < 0.7:
    gap = 0.7 - best_score
    print(f"\n⚠️  Best RNet score ({best_score:.4f}) is {gap:.4f} below official threshold (0.7)")
    print(f"This explains why Pure Python MTCNN fails with official thresholds.")

print(f"\nOfficial MATLAB/C++ MTCNN thresholds: [0.6, 0.7, 0.7]")
print(f"Pure Python must use: [0.6, 0.6, 0.6] to compensate")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

print("\n1. Run layer-by-layer analysis on the best-scoring face")
print("2. Compare Pure Python RNet conv/fc weights to C++ to verify loading")
print("3. Test if preprocessing matches C++ exactly")
