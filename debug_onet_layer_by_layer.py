#!/usr/bin/env python3
"""
Layer-by-layer debugging of ONet on Face 3 (best ONet score: 0.55).
Similar to debug_rnet_layer_by_layer.py but for ONet's 14 layers.
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
print("ONET LAYER-BY-LAYER DEBUG - FACE 3")
print("=" * 80)

# Load models
model_dir = os.path.expanduser(
    "~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/"
    "face_detection/mtcnn/convert_to_cpp/"
)

print("\nLoading networks...")
pnet = CPPCNN(os.path.join(model_dir, "PNet.dat"))
rnet = CPPCNN(os.path.join(model_dir, "RNet.dat"))
onet = CPPCNN(os.path.join(model_dir, "ONet.dat"))

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
img_h, img_w = img.shape[:2]
img_float = img.astype(np.float32)

print(f"\nTest image: {img.shape}")

# Run PNet→RNet to get Face 3
print("\n" + "=" * 80)
print("STAGE 1 & 2: Run PNet→RNet to extract Face 3")
print("=" * 80)

min_face_size = 40
factor = 0.709
pnet_threshold = 0.6
rnet_threshold = 0.7

# Build pyramid
m = 12.0 / min_face_size
min_l = min(img_h, img_w) * m

scales = []
scale = m
while min_l >= 12:
    scales.append(scale)
    scale *= factor
    min_l *= factor

total_boxes = []

for i, scale in enumerate(scales):
    hs = int(np.ceil(img_h * scale))
    ws = int(np.ceil(img_w * scale))

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
        keep = nms(boxes, 0.5, 'Union')
        boxes = boxes[keep]
        total_boxes.append(boxes)

total_boxes = np.vstack(total_boxes)
keep = nms(total_boxes, 0.7, 'Union')
total_boxes = total_boxes[keep]

# Run RNet
total_boxes = square_bbox(total_boxes)

rnet_inputs = []
rnet_input_indices = []

for i in range(total_boxes.shape[0]):
    x1 = int(max(0, total_boxes[i, 0]))
    y1 = int(max(0, total_boxes[i, 1]))
    x2 = int(min(img_w, total_boxes[i, 2]))
    y2 = int(min(img_h, total_boxes[i, 3]))

    if x2 <= x1 or y2 <= y1:
        continue

    face = img_float[y1:y2, x1:x2]
    face = cv2.resize(face, (24, 24))
    rnet_inputs.append(preprocess(face))
    rnet_input_indices.append(i)

total_boxes = total_boxes[rnet_input_indices]

# Run RNet
rnet_outputs = []
for face_data in rnet_inputs:
    output = rnet(face_data)[-1]
    rnet_outputs.append(output)

output = np.vstack(rnet_outputs)

# Calculate scores
scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

# Filter by threshold
keep = scores > rnet_threshold
total_boxes = total_boxes[keep]
scores = scores[keep]
reg = output[keep, 2:6]

# NMS
keep = nms(total_boxes, 0.7, 'Union')
total_boxes = total_boxes[keep]
scores = scores[keep]
reg = reg[keep]

# Apply regression
w = total_boxes[:, 2] - total_boxes[:, 0]
h = total_boxes[:, 3] - total_boxes[:, 1]
total_boxes[:, 0] += reg[:, 0] * w
total_boxes[:, 1] += reg[:, 1] * h
total_boxes[:, 2] += reg[:, 2] * w
total_boxes[:, 3] += reg[:, 3] * h
total_boxes[:, 4] = scores

# Prepare ONet inputs
total_boxes = square_bbox(total_boxes)

# Get Face 3 (index 2)
face_idx = 2
x1 = int(max(0, total_boxes[face_idx, 0]))
y1 = int(max(0, total_boxes[face_idx, 1]))
x2 = int(min(img_w, total_boxes[face_idx, 2]))
y2 = int(min(img_h, total_boxes[face_idx, 3]))

face_crop = img_float[y1:y2, x1:x2]
face_crop = cv2.resize(face_crop, (48, 48))
face_data = preprocess(face_crop)

print(f"\nFace 3 extracted:")
print(f"  RNet score: {scores[face_idx]:.4f}")
print(f"  Bbox: ({x1}, {y1}) to ({x2}, {y2})")
print(f"  Crop size before resize: {y2-y1}x{x2-x1}")
print(f"  Input to ONet: {face_data.shape} (3, 48, 48)")

# Run ONet layer-by-layer
print("\n" + "=" * 80)
print("ONET LAYER-BY-LAYER ANALYSIS")
print("=" * 80)

print(f"\nONet has {len(onet.layers)} layers")
print("Expected architecture:")
print("  Conv(3→32, 3x3) → PReLU → MaxPool(3x3, s=2)")
print("  → Conv(32→64, 3x3) → PReLU → MaxPool(3x3, s=2)")
print("  → Conv(64→64, 3x3) → PReLU → MaxPool(2x2, s=2)")
print("  → Conv(64→128, 2x2) → PReLU")
print("  → FC(1152→256) → PReLU → FC(256→16)")

# Manual layer-by-layer execution
activation = face_data

for i, layer in enumerate(onet.layers):
    prev_shape = activation.shape if hasattr(activation, 'shape') else len(activation)

    activation = layer.forward(activation)

    curr_shape = activation.shape if hasattr(activation, 'shape') else len(activation)

    # Get statistics
    if isinstance(activation, np.ndarray):
        act_min = np.min(activation)
        act_max = np.max(activation)
        act_mean = np.mean(activation)
        act_std = np.std(activation)

        print(f"\nLayer {i}: {layer.__class__.__name__}")
        print(f"  Input shape: {prev_shape}")
        print(f"  Output shape: {curr_shape}")
        print(f"  Output stats: min={act_min:.4f}, max={act_max:.4f}, mean={act_mean:.4f}, std={act_std:.4f}")

        # Check for issues
        if np.isnan(activation).any():
            print("  ⚠️  WARNING: NaN values detected!")
        if np.isinf(activation).any():
            print("  ⚠️  WARNING: Inf values detected!")
        if act_std < 1e-6:
            print("  ⚠️  WARNING: Very low variance (possible dead neurons)")

        # Special analysis for final layer
        if i == len(onet.layers) - 1:
            print("\n" + "=" * 80)
            print("FINAL OUTPUT ANALYSIS")
            print("=" * 80)

            logit_not_face = activation[0]
            logit_face = activation[1]
            score = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

            print(f"\nFinal logits:")
            print(f"  Logit not-face: {logit_not_face:.4f}")
            print(f"  Logit face: {logit_face:.4f}")
            print(f"  Difference: {logit_face - logit_not_face:.4f}")
            print(f"\nFinal score: {score:.4f}")
            print(f"  Target threshold: 0.7")
            print(f"  Gap to threshold: {score - 0.7:.4f}")

            print(f"\nBbox regression: [{activation[2]:.4f}, {activation[3]:.4f}, {activation[4]:.4f}, {activation[5]:.4f}]")
            print(f"Landmarks (5 points): {activation[6:16]}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print("\nKey observations to look for:")
print("1. Are any layers producing NaN or Inf values?")
print("2. Is any layer killing activations (very low variance)?")
print("3. Where do the activations diverge from expected values?")
print("4. Are the logit outputs reasonable?")
