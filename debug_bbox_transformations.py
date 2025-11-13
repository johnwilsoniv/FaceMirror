#!/usr/bin/env python3
"""
Debug bbox transformations through PNet→RNet→ONet pipeline.
Track how bboxes change at each stage to identify where they go wrong.
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

def print_bbox(bbox, name):
    """Print bbox info."""
    x1, y1, x2, y2 = bbox[:4]
    w = x2 - x1
    h = y2 - y1
    print(f"  {name}: ({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f}), size: {w:.1f}×{h:.1f}")


print("=" * 80)
print("BBOX TRANSFORMATION DEBUGGING")
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

print(f"Test image: {img.shape}")
print(f"C++ Gold Standard: x=331.6, y=753.5, w=367.9, h=422.8")

# Run PNet
print("\n" + "=" * 80)
print("STAGE 1: PNet")
print("=" * 80)

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

total_boxes = []

for i, scale in enumerate(scales):
    hs = int(np.ceil(img_h * scale))
    ws = int(np.ceil(img_w * scale))

    img_scaled = cv2.resize(img_float, (ws, hs))
    img_data = preprocess(img_scaled)

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

print(f"\nPNet: {len(total_boxes)} boxes after NMS")
print(f"Top 5 boxes by score:")
sorted_idx = np.argsort(total_boxes[:, 4])[::-1][:5]
for i, idx in enumerate(sorted_idx):
    print_bbox(total_boxes[idx], f"Box {i+1} (score: {total_boxes[idx, 4]:.3f})")

# Square bbox for RNet
print("\n" + "=" * 80)
print("STAGE 1→2: Square bbox transformation")
print("=" * 80)

total_boxes_squared = square_bbox(total_boxes)

print(f"\nTop 5 boxes after squaring:")
for i, idx in enumerate(sorted_idx):
    print_bbox(total_boxes_squared[idx], f"Box {i+1} (was {total_boxes[idx, 2]-total_boxes[idx, 0]:.1f}×{total_boxes[idx, 3]-total_boxes[idx, 1]:.1f})")

# Run RNet
print("\n" + "=" * 80)
print("STAGE 2: RNet")
print("=" * 80)

rnet_inputs = []
rnet_input_indices = []

for i in range(total_boxes_squared.shape[0]):
    x1 = int(max(0, total_boxes_squared[i, 0]))
    y1 = int(max(0, total_boxes_squared[i, 1]))
    x2 = int(min(img_w, total_boxes_squared[i, 2]))
    y2 = int(min(img_h, total_boxes_squared[i, 3]))

    if x2 <= x1 or y2 <= y1:
        continue

    face = img_float[y1:y2, x1:x2]
    face = cv2.resize(face, (24, 24))
    rnet_inputs.append(preprocess(face))
    rnet_input_indices.append(i)

total_boxes_squared = total_boxes_squared[rnet_input_indices]

# Run RNet
rnet_outputs = []
for face_data in rnet_inputs:
    output = rnet(face_data)[-1]
    rnet_outputs.append(output)

output = np.vstack(rnet_outputs)

# Calculate scores
scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

# Filter by threshold
rnet_threshold = 0.7
keep = scores > rnet_threshold
total_boxes_after_rnet = total_boxes_squared[keep]
scores_after_rnet = scores[keep]
reg = output[keep, 2:6]

print(f"\nRNet: {len(total_boxes_after_rnet)} boxes passed threshold {rnet_threshold}")
print(f"Boxes before regression:")
for i in range(len(total_boxes_after_rnet)):
    print_bbox(total_boxes_after_rnet[i], f"Face {i+1} (score: {scores_after_rnet[i]:.3f})")

# NMS
keep = nms(total_boxes_after_rnet, 0.7, 'Union')
total_boxes_after_rnet = total_boxes_after_rnet[keep]
scores_after_rnet = scores_after_rnet[keep]
reg = reg[keep]

print(f"\nAfter NMS: {len(total_boxes_after_rnet)} boxes")

# Apply regression
print("\n" + "=" * 80)
print("STAGE 2→3: Bbox regression")
print("=" * 80)

print(f"\nRegression offsets:")
for i in range(len(reg)):
    print(f"  Face {i+1}: [{reg[i, 0]:.4f}, {reg[i, 1]:.4f}, {reg[i, 2]:.4f}, {reg[i, 3]:.4f}]")

w = total_boxes_after_rnet[:, 2] - total_boxes_after_rnet[:, 0]
h = total_boxes_after_rnet[:, 3] - total_boxes_after_rnet[:, 1]

print(f"\nBefore regression:")
for i in range(len(total_boxes_after_rnet)):
    print_bbox(total_boxes_after_rnet[i], f"Face {i+1}")

total_boxes_after_rnet[:, 0] += reg[:, 0] * w
total_boxes_after_rnet[:, 1] += reg[:, 1] * h
total_boxes_after_rnet[:, 2] += reg[:, 2] * w
total_boxes_after_rnet[:, 3] += reg[:, 3] * h
total_boxes_after_rnet[:, 4] = scores_after_rnet

print(f"\nAfter regression:")
for i in range(len(total_boxes_after_rnet)):
    print_bbox(total_boxes_after_rnet[i], f"Face {i+1}")

# Square bbox for ONet
print("\n" + "=" * 80)
print("STAGE 2→3: Square bbox for ONet")
print("=" * 80)

total_boxes_squared_for_onet = square_bbox(total_boxes_after_rnet)

print(f"\nBboxes for ONet input (squared):")
for i in range(len(total_boxes_squared_for_onet)):
    print_bbox(total_boxes_squared_for_onet[i], f"Face {i+1}")

# Extract actual crops
print("\n" + "=" * 80)
print("ACTUAL CROPS TO BE FED TO ONET")
print("=" * 80)

for i in range(total_boxes_squared_for_onet.shape[0]):
    x1 = int(max(0, total_boxes_squared_for_onet[i, 0]))
    y1 = int(max(0, total_boxes_squared_for_onet[i, 1]))
    x2 = int(min(img_w, total_boxes_squared_for_onet[i, 2]))
    y2 = int(min(img_h, total_boxes_squared_for_onet[i, 3]))

    print(f"\nFace {i+1}:")
    print(f"  Clipped coords: ({x1}, {y1}) → ({x2}, {y2})")
    print(f"  Crop size: {x2-x1}×{y2-y1} (will be resized to 48×48)")

    # Visualize on image
    img_vis = img.copy()
    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(img_vis, f"Face {i+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(f'debug_bbox_face_{i+1}.jpg', img_vis)

    # Save crop
    face_crop = img[y1:y2, x1:x2]
    cv2.imwrite(f'debug_bbox_crop_{i+1}.jpg', face_crop)

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print(f"\nC++ Gold Standard: x=331.6, y=753.5, w=367.9, h=422.8")
print(f"  This is the expected FINAL bbox after all stages")
print(f"\nOur Face 1 bbox for ONet:")
if len(total_boxes_squared_for_onet) > 0:
    x1, y1, x2, y2 = total_boxes_squared_for_onet[0, :4]
    w = x2 - x1
    h = y2 - y1
    print(f"  x={x1:.1f}, y={y1:.1f}, w={w:.1f}, h={h:.1f}")

    # Check if this is reasonable
    if w < 100 or h < 100:
        print(f"\n⚠️  WARNING: Bbox too small! Face should be ~368×423, got {w:.1f}×{h:.1f}")
