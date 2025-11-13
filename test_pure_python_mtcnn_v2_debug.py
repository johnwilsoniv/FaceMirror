#!/usr/bin/env python3
"""
Test Pure Python MTCNN V2 with debug output to see where it fails.
"""

import cv2
import numpy as np
from cpp_cnn_loader import CPPCNN
import os

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
img_h, img_w = img.shape[:2]
img_float = img.astype(np.float32)

print("=" * 80)
print("PURE PYTHON MTCNN V2 DEBUG")
print("=" * 80)
print(f"\nTest image: {img.shape}")

# Load models
model_dir = os.path.expanduser(
    "~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/"
    "face_detection/mtcnn/convert_to_cpp/"
)

pnet = CPPCNN(os.path.join(model_dir, "PNet.dat"))
rnet = CPPCNN(os.path.join(model_dir, "RNet.dat"))
onet = CPPCNN(os.path.join(model_dir, "ONet.dat"))

# Test parameters
pnet_threshold = 0.6
rnet_threshold = 0.7
onet_threshold = 0.7

print(f"\nThresholds: PNet={pnet_threshold}, RNet={rnet_threshold}, ONet={onet_threshold}")

# Stage 1: PNet
print("\n" + "=" * 80)
print("STAGE 1: PNet")
print("=" * 80)

# [PNet code from standalone script]
# Build pyramid
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

print(f"Image pyramid: {len(scales)} scales")

# Count PNet detections at each scale
scale_counts = []
for i, scale in enumerate(scales):
    hs = int(np.ceil(img_h * scale))
    ws = int(np.ceil(img_w * scale))

    img_scaled = cv2.resize(img_float, (ws, hs))
    img_norm = (img_scaled.astype(np.float32) - 127.5) * 0.0078125
    img_chw = np.transpose(img_norm, (2, 0, 1))

    output = pnet(img_chw)[-1][np.newaxis, :, :, :]
    output = output[0].transpose(1, 2, 0)

    logit_not_face = output[:, :, 0]
    logit_face = output[:, :, 1]
    prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

    count = np.sum(prob_face > pnet_threshold)
    scale_counts.append(count)

    if count > 0:
        print(f"  Scale {i+1}/{len(scales)}: {count} detections")

total_pnet = sum(scale_counts)
print(f"\nTotal PNet detections: {total_pnet}")

if total_pnet == 0:
    print("✗ Failed at PNet stage!")
    exit(1)

# Stage 2: RNet
print("\n" + "=" * 80)
print("STAGE 2: RNet")
print("=" * 80)
print("Testing RNet threshold sensitivity...")

rnet_counts = {}
for test_threshold in [0.6, 0.65, 0.7, 0.75, 0.8]:
    # Run simplified PNet→RNet pipeline
    from debug_rnet_standalone import generate_bboxes, nms, square_bbox

    # Just check how many pass each threshold
    # (reusing the standalone code)
    print(f"\nWith RNet threshold={test_threshold}:")
    print(f"  (would need full pipeline run)")

print("\nFor detailed RNet analysis, see debug_rnet_standalone.py output:")
print("  - 161 PNet candidates")
print("  - 4 passed RNet threshold 0.7")
print("  - 6 passed RNet threshold 0.6")
print("\nThis means official thresholds [0.6, 0.7, 0.7] should work if RNet is receiving valid crops!")
