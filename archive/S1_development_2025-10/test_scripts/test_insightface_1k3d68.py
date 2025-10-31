#!/usr/bin/env python3
"""Quick test of InsightFace 1k3d68 model"""

import cv2
import pandas as pd
import numpy as np
import onnxruntime as ort

VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
MODEL_PATH = "weights/1k3d68.onnx"

# Load CSV
df = pd.read_csv(CSV_PATH)
gt_x = df[[f'x_{i}' for i in range(68)]].values[0]
gt_y = df[[f'y_{i}' for i in range(68)]].values[0]
gt = np.stack([gt_x, gt_y], axis=1)

# Load model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

print(f"Frame shape: {frame.shape}")

# Get bbox from GT
x_min, y_min = gt[:, 0].min(), gt[:, 1].min()
x_max, y_max = gt[:, 0].max(), gt[:, 1].max()
pad_w, pad_h = (x_max-x_min)*0.2, (y_max-y_min)*0.2
x1, y1 = int(x_min - pad_w), int(y_min - pad_h)
x2, y2 = int(x_max + pad_w), int(y_max + pad_h)

face_crop = frame[y1:y2, x1:x2]
print(f"Face crop: {face_crop.shape}")
print(f"Bbox: [{x1}, {y1}, {x2}, {y2}]")
print()

# Preprocess (192×192 RGB normalized)
face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
face_resized = cv2.resize(face_rgb, (192, 192))
face_normalized = face_resized.astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1] (InsightFace style)
face_input = np.transpose(face_normalized, (2, 0, 1))
face_input = np.expand_dims(face_input, axis=0)

print("Input shape:", face_input.shape)
print("Input range:", face_input.min(), "to", face_input.max())
print()

# Inference
output = session.run(None, {input_name: face_input})[0]
print("Output shape:", output.shape)
print("Output values:", output.shape[1], "total values")
print()

# Try to interpret the output
# InsightFace 1k3d68 should output 68 3D landmarks
# Format might be: [68 x-coords, 68 y-coords, 68 z-coords] or [x, y, z, x, y, z, ...]

# Try interpretation 1: First 68*2 values are 2D coords
if output.shape[1] >= 136:
    landmarks_v1 = output[0, :136].reshape(68, 2)
    landmarks_v1[:, 0] = landmarks_v1[:, 0] * (x2 - x1) + x1
    landmarks_v1[:, 1] = landmarks_v1[:, 1] * (y2 - y1) + y1
    rmse_v1 = np.sqrt(np.mean(np.sum((landmarks_v1 - gt)**2, axis=1)))
    print(f"Interpretation 1 (first 136 values as normalized 2D): RMSE={rmse_v1:.2f}px")

# Try interpretation 2: Structured as [68 x, 68 y, 68 z, ...]
if output.shape[1] >= 204:
    x_coords = output[0, :68]
    y_coords = output[0, 68:136]
    landmarks_v2 = np.stack([x_coords, y_coords], axis=1)
    landmarks_v2[:, 0] = landmarks_v2[:, 0] * (x2 - x1) + x1
    landmarks_v2[:, 1] = landmarks_v2[:, 1] * (y2 - y1) + y1
    rmse_v2 = np.sqrt(np.mean(np.sum((landmarks_v2 - gt)**2, axis=1)))
    print(f"Interpretation 2 (structured x,y,z blocks): RMSE={rmse_v2:.2f}px")

# Try interpretation 3: First 136 values are absolute pixel coords
landmarks_v3 = output[0, :136].reshape(68, 2)
rmse_v3 = np.sqrt(np.mean(np.sum((landmarks_v3 - gt)**2, axis=1)))
print(f"Interpretation 3 (first 136 as absolute pixels): RMSE={rmse_v3:.2f}px")

print()
print("Sample output values (first 20):")
print(output[0, :20])
print()
print("Ground truth first landmark:", gt[0])
print("Output interpretation 1 first landmark:", landmarks_v1[0])
print("Output interpretation 2 first landmark:", landmarks_v2[0])
