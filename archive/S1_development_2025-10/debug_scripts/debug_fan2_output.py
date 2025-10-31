#!/usr/bin/env python3
"""Debug FAN2 output format"""

import cv2
import pandas as pd
import numpy as np
import onnxruntime as ort

VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
FAN2_MODEL = "weights/fan2_68_landmark.onnx"

# Load CSV
df = pd.read_csv(CSV_PATH)
landmark_cols_x = [f'x_{i}' for i in range(68)]
landmark_cols_y = [f'y_{i}' for i in range(68)]
landmarks_x = df[landmark_cols_x].values[0]
landmarks_y = df[landmark_cols_y].values[0]
gt_landmarks = np.stack([landmarks_x, landmarks_y], axis=1)

print("Ground truth landmarks (first frame):")
print(f"  Range X: {gt_landmarks[:, 0].min():.1f} to {gt_landmarks[:, 0].max():.1f}")
print(f"  Range Y: {gt_landmarks[:, 1].min():.1f} to {gt_landmarks[:, 1].max():.1f}")
print()

# Load FAN2
session = ort.InferenceSession(FAN2_MODEL)
input_name = session.get_inputs()[0].name

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

print(f"Frame shape: {frame.shape}")
print()

# Get face crop using GT bbox
x_min = gt_landmarks[:, 0].min()
y_min = gt_landmarks[:, 1].min()
x_max = gt_landmarks[:, 0].max()
y_max = gt_landmarks[:, 1].max()
width = x_max - x_min
height = y_max - y_min
pad_w = width * 0.2
pad_h = height * 0.2
x1 = int(x_min - pad_w)
y1 = int(y_min - pad_h)
x2 = int(x_max + pad_w)
y2 = int(y_max + pad_h)

face_crop = frame[y1:y2, x1:x2]
print(f"Face crop shape: {face_crop.shape}")
print(f"Bounding box: [{x1}, {y1}, {x2}, {y2}]")
print()

# Preprocess
face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
face_resized = cv2.resize(face_rgb, (256, 256))
face_normalized = face_resized.astype(np.float32) / 255.0
face_input = np.transpose(face_normalized, (2, 0, 1))
face_input = np.expand_dims(face_input, axis=0)

print("Input shape:", face_input.shape)
print()

# Inference
outputs = session.run(None, {input_name: face_input})
output = outputs[0][0]  # (68, 3)

print("Output shape:", output.shape)
print("Output range:")
print(f"  X: {output[:, 0].min():.3f} to {output[:, 0].max():.3f}")
print(f"  Y: {output[:, 1].min():.3f} to {output[:, 1].max():.3f}")
print(f"  Score: {output[:, 2].min():.3f} to {output[:, 2].max():.3f}")
print()

print("First 5 landmarks (raw output):")
for i in range(5):
    print(f"  Point {i}: x={output[i, 0]:.3f}, y={output[i, 1]:.3f}, score={output[i, 2]:.3f}")
print()

# Try different interpretations
print("Testing different coordinate interpretations:")
print()

# Interpretation 1: Output is in [0, 256] range
print("1. Output in [0, 256] range (scale to crop):")
landmarks_v1 = output[:, :2].copy()
landmarks_v1[:, 0] = landmarks_v1[:, 0] * (x2 - x1) / 256.0 + x1
landmarks_v1[:, 1] = landmarks_v1[:, 1] * (y2 - y1) / 256.0 + y1
rmse_v1 = np.sqrt(np.mean(np.sum((landmarks_v1 - gt_landmarks) ** 2, axis=1)))
print(f"   RMSE: {rmse_v1:.2f} pixels")

# Interpretation 2: Output is normalized [0, 1]
print("2. Output in [0, 1] range (normalized):")
landmarks_v2 = output[:, :2].copy()
landmarks_v2[:, 0] = landmarks_v2[:, 0] * (x2 - x1) + x1
landmarks_v2[:, 1] = landmarks_v2[:, 1] * (y2 - y1) + y1
rmse_v2 = np.sqrt(np.mean(np.sum((landmarks_v2 - gt_landmarks) ** 2, axis=1)))
print(f"   RMSE: {rmse_v2:.2f} pixels")

# Interpretation 3: Output is already absolute
print("3. Output is absolute coordinates:")
landmarks_v3 = output[:, :2].copy()
rmse_v3 = np.sqrt(np.mean(np.sum((landmarks_v3 - gt_landmarks) ** 2, axis=1)))
print(f"   RMSE: {rmse_v3:.2f} pixels")

# Interpretation 4: Output relative to crop, no scaling
print("4. Output relative to crop (add bbox offset):")
landmarks_v4 = output[:, :2].copy()
landmarks_v4[:, 0] += x1
landmarks_v4[:, 1] += y1
rmse_v4 = np.sqrt(np.mean(np.sum((landmarks_v4 - gt_landmarks) ** 2, axis=1)))
print(f"   RMSE: {rmse_v4:.2f} pixels")

print()
print(f"Best interpretation: {np.argmin([rmse_v1, rmse_v2, rmse_v3, rmse_v4]) + 1}")
print(f"Best RMSE: {min(rmse_v1, rmse_v2, rmse_v3, rmse_v4):.2f} pixels")
