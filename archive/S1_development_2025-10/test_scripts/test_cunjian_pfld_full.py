#!/usr/bin/env python3
"""Full 50-frame validation of cunjian PFLD model"""

import cv2
import pandas as pd
import numpy as np
import onnxruntime as ort

VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
MODEL_PATH = "weights/pfld_cunjian.onnx"

# Load CSV ground truth (50 frames)
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} frames of ground truth data")

# Load model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

all_rmse = []
all_nme = []
all_mean_errors = []

for frame_idx in range(min(50, len(df))):
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {frame_idx}")
        break

    # Get ground truth for this frame
    gt_x = df.iloc[frame_idx][[f'x_{i}' for i in range(68)]].values
    gt_y = df.iloc[frame_idx][[f'y_{i}' for i in range(68)]].values
    gt_landmarks = np.stack([gt_x, gt_y], axis=1)

    # Get bbox from ground truth with 10% padding
    x_min, y_min = gt_landmarks[:, 0].min(), gt_landmarks[:, 1].min()
    x_max, y_max = gt_landmarks[:, 0].max(), gt_landmarks[:, 1].max()
    w = x_max - x_min
    h = y_max - y_min
    size = int(max([w, h]) * 1.1)
    cx = int(x_min + w / 2)
    cy = int(y_min + h / 2)
    x1 = cx - size // 2
    x2 = x1 + size
    y1 = cy - size // 2
    y2 = y1 + size

    # Clip to image bounds
    height, width = frame.shape[:2]
    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)
    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)

    # Crop face
    cropped = frame[y1:y2, x1:x2]

    # Add padding if needed
    if dx > 0 or dy > 0 or edx > 0 or edy > 0:
        cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx),
                                      cv2.BORDER_CONSTANT, 0)

    # Preprocess
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(cropped_rgb, (112, 112))
    face_normalized = face_resized.astype(np.float32) / 255.0
    face_input = np.transpose(face_normalized, (2, 0, 1))
    face_input = np.expand_dims(face_input, axis=0)

    # Run inference
    output = session.run(None, {input_name: face_input})[0]
    landmarks = output.reshape(-1, 2)

    # Reproject to original coordinates
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    landmarks_reprojected = np.zeros_like(landmarks)
    for i, point in enumerate(landmarks):
        landmarks_reprojected[i, 0] = point[0] * bbox_w + x1
        landmarks_reprojected[i, 1] = point[1] * bbox_h + y1

    # Calculate metrics
    errors = np.sqrt(np.sum((landmarks_reprojected - gt_landmarks)**2, axis=1))
    rmse = np.sqrt(np.mean(errors**2))

    # Calculate NME
    left_eye = gt_landmarks[36:42].mean(axis=0)
    right_eye = gt_landmarks[42:48].mean(axis=0)
    interocular_dist = np.linalg.norm(left_eye - right_eye)
    nme = np.mean(errors) / interocular_dist * 100

    all_rmse.append(rmse)
    all_nme.append(nme)
    all_mean_errors.append(errors.mean())

    if (frame_idx + 1) % 10 == 0:
        print(f"Processed {frame_idx + 1} frames...")

cap.release()

# Calculate summary statistics
print(f"\n{'='*70}")
print(f"CUNJIAN PFLD - 50 FRAME VALIDATION RESULTS:")
print(f"{'='*70}")
print(f"\nRMSE:")
print(f"  Mean: {np.mean(all_rmse):.2f} pixels")
print(f"  Std:  {np.std(all_rmse):.2f} pixels")
print(f"  Min:  {np.min(all_rmse):.2f} pixels")
print(f"  Max:  {np.max(all_rmse):.2f} pixels")

print(f"\nNME:")
print(f"  Mean: {np.mean(all_nme):.2f}%")
print(f"  Std:  {np.std(all_nme):.2f}%")
print(f"  Min:  {np.min(all_nme):.2f}%")
print(f"  Max:  {np.max(all_nme):.2f}%")

print(f"\nMean Error per Landmark:")
print(f"  Mean: {np.mean(all_mean_errors):.2f} pixels")
print(f"  Std:  {np.std(all_mean_errors):.2f} pixels")

print(f"\n{'='*70}")
print(f"COMPARISON WITH OTHER MODELS:")
print(f"{'='*70}")
print(f"{'Model':<20} {'RMSE (px)':<15} {'NME (%)':<15}")
print(f"{'-'*50}")
print(f"{'Wrong PFLD':<20} {'13.26':<15} {'~11.0':<15}")
print(f"{'FAN2':<20} {'6.95':<15} {'5.79':<15}")
print(f"{'Cunjian PFLD':<20} {f'{np.mean(all_rmse):.2f}':<15} {f'{np.mean(all_nme):.2f}':<15}")
print(f"\nExpected (from paper): 3.97% NME on 300W Full Set")

# Determine winner
print(f"\n{'='*70}")
print(f"ANALYSIS:")
print(f"{'='*70}")
if np.mean(all_nme) < 5.79:
    nme_improvement = ((5.79 - np.mean(all_nme)) / 5.79) * 100
    print(f"✅ Cunjian PFLD has {nme_improvement:.1f}% better NME than FAN2")
else:
    nme_degradation = ((np.mean(all_nme) - 5.79) / 5.79) * 100
    print(f"❌ Cunjian PFLD has {nme_degradation:.1f}% worse NME than FAN2")

if np.mean(all_rmse) < 6.95:
    rmse_improvement = ((6.95 - np.mean(all_rmse)) / 6.95) * 100
    print(f"✅ Cunjian PFLD has {rmse_improvement:.1f}% better RMSE than FAN2")
else:
    rmse_degradation = ((np.mean(all_rmse) - 6.95) / 6.95) * 100
    print(f"❌ Cunjian PFLD has {rmse_degradation:.1f}% worse RMSE than FAN2")

print(f"\nKey insight: NME normalizes by inter-ocular distance, making it more")
print(f"robust for AU extraction where relative positions matter more than absolute pixels.")
