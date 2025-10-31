#!/usr/bin/env python3
"""Test cunjian PFLD model accuracy against ground truth"""

import cv2
import pandas as pd
import numpy as np
import onnxruntime as ort

VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
MODEL_PATH = "weights/pfld_cunjian.onnx"

# Load CSV ground truth
df = pd.read_csv(CSV_PATH)
gt_x = df[[f'x_{i}' for i in range(68)]].values[0]
gt_y = df[[f'y_{i}' for i in range(68)]].values[0]
gt_landmarks = np.stack([gt_x, gt_y], axis=1)

# Load model
session = ort.InferenceSession(MODEL_PATH)
print(f"Model inputs: {[(inp.name, inp.shape) for inp in session.get_inputs()]}")
print(f"Model outputs: {[(out.name, out.shape) for out in session.get_outputs()]}")

# Open video and get first frame
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

print(f"\nFrame shape: {frame.shape}")

# Get bbox from ground truth with 10% padding (following cunjian approach)
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

print(f"BBox: [{x1}, {y1}, {x2}, {y2}] (size: {x2-x1}x{y2-y1})")

# Crop face
cropped = frame[y1:y2, x1:x2]

# Add padding if face was at edge
if dx > 0 or dy > 0 or edx > 0 or edy > 0:
    cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx),
                                  cv2.BORDER_CONSTANT, 0)
    print(f"Added padding: dy={dy}, edy={edy}, dx={dx}, edx={edx}")

print(f"Cropped face shape: {cropped.shape}")

# Preprocess following cunjian approach
cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
face_resized = cv2.resize(cropped_rgb, (112, 112))

# Convert to float and normalize to [0, 1] (ToTensor equivalent)
face_normalized = face_resized.astype(np.float32) / 255.0
face_input = np.transpose(face_normalized, (2, 0, 1))  # HWC -> CHW
face_input = np.expand_dims(face_input, axis=0)  # Add batch dimension

print(f"\nInput shape: {face_input.shape}")
print(f"Input range: [{face_input.min():.3f}, {face_input.max():.3f}]")

# Run inference
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: face_input})[0]

print(f"\nOutput shape: {output.shape}")
print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

# Post-process landmarks (output is in [0, 1] normalized space)
landmarks = output.reshape(-1, 2)

# Reproject to original image coordinates
# Following cunjian's reprojectLandmark logic
bbox_w = x2 - x1
bbox_h = y2 - y1
landmarks_reprojected = np.zeros_like(landmarks)
for i, point in enumerate(landmarks):
    landmarks_reprojected[i, 0] = point[0] * bbox_w + x1
    landmarks_reprojected[i, 1] = point[1] * bbox_h + y1

# Calculate RMSE
errors = np.sqrt(np.sum((landmarks_reprojected - gt_landmarks)**2, axis=1))
rmse = np.sqrt(np.mean(errors**2))

# Calculate NME (normalized by inter-ocular distance)
left_eye = gt_landmarks[36:42].mean(axis=0)
right_eye = gt_landmarks[42:48].mean(axis=0)
interocular_dist = np.linalg.norm(left_eye - right_eye)
nme = np.mean(errors) / interocular_dist * 100

print(f"\n{'='*60}")
print(f"CUNJIAN PFLD RESULTS:")
print(f"{'='*60}")
print(f"RMSE: {rmse:.2f} pixels")
print(f"NME: {nme:.2f}%")
print(f"Mean error: {errors.mean():.2f} pixels")
print(f"Max error: {errors.max():.2f} pixels")
print(f"Min error: {errors.min():.2f} pixels")
print(f"\nComparison:")
print(f"  Wrong PFLD (HuggingFace): 13.26px RMSE / ~11% NME")
print(f"  FAN2:                      6.95px RMSE / 5.79% NME")
print(f"  Cunjian PFLD:             {rmse:.2f}px RMSE / {nme:.2f}% NME")
print(f"\nExpected (from paper): 3.97% NME on 300W Full Set")

# Visualize
vis_frame = frame.copy()
for x, y in landmarks_reprojected:
    cv2.circle(vis_frame, (int(x), int(y)), 3, (255, 0, 0), -1)  # BLUE = predicted
for x, y in gt_landmarks:
    cv2.circle(vis_frame, (int(x), int(y)), 1, (0, 255, 0), -1)  # GREEN = ground truth

cv2.putText(vis_frame, f"Cunjian PFLD: RMSE={rmse:.2f}px, NME={nme:.2f}%",
            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
cv2.putText(vis_frame, "BLUE=PFLD, GREEN=GT",
            (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

cv2.imwrite("comparison_cunjian_pfld.jpg", vis_frame)
print(f"\nSaved visualization: comparison_cunjian_pfld.jpg")
