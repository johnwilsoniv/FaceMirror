#!/usr/bin/env python3
"""Quick 5-frame accuracy test."""

import numpy as np
import pandas as pd
import cv2
import sys

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pymtcnn')

from pyclnf import CLNF

# Test parameters
VIDEO = '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Normal Cohort/IMG_0422.MOV'
CPP_CSV = '/tmp/cpp_0422/IMG_0422.csv'
N_FRAMES = 5

print("=" * 60)
print("QUICK ACCURACY TEST - 5 FRAMES")
print("=" * 60)

# Load C++ landmarks
cpp_df = pd.read_csv(CPP_CSV).head(N_FRAMES)
x_cols = [f'x_{i}' for i in range(68)]
y_cols = [f'y_{i}' for i in range(68)]

cpp_landmarks = []
for _, row in cpp_df.iterrows():
    lm = np.stack([row[x_cols].values, row[y_cols].values], axis=1).astype(np.float32)
    cpp_landmarks.append(lm)

# Initialize Python CLNF
print("\nInitializing pyclnf...")
clnf = CLNF(convergence_profile='video', detector='pymtcnn',
            use_validator=False, use_eye_refinement=False)

# Process frames
print(f"\nProcessing {N_FRAMES} frames...")
cap = cv2.VideoCapture(VIDEO)

py_landmarks = []
for i in range(N_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    landmarks, info = clnf.detect_and_fit(frame)
    py_landmarks.append(landmarks)

    # Compute error for this frame
    jaw_err = np.linalg.norm(landmarks[:17] - cpp_landmarks[i][:17], axis=1).mean()
    overall_err = np.linalg.norm(landmarks - cpp_landmarks[i], axis=1).mean()
    print(f"  Frame {i}: jaw={jaw_err:.2f}px, overall={overall_err:.2f}px")

cap.release()

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

jaw_errors = [np.linalg.norm(py[:17] - cpp[:17], axis=1).mean()
              for py, cpp in zip(py_landmarks, cpp_landmarks)]
overall_errors = [np.linalg.norm(py - cpp, axis=1).mean()
                  for py, cpp in zip(py_landmarks, cpp_landmarks)]

print(f"Jaw error:     mean={np.mean(jaw_errors):.2f}px, max={np.max(jaw_errors):.2f}px")
print(f"Overall error: mean={np.mean(overall_errors):.2f}px, max={np.max(overall_errors):.2f}px")

# Check Local[0] value
params = clnf.inner_model._current_params
print(f"\nFinal Local[0]: {params[6]:.2f}")
print(f"Final scale: {params[0]:.4f}")
