#!/usr/bin/env python3
"""
Simple test with pure PyTorch (no ONNX) to avoid CoreML crashes
"""

import sys
import numpy as np
import cv2
from pathlib import Path
import csv

sys.path.insert(0, str(Path(__file__).parent))

# Force PyTorch by importing directly
from openface.landmark_detection import LandmarkDetector
from openface.multitask_model import MultitaskPredictor
from openface3_to_18au_adapter import OpenFace3To18AUAdapter

video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/D Normal Pts/IMG_0942.MOV"
output_csv = Path.home() / "Desktop/IMG_0942_PyTorch.csv"

weights_dir = Path(__file__).parent / 'weights'

print("Initializing pure PyTorch models (no ONNX, no CoreML)...")

# Initialize MTL model WITHOUT ONNX (just pass model_path, no onnx_model_path)
print("  Loading MTL model...")
multitask_model = MultitaskPredictor(
    model_path=str(weights_dir / 'MTL_backbone.pth'),
    device='cpu'
)
print("  ✓ MTL model loaded (PyTorch)")

# Initialize AU adapter
au_adapter = OpenFace3To18AUAdapter()
print("  ✓ AU adapter initialized")

# Process video
print(f"\nProcessing: {video_path}")
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"  Total frames: {total_frames}")
print(f"  FPS: {fps:.2f}")

csv_rows = []
frame_index = 0

print("Processing frames (using full frame, no face detection)...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_index / fps
    h, w = frame.shape[:2]

    try:
        # Use full frame (skip face detection for simplicity)
        cropped_face = frame
        confidence = 1.0

        # Extract AUs using multitask model
        emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_face)

        # Convert to CSV row (no landmarks)
        csv_row = au_adapter.get_csv_row_dict(
            au_vector_8d=au_output,
            landmarks_98=None,  # No landmarks
            frame_num=frame_index,
            timestamp=timestamp,
            confidence=confidence,
            success=1
        )

        csv_rows.append(csv_row)

    except Exception as e:
        # Create failed frame row
        dummy_au_8d = np.zeros(8)
        csv_row = au_adapter.get_csv_row_dict(
            au_vector_8d=dummy_au_8d,
            landmarks_98=None,
            frame_num=frame_index,
            timestamp=timestamp,
            confidence=0.0,
            success=0
        )
        csv_rows.append(csv_row)
        print(f"  Frame {frame_index} failed: {e}")

    frame_index += 1

    # Print progress every 100 frames
    if frame_index % 100 == 0:
        print(f"  Progress: {frame_index}/{total_frames} frames", end='\r')

cap.release()

# Write CSV
output_csv.parent.mkdir(parents=True, exist_ok=True)
with open(output_csv, 'w', newline='') as csvfile:
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

print(f"\n✓ Processed {len(csv_rows)} frames")
print(f"✓ Output: {output_csv}")
