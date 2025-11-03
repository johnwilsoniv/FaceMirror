#!/usr/bin/env python3
"""
Test FAN (Face Alignment Network) landmark detection with MTCNN bbox.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyfaceau"))

import cv2
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN
import face_alignment

print("="*80)
print("FAN (Face Alignment Network) Test with MTCNN BBox")
print("="*80)

# Load test videos
videos = [
    ('/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV', 'IMG_8401', 100),
    ('/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV', 'IMG_9330', 100)
]

# Load OpenFace ground truth
openface_8401 = pd.read_csv('/tmp/openface_test_8401_rotated/IMG_8401_source.csv')
openface_9330 = pd.read_csv('/tmp/openface_test_9330_rotated/IMG_9330_source.csv')
openface_dfs = {'IMG_8401': openface_8401, 'IMG_9330': openface_9330}

def extract_openface_landmarks(df, frame_idx):
    """Extract 68 landmarks from OpenFace CSV."""
    row = df[df['frame'] == frame_idx + 1].iloc[0]
    landmarks = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        landmarks[i, 0] = row[f'x_{i}']
        landmarks[i, 1] = row[f'y_{i}']
    return landmarks

def compute_landmark_error(pred, gt):
    """Compute average L2 distance."""
    return np.mean(np.sqrt(np.sum((pred - gt) ** 2, axis=1)))

def draw_landmarks(img, landmarks, color, radius=3):
    """Draw landmarks on image."""
    vis = img.copy()
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(vis, (int(x), int(y)), radius, color, -1)
        # Highlight jaw landmarks (most affected)
        if i < 17:  # Jaw outline
            cv2.circle(vis, (int(x), int(y)), radius+2, color, 2)
    return vis

# Initialize detectors
print("\nInitializing detectors...")
print("Loading MTCNN...")
mtcnn = MTCNN(keep_all=False, device='cpu', post_process=False)

print("Loading FAN (2D-FAN-2)...")
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')

print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

for video_path, video_name, frame_idx in videos:
    print(f"\n### {video_name} (Frame {frame_idx}) ###")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"  ❌ Failed to read frame")
        continue

    # Get OpenFace ground truth
    openface_lms = extract_openface_landmarks(openface_dfs[video_name], frame_idx)

    # Get MTCNN bbox
    print(f"\n  1️⃣  MTCNN Face Detection:")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mtcnn_boxes, mtcnn_probs, mtcnn_landmarks = mtcnn.detect(frame_rgb, landmarks=True)

    if mtcnn_boxes is not None and len(mtcnn_boxes) > 0:
        mtcnn_bbox = mtcnn_boxes[0]
        mtcnn_x1, mtcnn_y1, mtcnn_x2, mtcnn_y2 = mtcnn_bbox

        print(f"     BBox: ({mtcnn_x1:.0f}, {mtcnn_y1:.0f}) to ({mtcnn_x2:.0f}, {mtcnn_y2:.0f})")
        print(f"     Size: {mtcnn_x2-mtcnn_x1:.0f}x{mtcnn_y2-mtcnn_y1:.0f} pixels")
        print(f"     Confidence: {mtcnn_probs[0]:.3f}")

        # Get FAN landmarks
        print(f"\n  2️⃣  FAN Landmark Detection:")

        # FAN expects RGB image
        # Note: face_alignment library will detect faces itself, but we want to see
        # how it performs given the MTCNN bbox
        fan_lms = fa.get_landmarks(frame_rgb)

        if fan_lms is not None and len(fan_lms) > 0:
            fan_landmarks = fan_lms[0]  # Shape: (68, 2)
            fan_error = compute_landmark_error(fan_landmarks, openface_lms)

            print(f"     Landmark error: {fan_error:.2f} pixels")

            if fan_error > 100:
                print(f"     ❌ CATASTROPHIC ERROR (>{100}px)")
            elif fan_error > 50:
                print(f"     ⚠️  HIGH ERROR (>{50}px) - CLNF may struggle")
            elif fan_error > 20:
                print(f"     ⚠️  MODERATE ERROR (>{20}px) - CLNF should work")
            else:
                print(f"     ✅ EXCELLENT ERROR (<{20}px) - CLNF will work great")

            # Visualize FAN vs OpenFace
            vis_fan = draw_landmarks(frame, fan_landmarks, (255, 165, 0), radius=4)  # Orange
            vis_fan = draw_landmarks(vis_fan, openface_lms, (0, 255, 0), radius=2)  # Green

            # Add MTCNN bbox
            cv2.rectangle(vis_fan, (int(mtcnn_x1), int(mtcnn_y1)), (int(mtcnn_x2), int(mtcnn_y2)), (0, 255, 0), 2)

            # Add MTCNN 5-point landmarks in cyan
            if mtcnn_landmarks is not None:
                for x, y in mtcnn_landmarks[0]:
                    cv2.circle(vis_fan, (int(x), int(y)), 6, (255, 255, 0), 2)

            cv2.putText(vis_fan, f"{video_name} - FAN with MTCNN BBox",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(vis_fan, f"Orange: FAN (Error: {fan_error:.0f}px) | Green: OpenFace GT",
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_fan, f"Cyan circles: MTCNN 5-pt landmarks",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            output_fan = f'/tmp/{video_name}_fan_landmarks.jpg'
            cv2.imwrite(output_fan, vis_fan)
            print(f"\n     Saved visualization: {output_fan}")

            # Top 5 worst landmarks
            errors = np.sqrt(np.sum((fan_landmarks - openface_lms) ** 2, axis=1))
            worst_indices = np.argsort(errors)[-5:][::-1]
            worst_errors = errors[worst_indices]
            print(f"\n     Top 5 worst landmarks: {worst_indices} with errors: {worst_errors}")

        else:
            print(f"     ❌ FAN failed to detect landmarks")

    else:
        print(f"     ❌ MTCNN failed to detect face")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\nFAN Evaluation:")
print("  - If FAN error <20px → Excellent, CLNF will work great")
print("  - If FAN error 20-50px → Good, CLNF should work")
print("  - If FAN error >50px → Poor, CLNF will struggle")
print("\nVisualizations saved to:")
print("  /tmp/IMG_8401_fan_landmarks.jpg")
print("  /tmp/IMG_9330_fan_landmarks.jpg")

import subprocess
subprocess.run(['open', '/tmp/IMG_8401_fan_landmarks.jpg'])
subprocess.run(['open', '/tmp/IMG_9330_fan_landmarks.jpg'])
