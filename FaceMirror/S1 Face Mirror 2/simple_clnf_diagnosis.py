#!/usr/bin/env python3
"""
Simple diagnosis: Check RetinaFace bbox and PFLD initialization quality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyfaceau"))

import cv2
import numpy as np
import pandas as pd
from pyfaceau.detectors.retinaface import ONNXRetinaFaceDetector
from pyfaceau.detectors.pfld import CunjianPFLDDetector

print("="*80)
print("SIMPLE CLNF DIAGNOSIS")
print("="*80)

# Load test videos
video_8401 = '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV'
video_9330 = '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV'

# Load OpenFace ground truth
openface_8401 = pd.read_csv('/tmp/openface_test_8401_rotated/IMG_8401_source.csv')
openface_9330 = pd.read_csv('/tmp/openface_test_9330_rotated/IMG_9330_source.csv')

def extract_openface_landmarks(df, frame_idx):
    """Extract 68 landmarks from OpenFace CSV."""
    row = df[df['frame'] == frame_idx + 1].iloc[0]
    landmarks = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        landmarks[i, 0] = row[f'x_{i}']
        landmarks[i, 1] = row[f'y_{i}']
    return landmarks

def compute_landmark_error(pred, gt):
    """Compute average L2 distance between predicted and ground truth landmarks."""
    return np.mean(np.sqrt(np.sum((pred - gt) ** 2, axis=1)))

# Initialize detectors
print("\nInitializing detectors...")
model_dir = Path(__file__).parent / 'weights'
face_detector = ONNXRetinaFaceDetector(
    str(model_dir / 'retinaface_mobilenet025_coreml.onnx'),
    use_coreml=False,  # Use ONNX for consistency
    confidence_threshold=0.5
)
landmark_detector = CunjianPFLDDetector(
    str(model_dir / 'pfld_cunjian.onnx'),
    use_coreml=False
)

print("\n" + "="*80)
print("HYPOTHESIS 1: Face Detection & Bounding Box Quality")
print("="*80)

for video_path, video_name in [(video_8401, 'IMG_8401'), (video_9330, 'IMG_9330')]:
    print(f"\n### Testing {video_name} ###")
    cap = cv2.VideoCapture(video_path)

    # Test frame 100
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()

    if ret:
        # Detect face with RetinaFace
        detections, _ = face_detector.detect_faces(frame)

        if len(detections) > 0:
            bbox = detections[0]
            x1, y1, x2, y2 = bbox[:4]  # First 4 values are bbox coordinates

            print(f"  RetinaFace bbox: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
            print(f"  Face size: {x2-x1:.0f}x{y2-y1:.0f} pixels")

            # Visualize bbox
            vis = frame.copy()
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.putText(vis, f"{video_name} - RetinaFace Detection",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(vis, f"BBox: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})",
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show face crop
            face_crop = frame[int(y1):int(y2), int(x1):int(x2)]

            output_path = f'/tmp/diagnosis_{video_name}_bbox.jpg'
            cv2.imwrite(output_path, vis)
            print(f"  Saved bbox visualization: {output_path}")

            output_crop = f'/tmp/diagnosis_{video_name}_face_crop.jpg'
            cv2.imwrite(output_crop, face_crop)
            print(f"  Saved face crop: {output_crop}")
        else:
            print(f"  ❌ No face detected!")

    cap.release()

print("\n" + "="*80)
print("HYPOTHESIS 2: PFLD Initialization Quality")
print("="*80)

for video_path, video_name, openface_df in [
    (video_8401, 'IMG_8401', openface_8401),
    (video_9330, 'IMG_9330', openface_9330)
]:
    print(f"\n### Testing {video_name} ###")
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()

    if ret:
        # Detect with PFLD
        detections, _ = face_detector.detect_faces(frame)

        if len(detections) > 0:
            bbox = detections[0][:4]  # First 4 values are bbox coordinates
            pfld_landmarks, _ = landmark_detector.detect_landmarks(frame, bbox)

            # Get OpenFace ground truth
            openface_landmarks = extract_openface_landmarks(openface_df, 100)

            # Compute error
            pfld_error = compute_landmark_error(pfld_landmarks, openface_landmarks)

            print(f"  PFLD initialization error: {pfld_error:.2f} pixels")

            if pfld_error > 10:
                print(f"  ⚠️  PFLD error is HIGH (>{10}px) - poor initialization!")
                print(f"  This could explain why CLNF fails - bad starting point!")
            else:
                print(f"  ✅ PFLD error is reasonable (<{10}px)")

            # Visualize
            vis = frame.copy()
            for i, (x, y) in enumerate(pfld_landmarks):
                cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
            for i, (x, y) in enumerate(openface_landmarks):
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

            # Highlight jaw/mouth region (most affected by markings/paralysis)
            for i in range(48, 68):  # Mouth landmarks
                x, y = pfld_landmarks[i]
                cv2.circle(vis, (int(x), int(y)), 5, (255, 0, 255), 2)

            cv2.putText(vis, f"{video_name} - PFLD (Red) vs OpenFace (Green)",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis, f"Error: {pfld_error:.2f} pixels",
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis, f"Magenta circles: Mouth region (critical for paralysis)",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            output_path = f'/tmp/diagnosis_{video_name}_pfld_vs_openface.jpg'
            cv2.imwrite(output_path, vis)
            print(f"  Saved comparison: {output_path}")

            # Show error distribution
            errors = np.sqrt(np.sum((pfld_landmarks - openface_landmarks) ** 2, axis=1))
            worst_landmarks = np.argsort(errors)[-5:]
            print(f"  Top 5 worst landmarks: {worst_landmarks} with errors: {errors[worst_landmarks]}")

    cap.release()

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
print("\nOpen these images to see the problem:")
print("  /tmp/diagnosis_IMG_8401_bbox.jpg - RetinaFace bounding box")
print("  /tmp/diagnosis_IMG_8401_face_crop.jpg - Detected face crop")
print("  /tmp/diagnosis_IMG_8401_pfld_vs_openface.jpg - PFLD vs OpenFace landmarks")
print("  /tmp/diagnosis_IMG_9330_bbox.jpg")
print("  /tmp/diagnosis_IMG_9330_face_crop.jpg")
print("  /tmp/diagnosis_IMG_9330_pfld_vs_openface.jpg")

# Open images
import subprocess
subprocess.run(['open', '/tmp/diagnosis_IMG_8401_bbox.jpg'])
subprocess.run(['open', '/tmp/diagnosis_IMG_8401_face_crop.jpg'])
subprocess.run(['open', '/tmp/diagnosis_IMG_8401_pfld_vs_openface.jpg'])
subprocess.run(['open', '/tmp/diagnosis_IMG_9330_bbox.jpg'])
subprocess.run(['open', '/tmp/diagnosis_IMG_9330_face_crop.jpg'])
subprocess.run(['open', '/tmp/diagnosis_IMG_9330_pfld_vs_openface.jpg'])
