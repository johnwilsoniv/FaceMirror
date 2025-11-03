#!/usr/bin/env python3
"""
Quick test: Compare MTCNN vs RetinaFace bbox quality on challenging frames.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyfaceau"))

import cv2
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN
from pyfaceau.detectors.retinaface import ONNXRetinaFaceDetector
from pyfaceau.detectors.pfld import CunjianPFLDDetector

print("="*80)
print("MTCNN vs RetinaFace BBox Quality Test")
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

# Initialize detectors
print("\nInitializing detectors...")
model_dir = Path(__file__).parent / 'weights'

# RetinaFace
retinaface = ONNXRetinaFaceDetector(
    str(model_dir / 'retinaface_mobilenet025_coreml.onnx'),
    use_coreml=False,
    confidence_threshold=0.5
)

# MTCNN
print("Loading MTCNN...")
mtcnn = MTCNN(keep_all=False, device='cpu', post_process=False)

# PFLD
pfld = CunjianPFLDDetector(
    str(model_dir / 'pfld_cunjian.onnx'),
    use_coreml=False
)

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

    # Test 1: RetinaFace
    print(f"\n  1️⃣  RetinaFace + PFLD:")
    rf_detections, _ = retinaface.detect_faces(frame)

    if len(rf_detections) > 0:
        rf_bbox = rf_detections[0][:4]
        rf_x1, rf_y1, rf_x2, rf_y2 = rf_bbox

        print(f"     BBox: ({rf_x1:.0f}, {rf_y1:.0f}) to ({rf_x2:.0f}, {rf_y2:.0f})")
        print(f"     Size: {rf_x2-rf_x1:.0f}x{rf_y2-rf_y1:.0f} pixels")

        # Get PFLD landmarks
        rf_pfld_lms, _ = pfld.detect_landmarks(frame, rf_bbox)
        rf_error = compute_landmark_error(rf_pfld_lms, openface_lms)

        print(f"     PFLD error: {rf_error:.2f} pixels")

        if rf_error > 100:
            print(f"     ❌ CATASTROPHIC ERROR (>{100}px)")
        elif rf_error > 20:
            print(f"     ⚠️  HIGH ERROR (>{20}px)")
        else:
            print(f"     ✅ ACCEPTABLE ERROR (<{20}px)")
    else:
        print(f"     ❌ No face detected")
        rf_bbox = None
        rf_error = float('inf')

    # Test 2: MTCNN
    print(f"\n  2️⃣  MTCNN + PFLD:")

    # Convert BGR to RGB for MTCNN
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect with MTCNN
    mtcnn_boxes, mtcnn_probs, mtcnn_landmarks = mtcnn.detect(frame_rgb, landmarks=True)

    if mtcnn_boxes is not None and len(mtcnn_boxes) > 0:
        mtcnn_bbox = mtcnn_boxes[0]
        mtcnn_x1, mtcnn_y1, mtcnn_x2, mtcnn_y2 = mtcnn_bbox

        print(f"     BBox: ({mtcnn_x1:.0f}, {mtcnn_y1:.0f}) to ({mtcnn_x2:.0f}, {mtcnn_y2:.0f})")
        print(f"     Size: {mtcnn_x2-mtcnn_x1:.0f}x{mtcnn_y2-mtcnn_y1:.0f} pixels")
        print(f"     Confidence: {mtcnn_probs[0]:.3f}")

        # Get PFLD landmarks
        mtcnn_pfld_lms, _ = pfld.detect_landmarks(frame, mtcnn_bbox)
        mtcnn_error = compute_landmark_error(mtcnn_pfld_lms, openface_lms)

        print(f"     PFLD error: {mtcnn_error:.2f} pixels")

        if mtcnn_error > 100:
            print(f"     ❌ CATASTROPHIC ERROR (>{100}px)")
        elif mtcnn_error > 20:
            print(f"     ⚠️  HIGH ERROR (>{20}px)")
        else:
            print(f"     ✅ ACCEPTABLE ERROR (<{20}px)")
    else:
        print(f"     ❌ No face detected")
        mtcnn_bbox = None
        mtcnn_error = float('inf')

    # Comparison
    print(f"\n  📊 Comparison:")
    if rf_bbox is not None and mtcnn_bbox is not None:
        improvement = rf_error - mtcnn_error
        if improvement > 0:
            print(f"     ✅ MTCNN is {improvement:.2f} pixels better ({improvement/rf_error*100:.1f}% improvement)")
        elif improvement < 0:
            print(f"     ❌ RetinaFace is {-improvement:.2f} pixels better")
        else:
            print(f"     ➖ Both are equal")

        # Check if MTCNN makes it viable for CLNF
        if rf_error > 100 and mtcnn_error < 50:
            print(f"     🎯 MTCNN makes CLNF viable! (error reduced from {rf_error:.0f}px to {mtcnn_error:.0f}px)")

    # Visualize comparison
    vis = frame.copy()

    # Draw RetinaFace bbox in red
    if rf_bbox is not None:
        cv2.rectangle(vis, (int(rf_x1), int(rf_y1)), (int(rf_x2), int(rf_y2)), (0, 0, 255), 2)
        cv2.putText(vis, f"RetinaFace (err: {rf_error:.0f}px)",
                   (int(rf_x1), int(rf_y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw MTCNN bbox in green
    if mtcnn_bbox is not None:
        cv2.rectangle(vis, (int(mtcnn_x1), int(mtcnn_y1)), (int(mtcnn_x2), int(mtcnn_y2)), (0, 255, 0), 2)
        cv2.putText(vis, f"MTCNN (err: {mtcnn_error:.0f}px)",
                   (int(mtcnn_x1), int(mtcnn_y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = f'/tmp/mtcnn_vs_retinaface_{video_name}.jpg'
    cv2.imwrite(output_path, vis)
    print(f"\n     Saved comparison: {output_path}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\nConclusion:")
print("  - If MTCNN gives <50px error → CLNF has a chance")
print("  - If MTCNN still >100px error → Need different approach")
print("\nVisual comparison saved to:")
print("  /tmp/mtcnn_vs_retinaface_IMG_8401.jpg")
print("  /tmp/mtcnn_vs_retinaface_IMG_9330.jpg")

import subprocess
subprocess.run(['open', '/tmp/mtcnn_vs_retinaface_IMG_8401.jpg'])
subprocess.run(['open', '/tmp/mtcnn_vs_retinaface_IMG_9330.jpg'])
