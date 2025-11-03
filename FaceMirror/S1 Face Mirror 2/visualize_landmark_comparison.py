#!/usr/bin/env python3
"""
Visualize landmarks from RetinaFace+PFLD, MTCNN+PFLD, and OpenFace ground truth.
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
print("LANDMARK VISUALIZATION: RetinaFace vs MTCNN vs OpenFace")
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
model_dir = Path(__file__).parent / 'weights'

retinaface = ONNXRetinaFaceDetector(
    str(model_dir / 'retinaface_mobilenet025_coreml.onnx'),
    use_coreml=False,
    confidence_threshold=0.5
)

print("Loading MTCNN...")
mtcnn = MTCNN(keep_all=False, device='cpu', post_process=False)

pfld = CunjianPFLDDetector(
    str(model_dir / 'pfld_cunjian.onnx'),
    use_coreml=False
)

for video_path, video_name, frame_idx in videos:
    print(f"\n{'='*80}")
    print(f"Processing {video_name} (Frame {frame_idx})")
    print('='*80)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"  ❌ Failed to read frame")
        continue

    # Get OpenFace ground truth
    openface_lms = extract_openface_landmarks(openface_dfs[video_name], frame_idx)

    # 1. RetinaFace + PFLD
    print(f"\n1. RetinaFace + PFLD:")
    rf_detections, _ = retinaface.detect_faces(frame)

    if len(rf_detections) > 0:
        rf_bbox = rf_detections[0][:4]
        rf_pfld_lms, _ = pfld.detect_landmarks(frame, rf_bbox)
        rf_error = compute_landmark_error(rf_pfld_lms, openface_lms)
        print(f"   Error: {rf_error:.2f} pixels")

        # Visualize
        vis_rf = draw_landmarks(frame, rf_pfld_lms, (0, 0, 255), radius=4)  # Red
        vis_rf = draw_landmarks(vis_rf, openface_lms, (0, 255, 0), radius=2)  # Green

        # Add bbox
        x1, y1, x2, y2 = rf_bbox
        cv2.rectangle(vis_rf, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        cv2.putText(vis_rf, f"{video_name} - RetinaFace+PFLD",
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(vis_rf, f"Red: PFLD (Error: {rf_error:.0f}px) | Green: OpenFace GT",
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        output_rf = f'/tmp/{video_name}_retinaface_landmarks.jpg'
        cv2.imwrite(output_rf, vis_rf)
        print(f"   Saved: {output_rf}")

    # 2. MTCNN + PFLD
    print(f"\n2. MTCNN + PFLD:")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mtcnn_boxes, mtcnn_probs, mtcnn_5pt = mtcnn.detect(frame_rgb, landmarks=True)

    if mtcnn_boxes is not None and len(mtcnn_boxes) > 0:
        mtcnn_bbox = mtcnn_boxes[0]
        mtcnn_pfld_lms, _ = pfld.detect_landmarks(frame, mtcnn_bbox)
        mtcnn_error = compute_landmark_error(mtcnn_pfld_lms, openface_lms)
        print(f"   Error: {mtcnn_error:.2f} pixels")

        # Visualize
        vis_mtcnn = draw_landmarks(frame, mtcnn_pfld_lms, (255, 0, 255), radius=4)  # Magenta
        vis_mtcnn = draw_landmarks(vis_mtcnn, openface_lms, (0, 255, 0), radius=2)  # Green

        # Add MTCNN 5-point landmarks in cyan
        if mtcnn_5pt is not None:
            for x, y in mtcnn_5pt[0]:
                cv2.circle(vis_mtcnn, (int(x), int(y)), 6, (255, 255, 0), 2)

        # Add bbox
        x1, y1, x2, y2 = mtcnn_bbox
        cv2.rectangle(vis_mtcnn, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)

        cv2.putText(vis_mtcnn, f"{video_name} - MTCNN+PFLD",
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(vis_mtcnn, f"Magenta: PFLD (Error: {mtcnn_error:.0f}px) | Green: OpenFace GT",
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_mtcnn, f"Cyan circles: MTCNN 5-pt landmarks",
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        output_mtcnn = f'/tmp/{video_name}_mtcnn_landmarks.jpg'
        cv2.imwrite(output_mtcnn, vis_mtcnn)
        print(f"   Saved: {output_mtcnn}")

    # 3. Side-by-side comparison
    print(f"\n3. Creating side-by-side comparison...")

    if len(rf_detections) > 0 and mtcnn_boxes is not None:
        # Create side-by-side comparison
        h, w = frame.shape[:2]
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)

        # Left: RetinaFace + PFLD
        comparison[:, :w] = vis_rf

        # Right: MTCNN + PFLD
        comparison[:, w:] = vis_mtcnn

        # Add dividing line
        cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 3)

        output_comparison = f'/tmp/{video_name}_landmark_comparison.jpg'
        cv2.imwrite(output_comparison, comparison)
        print(f"   Saved: {output_comparison}")

    # 4. Just OpenFace ground truth for reference
    vis_gt = draw_landmarks(frame, openface_lms, (0, 255, 0), radius=3)
    cv2.putText(vis_gt, f"{video_name} - OpenFace Ground Truth",
               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(vis_gt, "These are the CORRECT landmarks",
               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    output_gt = f'/tmp/{video_name}_openface_gt.jpg'
    cv2.imwrite(output_gt, vis_gt)
    print(f"\n4. OpenFace ground truth saved: {output_gt}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print("\nFiles created:")
print("  IMG_8401:")
print("    /tmp/IMG_8401_retinaface_landmarks.jpg")
print("    /tmp/IMG_8401_mtcnn_landmarks.jpg")
print("    /tmp/IMG_8401_landmark_comparison.jpg")
print("    /tmp/IMG_8401_openface_gt.jpg")
print("\n  IMG_9330:")
print("    /tmp/IMG_9330_retinaface_landmarks.jpg")
print("    /tmp/IMG_9330_mtcnn_landmarks.jpg")
print("    /tmp/IMG_9330_landmark_comparison.jpg")
print("    /tmp/IMG_9330_openface_gt.jpg")

print("\nOpening comparison images...")
import subprocess
subprocess.run(['open', '/tmp/IMG_8401_landmark_comparison.jpg'])
subprocess.run(['open', '/tmp/IMG_9330_landmark_comparison.jpg'])
subprocess.run(['open', '/tmp/IMG_8401_openface_gt.jpg'])
subprocess.run(['open', '/tmp/IMG_9330_openface_gt.jpg'])
