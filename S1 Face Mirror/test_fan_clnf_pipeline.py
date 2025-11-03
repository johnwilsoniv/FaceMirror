#!/usr/bin/env python3
"""
Test FAN initialization + CLNF refinement pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyfaceau"))

import cv2
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN
import face_alignment
from pyfaceau.clnf.clnf_detector import CLNFDetector

print("="*80)
print("FAN + CLNF Pipeline Test")
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

print("Loading CLNF...")
model_dir = Path(__file__).parent / 'weights' / 'clnf'
clnf = CLNFDetector(
    model_dir=model_dir,
    max_iterations=5,
    convergence_threshold=0.01
)

print("\n" + "="*80)
print("PIPELINE RESULTS")
print("="*80)

for video_path, video_name, frame_idx in videos:
    print(f"\n{'='*80}")
    print(f"{video_name} (Frame {frame_idx})")
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

    # Get MTCNN bbox
    print(f"\n  Step 1: MTCNN Face Detection")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mtcnn_boxes, mtcnn_probs, mtcnn_landmarks = mtcnn.detect(frame_rgb, landmarks=True)

    if mtcnn_boxes is None or len(mtcnn_boxes) == 0:
        print(f"     ❌ MTCNN failed to detect face")
        continue

    mtcnn_bbox = mtcnn_boxes[0]
    print(f"     ✅ BBox: ({mtcnn_bbox[0]:.0f}, {mtcnn_bbox[1]:.0f}) to ({mtcnn_bbox[2]:.0f}, {mtcnn_bbox[3]:.0f})")
    print(f"     Confidence: {mtcnn_probs[0]:.3f}")

    # Get FAN landmarks
    print(f"\n  Step 2: FAN Landmark Detection")
    fan_lms = fa.get_landmarks(frame_rgb)

    if fan_lms is None or len(fan_lms) == 0:
        print(f"     ❌ FAN failed to detect landmarks")
        continue

    fan_landmarks = fan_lms[0]  # Shape: (68, 2)
    fan_error = compute_landmark_error(fan_landmarks, openface_lms)
    print(f"     FAN initialization error: {fan_error:.2f} pixels")

    # Refine with CLNF
    print(f"\n  Step 3: CLNF Refinement")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        refined_landmarks, converged, num_iters = clnf.refine_landmarks(
            gray, fan_landmarks, scale_idx=0, regularization=0.5, multi_scale=True
        )

        refined_error = compute_landmark_error(refined_landmarks, openface_lms)
        improvement = fan_error - refined_error

        print(f"     Converged: {converged} after {num_iters} iterations")
        print(f"     CLNF refined error: {refined_error:.2f} pixels")
        print(f"     Improvement: {improvement:.2f} pixels ({improvement/fan_error*100:.1f}%)")

        if refined_error < 10:
            print(f"     ✅ EXCELLENT (<10px) - Matches OpenFace quality!")
        elif refined_error < 20:
            print(f"     ✅ GOOD (<20px) - Clinical quality")
        elif refined_error < 50:
            print(f"     ⚠️  ACCEPTABLE (<50px) - Usable")
        else:
            print(f"     ❌ POOR (>50px) - Not suitable")

    except Exception as e:
        print(f"     ❌ CLNF refinement failed: {e}")
        import traceback
        traceback.print_exc()
        continue

    # Visualize all three: FAN init, CLNF refined, OpenFace GT
    print(f"\n  Step 4: Creating Visualization")

    # Create three-panel comparison
    h, w = frame.shape[:2]
    comparison = np.zeros((h, w*3, 3), dtype=np.uint8)

    # Panel 1: FAN initialization
    vis_fan = draw_landmarks(frame, fan_landmarks, (255, 165, 0), radius=4)  # Orange
    vis_fan = draw_landmarks(vis_fan, openface_lms, (0, 255, 0), radius=2)  # Green
    cv2.putText(vis_fan, f"FAN Init ({fan_error:.0f}px)",
               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    comparison[:, :w] = vis_fan

    # Panel 2: CLNF refined
    vis_clnf = draw_landmarks(frame, refined_landmarks, (0, 255, 255), radius=4)  # Cyan
    vis_clnf = draw_landmarks(vis_clnf, openface_lms, (0, 255, 0), radius=2)  # Green
    cv2.putText(vis_clnf, f"CLNF Refined ({refined_error:.0f}px)",
               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    comparison[:, w:w*2] = vis_clnf

    # Panel 3: OpenFace ground truth
    vis_gt = draw_landmarks(frame, openface_lms, (0, 255, 0), radius=3)  # Green
    cv2.putText(vis_gt, "OpenFace Ground Truth",
               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    comparison[:, w*2:] = vis_gt

    # Add dividing lines
    cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 3)
    cv2.line(comparison, (w*2, 0), (w*2, h), (255, 255, 255), 3)

    # Add title
    title = f"{video_name}: Orange=FAN | Cyan=CLNF | Green=OpenFace GT"
    cv2.putText(comparison, title,
               (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    output_path = f'/tmp/{video_name}_fan_clnf_pipeline.jpg'
    cv2.imwrite(output_path, comparison)
    print(f"     Saved: {output_path}")

print("\n" + "="*80)
print("PIPELINE TEST COMPLETE")
print("="*80)
print("\nConclusion:")
print("  - If CLNF error <10px → Success! Matches OpenFace quality")
print("  - If CLNF error 10-20px → Good, clinical quality")
print("  - If CLNF error 20-50px → Acceptable, usable")
print("  - If CLNF error >50px → Failed, need different approach")
print("\nVisualizations:")
print("  /tmp/IMG_8401_fan_clnf_pipeline.jpg")
print("  /tmp/IMG_9330_fan_clnf_pipeline.jpg")

import subprocess
subprocess.run(['open', '/tmp/IMG_8401_fan_clnf_pipeline.jpg'])
subprocess.run(['open', '/tmp/IMG_9330_fan_clnf_pipeline.jpg'])
