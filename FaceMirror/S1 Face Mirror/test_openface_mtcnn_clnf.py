#!/usr/bin/env python3
"""
Test OpenFace MTCNN + Python CLNF on Challenging Cases

Tests the complete OpenFace pipeline in pure Python:
1. OpenFace MTCNN detection (with CLNF-compatible bbox correction)
2. Python CLNF refinement
3. Comparison with OpenFace C++ ground truth
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyfaceau"))

import cv2
import numpy as np
import pandas as pd
from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN
from pyfaceau.clnf.clnf_detector import CLNFDetector

print("="*80)
print("OpenFace MTCNN + Python CLNF Pipeline Test")
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

def bbox_to_landmarks_68(bbox):
    """
    Initialize 68 landmarks from bbox using simplified OpenFace approach.

    OpenFace uses:
    1. PDM mean shape scaled to fit bbox
    2. Centered in bbox

    We'll use a simple grid-based initialization for now.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Simplified 68-point initialization (OpenFace uses PDM mean shape)
    landmarks = np.zeros((68, 2), dtype=np.float32)

    # Jaw (0-16): bottom arc
    for i in range(17):
        t = i / 16.0
        landmarks[i, 0] = x1 + t * w
        landmarks[i, 1] = y2 - h * 0.2 + np.sin(t * np.pi) * h * 0.15

    # Eyebrows (17-26)
    for i in range(17, 22):  # Right eyebrow
        t = (i - 17) / 4.0
        landmarks[i, 0] = x1 + w * (0.2 + t * 0.2)
        landmarks[i, 1] = y1 + h * 0.3

    for i in range(22, 27):  # Left eyebrow
        t = (i - 22) / 4.0
        landmarks[i, 0] = x1 + w * (0.6 + t * 0.2)
        landmarks[i, 1] = y1 + h * 0.3

    # Nose (27-35)
    for i in range(27, 36):
        landmarks[i, 0] = cx
        landmarks[i, 1] = y1 + h * (0.4 + (i - 27) * 0.05)

    # Eyes (36-47)
    for i in range(36, 42):  # Right eye
        t = (i - 36) / 5.0
        landmarks[i, 0] = x1 + w * 0.3 + np.cos(t * 2 * np.pi) * w * 0.05
        landmarks[i, 1] = y1 + h * 0.45 + np.sin(t * 2 * np.pi) * h * 0.03

    for i in range(42, 48):  # Left eye
        t = (i - 42) / 5.0
        landmarks[i, 0] = x1 + w * 0.7 + np.cos(t * 2 * np.pi) * w * 0.05
        landmarks[i, 1] = y1 + h * 0.45 + np.sin(t * 2 * np.pi) * h * 0.03

    # Mouth (48-67)
    for i in range(48, 68):
        t = (i - 48) / 19.0
        landmarks[i, 0] = x1 + w * (0.3 + t * 0.4)
        landmarks[i, 1] = y1 + h * (0.75 + np.sin(t * np.pi) * 0.05)

    return landmarks

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
print("Loading OpenFace MTCNN...")
try:
    mtcnn = OpenFaceMTCNN()
    print("  ✅ OpenFace MTCNN loaded successfully")
except Exception as e:
    print(f"  ❌ Failed to load OpenFace MTCNN: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

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

    # Step 1: OpenFace MTCNN Detection
    print(f"\n  Step 1: OpenFace MTCNN Detection")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        bboxes, landmarks_5pt = mtcnn.detect(frame_rgb, return_landmarks=True)

        if len(bboxes) == 0:
            print(f"     ❌ No face detected")
            continue

        bbox = bboxes[0]
        lm_5pt = landmarks_5pt[0] if landmarks_5pt is not None else None

        print(f"     ✅ BBox: ({bbox[0]:.0f}, {bbox[1]:.0f}) to ({bbox[2]:.0f}, {bbox[3]:.0f})")
        print(f"     BBox has CLNF-compatible correction applied")
        if lm_5pt is not None:
            print(f"     5-point landmarks detected")

    except Exception as e:
        print(f"     ❌ MTCNN detection failed: {e}")
        import traceback
        traceback.print_exc()
        continue

    # Step 2: Initialize 68 landmarks from bbox
    print(f"\n  Step 2: Initialize 68 Landmarks from BBox")
    init_landmarks = bbox_to_landmarks_68(bbox)
    init_error = compute_landmark_error(init_landmarks, openface_lms)
    print(f"     Initial 68-point error: {init_error:.2f} pixels")

    # Step 3: CLNF Refinement
    print(f"\n  Step 3: CLNF Refinement")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        refined_landmarks, converged, num_iters = clnf.refine_landmarks(
            gray, init_landmarks, scale_idx=0, regularization=0.5, multi_scale=True
        )

        refined_error = compute_landmark_error(refined_landmarks, openface_lms)
        improvement = init_error - refined_error

        print(f"     Converged: {converged} after {num_iters} iterations")
        print(f"     CLNF refined error: {refined_error:.2f} pixels")
        print(f"     Improvement: {improvement:.2f} pixels ({improvement/init_error*100:.1f}%)")

        if refined_error < 10:
            print(f"     ✅ EXCELLENT (<10px) - Matches OpenFace C++ quality!")
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

    # Step 4: Visualization
    print(f"\n  Step 4: Creating Visualization")

    # Create three-panel comparison
    h, w = frame.shape[:2]
    comparison = np.zeros((h, w*3, 3), dtype=np.uint8)

    # Panel 1: MTCNN bbox + initialization
    vis_init = frame.copy()
    cv2.rectangle(vis_init, (int(bbox[0]), int(bbox[1])),
                 (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    vis_init = draw_landmarks(vis_init, init_landmarks, (255, 165, 0), radius=4)  # Orange
    vis_init = draw_landmarks(vis_init, openface_lms, (0, 255, 0), radius=2)  # Green
    cv2.putText(vis_init, f"MTCNN Init ({init_error:.0f}px)",
               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    comparison[:, :w] = vis_init

    # Panel 2: CLNF refined
    vis_clnf = draw_landmarks(frame, refined_landmarks, (0, 255, 255), radius=4)  # Cyan
    vis_clnf = draw_landmarks(vis_clnf, openface_lms, (0, 255, 0), radius=2)  # Green
    cv2.putText(vis_clnf, f"CLNF Refined ({refined_error:.0f}px)",
               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    comparison[:, w:w*2] = vis_clnf

    # Panel 3: OpenFace ground truth
    vis_gt = draw_landmarks(frame, openface_lms, (0, 255, 0), radius=3)  # Green
    cv2.putText(vis_gt, "OpenFace C++ GT",
               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    comparison[:, w*2:] = vis_gt

    # Add dividing lines
    cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 3)
    cv2.line(comparison, (w*2, 0), (w*2, h), (255, 255, 255), 3)

    # Add title
    title = f"{video_name}: Orange=Init | Cyan=CLNF | Green=OpenFace GT"
    cv2.putText(comparison, title,
               (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    output_path = f'/tmp/{video_name}_openface_mtcnn_clnf.jpg'
    cv2.imwrite(output_path, comparison)
    print(f"     Saved: {output_path}")

print("\n" + "="*80)
print("PIPELINE TEST COMPLETE")
print("="*80)
print("\nConclusion:")
print("  This test evaluates the full OpenFace pipeline in pure Python:")
print("  1. OpenFace MTCNN with CLNF-compatible bbox correction")
print("  2. Python CLNF refinement")
print("  3. Comparison with OpenFace C++ ground truth")
print("\nSuccess criteria:")
print("  - CLNF error <10px → SUCCESS (matches OpenFace C++)")
print("  - CLNF error 10-20px → GOOD (clinical quality)")
print("  - CLNF error 20-50px → ACCEPTABLE (usable)")
print("\nVisualizations:")
print("  /tmp/IMG_8401_openface_mtcnn_clnf.jpg")
print("  /tmp/IMG_9330_openface_mtcnn_clnf.jpg")

import subprocess
subprocess.run(['open', '/tmp/IMG_8401_openface_mtcnn_clnf.jpg'])
subprocess.run(['open', '/tmp/IMG_9330_openface_mtcnn_clnf.jpg'])
