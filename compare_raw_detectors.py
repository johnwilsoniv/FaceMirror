#!/usr/bin/env python3
"""
Compare raw detector outputs: OpenFace MTCNN vs Our RetinaFace+PFLD
Shows detections BEFORE any CLNF/SVR refinement
"""

import sys
import cv2
import numpy as np
import pandas as pd
import subprocess
import json
from pathlib import Path

# Import pyfaceau
from pyfaceau.detectors.retinaface import ONNXRetinaFaceDetector
from pyfaceau.detectors.pfld import ONNXPFLDDetector


def draw_mtcnn_detection(img, bbox, landmarks_norm, color=(0, 255, 0)):
    """Draw MTCNN bbox and 5 landmarks"""
    # Draw bbox
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    # Landmarks are normalized (0-1) relative to bbox
    if landmarks_norm is not None and len(landmarks_norm) == 10:
        for i in range(0, 10, 2):
            lm_x = int(x + landmarks_norm[i] * w)
            lm_y = int(y + landmarks_norm[i+1] * h)
            cv2.circle(img, (lm_x, lm_y), 3, color, -1)

    cv2.putText(img, "MTCNN (5 pts)", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_retinaface_pfld(img, bbox, landmarks_98, color=(255, 0, 0)):
    """Draw RetinaFace bbox and PFLD 98 landmarks"""
    # Draw bbox
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, "RetinaFace+PFLD (98 pts)", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw PFLD landmarks (98 points)
    if landmarks_98 is not None:
        for i in range(0, len(landmarks_98), 2):
            x = int(landmarks_98[i])
            y = int(landmarks_98[i+1])
            cv2.circle(img, (x, y), 1, color, -1)


def process_video_comparison(video_path, mtcnn_csv_path, output_dir, num_frames=3):
    """Compare raw detectors on a video"""

    print(f"\nProcessing: {video_path}")

    # Read MTCNN detections
    mtcnn_df = pd.read_csv(mtcnn_csv_path)
    print(f"  MTCNN detections: {len(mtcnn_df)} frames")

    # Initialize our detectors
    retinaface = ONNXRetinaFaceDetector()
    pfld = ONNXPFLDDetector()

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames-1, min(num_frames, len(mtcnn_df)), dtype=int)

    comparison_frames = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Create side-by-side canvas
        h, w = frame.shape[:2]
        canvas = np.zeros((h, w*2, 3), dtype=np.uint8)

        # Left: OpenFace MTCNN
        left_frame = frame.copy()
        if frame_idx < len(mtcnn_df):
            row = mtcnn_df.iloc[frame_idx]
            bbox = [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']]

            # Extract landmarks if available
            lm_cols = [c for c in mtcnn_df.columns if c.startswith('lm')]
            if len(lm_cols) == 10:
                landmarks = [row[c] for c in lm_cols]
                draw_mtcnn_detection(left_frame, bbox, landmarks)
            else:
                draw_mtcnn_detection(left_frame, bbox, None)

        canvas[:, :w] = left_frame

        # Right: Our RetinaFace + PFLD (raw, no refinement)
        right_frame = frame.copy()

        # RetinaFace detection
        face_info = retinaface.detect_faces(frame, score_threshold=0.5)

        if face_info and len(face_info) > 0:
            # Get first (highest confidence) detection
            bbox_xywh = face_info[0]['bbox']  # x, y, width, height
            landmarks_5 = face_info[0].get('landmarks', None)

            # Convert bbox to x1,y1,x2,y2 format
            bbox = [bbox_xywh[0], bbox_xywh[1],
                   bbox_xywh[0] + bbox_xywh[2],
                   bbox_xywh[1] + bbox_xywh[3]]

            # PFLD needs PIL format landmarks for alignment
            if landmarks_5 is not None:
                # Run PFLD to get 98 landmarks
                try:
                    landmarks_98 = pfld.detect_landmarks(frame, bbox, landmarks_5)
                    draw_retinaface_pfld(right_frame, bbox, landmarks_98)
                except Exception as e:
                    # If PFLD fails, just draw bbox
                    cv2.rectangle(right_frame,
                                (int(bbox[0]), int(bbox[1])),
                                (int(bbox[2]), int(bbox[3])),
                                (255, 0, 0), 2)
                    cv2.putText(right_frame, f"PFLD failed: {str(e)[:20]}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            else:
                # Just draw bbox if no landmarks
                cv2.rectangle(right_frame,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            (255, 0, 0), 2)

        canvas[:, w:] = right_frame

        # Add frame number
        cv2.putText(canvas, f"Frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        comparison_frames.append(canvas)

    cap.release()

    # Save comparison image
    if comparison_frames:
        final_grid = np.vstack(comparison_frames)
        output_path = output_dir / f"{Path(video_path).stem}_raw_detector_comparison.jpg"
        cv2.imwrite(str(output_path), final_grid)
        print(f"  Saved: {output_path}")

    return len(comparison_frames)


def main():
    # Test videos
    video_dir = Path("/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data")
    videos = [
        "IMG_0441_source.MOV",
        "IMG_0452_source.MOV",
        "IMG_0861_source.MOV",
        "IMG_0942_source.MOV"
    ]

    output_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/test_output")
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("RAW DETECTOR COMPARISON")
    print("Left: OpenFace MTCNN (5 landmarks, before CLNF)")
    print("Right: RetinaFace + PFLD (98 landmarks, before SVR)")
    print("="*60)

    for video_name in videos:
        video_path = video_dir / video_name

        if not video_path.exists():
            print(f"Skipping {video_name} - not found")
            continue

        # Run OpenFace to generate MTCNN CSV
        mtcnn_csv = Path("/tmp/mtcnn_debug.csv")

        # Run OpenFace
        print(f"\n  Running OpenFace on {video_name}...")
        import subprocess
        result = subprocess.run([
            "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction",
            "-f", str(video_path),
            "-out_dir", "/tmp/openface_raw_test"
        ], capture_output=True, timeout=180)

        if result.returncode != 0:
            print(f"  OpenFace failed on {video_name}")
            continue

        if not mtcnn_csv.exists():
            print(f"  MTCNN CSV not generated for {video_name}")
            continue

        # Process comparison
        process_video_comparison(video_path, mtcnn_csv, output_dir, num_frames=3)

    print("\n" + "="*60)
    print("Comparison complete! Check test_output/ for results")
    print("="*60)


if __name__ == "__main__":
    main()
