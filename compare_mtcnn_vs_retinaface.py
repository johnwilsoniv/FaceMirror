#!/usr/bin/env python3
"""
Compare raw face detectors: MTCNN (5 pts) vs RetinaFace (5 pts)
No landmark refinement - just the baseline detectors
"""

import sys
import cv2
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path

# Add pyfaceau to path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau.detectors.retinaface import ONNXRetinaFaceDetector


def draw_detector_output(img, bbox, landmarks_5, color, label):
    """
    Draw bbox and 5 facial landmarks

    Args:
        bbox: [x1, y1, x2, y2] or [x, y, w, h] (will auto-detect)
        landmarks_5: [(x1,y1), (x2,y2), ...] or [x1,y1,x2,y2,...]
        color: BGR tuple
        label: Text label for detector
    """
    if bbox is None:
        cv2.putText(img, f"{label}: No detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return

    # Draw bbox (handle both formats)
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        # Check if it's x,y,w,h format (w and h would be positive and reasonable)
        if x2 > 0 and y2 > 0 and x2 < img.shape[1] and y2 < img.shape[0] and x2 < x1 + 1000:
            # Likely x,y,w,h format
            x, y, w, h = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            bbox_for_norm = [x, y, w, h]
        else:
            # x1,y1,x2,y2 format
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            bbox_for_norm = [x1, y1, x2-x1, y2-y1]

        cv2.putText(img, label, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw 5 landmarks
    if landmarks_5 is not None:
        # Handle different formats
        if isinstance(landmarks_5, (list, np.ndarray)):
            if len(landmarks_5) == 10:
                # Flat array [x1,y1,x2,y2,...]
                # Check if normalized (0-1) or absolute
                max_val = np.max(landmarks_5)

                if max_val <= 1.0:
                    # Normalized - need to convert to absolute using bbox
                    x, y, w, h = bbox_for_norm
                    for i in range(0, 10, 2):
                        lm_x = int(x + landmarks_5[i] * w)
                        lm_y = int(y + landmarks_5[i+1] * h)
                        cv2.circle(img, (lm_x, lm_y), 3, color, -1)
                else:
                    # Absolute coordinates
                    for i in range(0, 10, 2):
                        cv2.circle(img, (int(landmarks_5[i]), int(landmarks_5[i+1])), 3, color, -1)

            elif len(landmarks_5) == 5:
                # List of (x,y) tuples
                for pt in landmarks_5:
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 3, color, -1)


def process_video_comparison(video_path, mtcnn_csv_path, retinaface_detector, output_dir, num_frames=3):
    """Compare MTCNN vs RetinaFace on a video"""

    print(f"\nProcessing: {video_path.name}")

    # Read MTCNN detections
    if not mtcnn_csv_path.exists():
        print(f"  ERROR: MTCNN CSV not found")
        return 0

    mtcnn_df = pd.read_csv(mtcnn_csv_path)
    print(f"  MTCNN detections: {len(mtcnn_df)} frames")

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

        # Left: OpenFace MTCNN (green)
        left_frame = frame.copy()
        if frame_idx < len(mtcnn_df):
            row = mtcnn_df.iloc[frame_idx]
            bbox = [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']]

            # Extract 5 landmarks (normalized 0-1)
            lm_cols = [c for c in mtcnn_df.columns if c.startswith('lm')]
            if len(lm_cols) == 10:
                landmarks = [row[c] for c in lm_cols]
                draw_detector_output(left_frame, bbox, landmarks, (0, 255, 0), "MTCNN (5 pts)")
            else:
                draw_detector_output(left_frame, bbox, None, (0, 255, 0), "MTCNN (bbox only)")

        canvas[:, :w] = left_frame

        # Right: RetinaFace (red)
        right_frame = frame.copy()

        try:
            # RetinaFace detection
            detections, _ = retinaface_detector.detect_faces(frame)

            if detections is not None and len(detections) > 0:
                # Get first (highest confidence) detection
                # Detection format: [x1, y1, x2, y2, conf, lm1_x, lm1_y, ..., lm5_x, lm5_y]
                det = detections[0]
                bbox = det[:4]  # [x1, y1, x2, y2]

                # Extract 5 landmarks from detection array (indices 5-14)
                if len(det) >= 15:
                    lm_5 = det[5:15]  # [lm1_x, lm1_y, lm2_x, lm2_y, ..., lm5_x, lm5_y]
                else:
                    lm_5 = None

                draw_detector_output(right_frame, bbox, lm_5, (0, 0, 255), "RetinaFace (5 pts)")
            else:
                cv2.putText(right_frame, "RetinaFace: No detection", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        except Exception as e:
            cv2.putText(right_frame, f"RetinaFace error: {str(e)[:30]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(f"  RetinaFace error on frame {frame_idx}: {e}")

        canvas[:, w:] = right_frame

        # Add frame number
        cv2.putText(canvas, f"Frame {frame_idx}", (10, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        comparison_frames.append(canvas)

    cap.release()

    # Save comparison image
    if comparison_frames:
        final_grid = np.vstack(comparison_frames)
        output_path = output_dir / f"{video_path.stem}_mtcnn_vs_retinaface.jpg"
        cv2.imwrite(str(output_path), final_grid)
        print(f"  Saved: {output_path.name}")

    return len(comparison_frames)


def main():
    # Test videos that OpenFace can process (no crashes)
    video_dir = Path("/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data")
    videos = [
        "IMG_0441_source.MOV",
        "IMG_0452_source.MOV",
        "IMG_0861_source.MOV",
        "IMG_0942_source.MOV"
    ]

    output_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/test_output")
    output_dir.mkdir(exist_ok=True)

    # Initialize RetinaFace
    print("Initializing RetinaFace...")
    weights_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/weights")
    retinaface = ONNXRetinaFaceDetector(
        str(weights_dir / 'retinaface_mobilenet025_coreml.onnx'),
        use_coreml=True,
        confidence_threshold=0.5,
        nms_threshold=0.4
    )
    print("  RetinaFace loaded\n")

    print("="*60)
    print("RAW DETECTOR COMPARISON")
    print("Left: OpenFace MTCNN (5 landmarks)")
    print("Right: RetinaFace (5 landmarks)")
    print("="*60)

    for video_name in videos:
        video_path = video_dir / video_name

        if not video_path.exists():
            print(f"\nSkipping {video_name} - not found")
            continue

        # Run OpenFace to generate MTCNN CSV
        mtcnn_csv = Path("/tmp/mtcnn_debug.csv")

        print(f"\n  Running OpenFace on {video_name}...")
        result = subprocess.run([
            "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction",
            "-f", str(video_path),
            "-out_dir", "/tmp/openface_raw_test"
        ], capture_output=True, timeout=180)

        if result.returncode != 0:
            print(f"  OpenFace failed (exit code {result.returncode})")
            continue

        # Process comparison
        process_video_comparison(video_path, mtcnn_csv, retinaface, output_dir, num_frames=3)

    print("\n" + "="*60)
    print("Comparison complete! Check test_output/ for results")
    print("="*60)


if __name__ == "__main__":
    main()
