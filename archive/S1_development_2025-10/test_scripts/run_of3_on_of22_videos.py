#!/usr/bin/env python3
"""
Simple script to run OpenFace 3.0 AU extraction on OpenFace 2.2-mirrored videos

This tests if OF2.2's higher-quality mirroring improves OF3.0 AU predictions.
"""

import config
config.apply_environment_settings()

import sys
import os
from pathlib import Path
import cv2
import pandas as pd
import numpy as np

# Suppress ONNX warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Loading OpenFace 3.0 components...")
from onnx_retinaface_detector import OptimizedFaceDetector
from onnx_mtl_detector import ONNXMultitaskPredictor
from openface3_to_18au_adapter import OpenFace3To18AUAdapter

def process_video(video_path, output_csv_path):
    """Process a single video with OF3.0 AU extraction"""

    print(f"\nProcessing: {video_path.name}")
    print(f"Output: {output_csv_path.name}")

    # Initialize models
    weights_dir = Path(__file__).parent / "weights"

    print("  Loading face detector...")
    face_detector = OptimizedFaceDetector(
        model_path=str(weights_dir / "retinaface_mobilenet025_coreml.onnx"),
        confidence_threshold=0.5,
        nms_threshold=0.4
    )

    print("  Loading AU predictor...")
    mtl_predictor = ONNXMultitaskPredictor(
        onnx_model_path=str(weights_dir / "mtl_efficientnet_b0_coreml.onnx")
    )

    print("  Loading AU adapter...")
    au_adapter = OpenFace3To18AUAdapter()

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"  Video: {total_frames} frames @ {fps:.2f} fps")
    print(f"  Processing...")

    # Storage for results
    results = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Progress update - show every 10 frames for debugging
        if frame_num % 10 == 0 or frame_num <= 10:
            pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
            print(f"    Frame {frame_num}/{total_frames} ({pct:.1f}%)", flush=True)

        try:
            # Detect face
            detections, _ = face_detector.detect_faces(frame)
        except Exception as e:
            print(f"\n    ERROR at frame {frame_num} during face detection: {e}", flush=True)
            import traceback
            traceback.print_exc()
            break

        if detections is None or len(detections) == 0:
            # No face detected - store NaN row
            row = {'frame': frame_num, 'success': 0, 'confidence': 0.0}
            for au in ['AU01_r', 'AU02_r', 'AU04_r', 'AU06_r', 'AU12_r',
                      'AU15_r', 'AU20_r', 'AU25_r', 'AU45_r']:
                row[au] = np.nan
            results.append(row)
            continue

        # Get largest face (detections is 2D array)
        det = detections[0]
        bbox = det[:4].astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        confidence = float(det[4])

        # Crop face with padding
        h, w = frame.shape[:2]
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            # Invalid crop
            row = {'frame': frame_num, 'success': 0, 'confidence': confidence}
            for au in ['AU01_r', 'AU02_r', 'AU04_r', 'AU06_r', 'AU12_r',
                      'AU15_r', 'AU20_r', 'AU25_r', 'AU45_r']:
                row[au] = np.nan
            results.append(row)
            continue

        # Predict AUs
        try:
            emotion, gaze, aus = mtl_predictor.predict(face_crop)

            # Convert 8 AUs to 18 AUs using adapter
            adapted_aus = au_adapter.adapt_frame(aus, landmarks_98=None, frame_num=frame_num)

            # Store results (only intensity values _r, not binary _c)
            row = {'frame': frame_num, 'success': 1, 'confidence': confidence}
            for au_name in au_adapter.expected_aus_r:
                row[au_name] = adapted_aus[au_name]

            results.append(row)
        except Exception as e:
            print(f"\n    ERROR at frame {frame_num} during AU prediction: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Store NaN row and continue
            row = {'frame': frame_num, 'success': 0, 'confidence': confidence}
            for au_name in au_adapter.expected_aus_r:
                row[au_name] = np.nan
            results.append(row)

    cap.release()
    print(f"\n  Processed {frame_num} frames")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"  Saved: {output_csv_path}")

    return frame_num


def main():
    """Main processing function"""

    # Input videos (mirrored by OF2.2)
    left_video = Path("/Users/johnwilsoniv/Documents/open2GR/1_Face_Mirror/output/IMG_0942_left_mirrored.mp4")
    right_video = Path("/Users/johnwilsoniv/Documents/open2GR/1_Face_Mirror/output/IMG_0942_right_mirrored.mp4")

    # Output directory
    output_dir = Path("/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/OP3 v OP 22")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OpenFace 3.0 Processing of OpenFace 2.2-Mirrored Videos")
    print("=" * 80)
    print("\nPurpose: Test if OF2.2's higher-quality mirroring improves OF3.0 AU predictions")
    print(f"\nInput videos:")
    print(f"  {left_video}")
    print(f"  {right_video}")
    print(f"\nOutput directory:")
    print(f"  {output_dir}")
    print("\n" + "=" * 80)

    # Check input files
    if not left_video.exists():
        print(f"\nERROR: Left video not found: {left_video}")
        sys.exit(1)
    if not right_video.exists():
        print(f"\nERROR: Right video not found: {right_video}")
        sys.exit(1)

    # Process both videos
    videos = [
        (left_video, output_dir / "IMG_0942_left_mirroredOP22_processedONNXv3.csv"),
        (right_video, output_dir / "IMG_0942_right_mirroredOP22_processedONNXv3.csv")
    ]

    for video_path, csv_path in videos:
        try:
            frame_count = process_video(video_path, csv_path)
            if frame_count == 0:
                print(f"  ERROR: No frames processed")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print("\nOutput files created:")
    for _, csv_path in videos:
        if csv_path.exists():
            print(f"  ✓ {csv_path.name}")
        else:
            print(f"  ✗ {csv_path.name} (not created)")

    print("\nNext step: Run comparison analysis")


if __name__ == "__main__":
    main()
