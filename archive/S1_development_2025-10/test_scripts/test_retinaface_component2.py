#!/usr/bin/env python3
"""
Test Component 2: Face Detection (RetinaFace) against C++ baseline

This validates that our ONNX RetinaFace detector successfully detects faces
in all frames where C++ OpenFace detected them.
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# Import our ONNX RetinaFace detector
from onnx_retinaface_detector import OptimizedFaceDetector

# Paths
VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
RETINAFACE_MODEL_PATH = "weights/Alignment_RetinaFace.pth"

def main():
    print("=" * 80)
    print("Component 2: Face Detection (RetinaFace) Validation")
    print("=" * 80)
    print()

    # Load C++ baseline CSV
    print(f"Loading C++ baseline from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    total_frames = len(df)
    cpp_success_count = df['success'].sum()
    print(f"  Total frames: {total_frames}")
    print(f"  C++ successful detections: {cpp_success_count}/{total_frames} ({100*cpp_success_count/total_frames:.1f}%)")
    print(f"  C++ mean confidence: {df['confidence'].mean():.3f}")
    print()

    # Load RetinaFace detector
    print(f"Loading RetinaFace detector: {RETINAFACE_MODEL_PATH}")
    detector = OptimizedFaceDetector(
        model_path=RETINAFACE_MODEL_PATH,
        device="cpu",
        confidence_threshold=0.02,
        nms_threshold=0.4,
        vis_threshold=0.5
    )
    print()

    # Open video
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_video_frames}")
    print()

    # Test RetinaFace on all frames
    print("Testing RetinaFace on all frames...")
    print()

    retinaface_results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        dets, _ = detector.detect_faces(frame, resize=1.0)

        # Check if face detected
        if dets is not None and len(dets) > 0:
            # Get primary detection
            det = dets[0]
            bbox = det[:4].astype(int)  # [x1, y1, x2, y2]
            confidence = det[4]
            landmarks_5pt = det[5:15].reshape(5, 2)  # 5 landmarks (eyes, nose, mouth corners)

            retinaface_results.append({
                'frame': frame_idx + 1,  # CSV uses 1-indexed frames
                'detected': 1,
                'confidence': confidence,
                'bbox_x1': bbox[0],
                'bbox_y1': bbox[1],
                'bbox_x2': bbox[2],
                'bbox_y2': bbox[3],
                'bbox_width': bbox[2] - bbox[0],
                'bbox_height': bbox[3] - bbox[1],
                'num_detections': len(dets)
            })
        else:
            retinaface_results.append({
                'frame': frame_idx + 1,
                'detected': 0,
                'confidence': 0.0,
                'bbox_x1': 0,
                'bbox_y1': 0,
                'bbox_x2': 0,
                'bbox_y2': 0,
                'bbox_width': 0,
                'bbox_height': 0,
                'num_detections': 0
            })

        frame_idx += 1

        # Progress
        if (frame_idx % 100) == 0:
            print(f"  Processed {frame_idx}/{total_video_frames} frames...")

    cap.release()
    print(f"  Completed: {frame_idx} frames processed")
    print()

    # Convert to DataFrame
    rf_df = pd.DataFrame(retinaface_results)

    # Compare with C++ baseline
    print("=" * 80)
    print("COMPARISON: RetinaFace vs C++ OpenFace")
    print("=" * 80)
    print()

    # Detection success rate
    rf_success_count = rf_df['detected'].sum()
    print(f"RetinaFace Detection Rate:")
    print(f"  Detected: {rf_success_count}/{frame_idx} ({100*rf_success_count/frame_idx:.1f}%)")
    print(f"  Mean confidence: {rf_df[rf_df['detected']==1]['confidence'].mean():.3f}")
    print()

    print(f"C++ OpenFace Detection Rate:")
    print(f"  Detected: {cpp_success_count}/{total_frames} ({100*cpp_success_count/total_frames:.1f}%)")
    print(f"  Mean confidence: {df[df['success']==1]['confidence'].mean():.3f}")
    print()

    # Frame-by-frame agreement
    # Merge DataFrames (only compare frames in both)
    min_frames = min(len(rf_df), len(df))
    rf_detected = rf_df['detected'].values[:min_frames]
    cpp_detected = df['success'].values[:min_frames]

    agreement = (rf_detected == cpp_detected).sum()
    print(f"Frame-by-Frame Agreement:")
    print(f"  Matching: {agreement}/{min_frames} ({100*agreement/min_frames:.1f}%)")
    print()

    # Find disagreements
    disagreements = np.where(rf_detected != cpp_detected)[0]
    if len(disagreements) > 0:
        print(f"Disagreements: {len(disagreements)} frames")
        print(f"  First 10 disagreeing frames: {disagreements[:10] + 1}")  # +1 for 1-indexed

        # Analyze disagreements
        rf_yes_cpp_no = np.sum((rf_detected == 1) & (cpp_detected == 0))
        rf_no_cpp_yes = np.sum((rf_detected == 0) & (cpp_detected == 1))
        print(f"  RetinaFace detected, C++ missed: {rf_yes_cpp_no}")
        print(f"  C++ detected, RetinaFace missed: {rf_no_cpp_yes}")
    else:
        print("✅ Perfect agreement! RetinaFace matches C++ OpenFace 100%")
    print()

    # Bounding box statistics
    print("=" * 80)
    print("RETINAFACE BOUNDING BOX STATISTICS")
    print("=" * 80)
    print()

    detected_frames = rf_df[rf_df['detected'] == 1]
    if len(detected_frames) > 0:
        print(f"Bounding Box Dimensions (pixels):")
        print(f"  Width:  mean={detected_frames['bbox_width'].mean():.1f}, "
              f"std={detected_frames['bbox_width'].std():.1f}, "
              f"min={detected_frames['bbox_width'].min()}, "
              f"max={detected_frames['bbox_width'].max()}")
        print(f"  Height: mean={detected_frames['bbox_height'].mean():.1f}, "
              f"std={detected_frames['bbox_height'].std():.1f}, "
              f"min={detected_frames['bbox_height'].min()}, "
              f"max={detected_frames['bbox_height'].max()}")
        print()

        print(f"Bounding Box Position (x1, y1):")
        print(f"  X: mean={detected_frames['bbox_x1'].mean():.1f}, "
              f"std={detected_frames['bbox_x1'].std():.1f}")
        print(f"  Y: mean={detected_frames['bbox_y1'].mean():.1f}, "
              f"std={detected_frames['bbox_y1'].std():.1f}")
        print()

        # Check for multiple detections
        multi_detections = rf_df[rf_df['num_detections'] > 1]
        if len(multi_detections) > 0:
            print(f"⚠️  Multiple faces detected in {len(multi_detections)} frames")
            print(f"   (Using primary detection only)")
        else:
            print(f"✅ Single face per frame in all detections")
        print()

    # Conclusion
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()

    if rf_success_count == cpp_success_count and agreement == min_frames:
        print("✅ COMPONENT 2 VALIDATED: RetinaFace matches C++ OpenFace perfectly")
        print()
        print("   Detection rate: 100% match")
        print("   Frame-by-frame agreement: 100%")
        print("   Ready for production use")
    elif agreement / min_frames > 0.95:
        print("✓ COMPONENT 2 WORKING: RetinaFace performs well")
        print()
        print(f"   Detection rate: {100*rf_success_count/frame_idx:.1f}% (C++: {100*cpp_success_count/total_frames:.1f}%)")
        print(f"   Frame-by-frame agreement: {100*agreement/min_frames:.1f}%")
        print("   Minor differences acceptable for production")
    else:
        print("⚠️  COMPONENT 2 NEEDS REVIEW: Significant differences detected")
        print()
        print(f"   Detection rate: {100*rf_success_count/frame_idx:.1f}% (C++: {100*cpp_success_count/total_frames:.1f}%)")
        print(f"   Frame-by-frame agreement: {100*agreement/min_frames:.1f}%")
        print("   Review disagreeing frames")
    print()

    # Save results for further analysis
    output_file = "retinaface_component2_validation.csv"
    rf_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    print()

if __name__ == "__main__":
    main()
