#!/usr/bin/env python3
"""
Simple diagnostic to check RetinaFace behavior during AU extraction.
Creates one processor at a time to avoid CoreML conflicts.
"""

import os
import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from openface_integration import OpenFace3Processor


def diagnose_retinaface(video_path, max_frames=30):
    """
    Check if RetinaFace is running and what bboxes it's creating

    Args:
        video_path: Path to mirrored video
        max_frames: Number of frames to analyze
    """
    video_path = Path(video_path)

    print("=" * 80)
    print("RETINAFACE DIAGNOSTIC")
    print("=" * 80)
    print(f"Video: {video_path.name}")
    print()

    # ============================================================================
    # TEST: Current behavior (RetinaFace should be running)
    # ============================================================================
    print("Creating processor with DEFAULT settings (skip_face_detection NOT specified)...")
    print("-" * 80)

    processor = OpenFace3Processor(device='cpu', debug_mode=False)

    print(f"\nProcessor configuration:")
    print(f"  skip_face_detection = {processor.skip_face_detection}")
    print(f"  face_detector = {processor.face_detector}")
    print(f"  Using RetinaFace: {'YES' if processor.face_detector is not None else 'NO'}")
    print()

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    print(f"Processing first {max_frames} frames...")
    print("=" * 80)

    # Process frames
    retinaface_detections = []
    frame_dimensions = []

    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame_dimensions.append((w, h))

        # Check if RetinaFace would be called
        if processor.skip_face_detection:
            # Should use full frame
            bbox = [0, 0, w, h, 1.0]
            detection_type = "FULL_FRAME"
        else:
            # Should call RetinaFace
            try:
                dets = processor.preprocess_image(frame)

                if dets is None or len(dets) == 0:
                    bbox = None
                    detection_type = "NO_FACE"
                else:
                    det = dets[0]
                    x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                    confidence = float(det[4]) if len(det) > 4 else 1.0
                    bbox = [x1, y1, x2, y2, confidence]
                    detection_type = "RETINAFACE"
            except Exception as e:
                bbox = None
                detection_type = f"ERROR: {str(e)}"

        retinaface_detections.append((detection_type, bbox))

        # Print every 5 frames
        if frame_idx % 5 == 0:
            if bbox is not None:
                x1, y1, x2, y2, conf = bbox
                bbox_area = (x2 - x1) * (y2 - y1)
                frame_area = w * h
                coverage = (bbox_area / frame_area) * 100

                print(f"Frame {frame_idx:3d}: {detection_type:12s} "
                      f"bbox=[{x1:4d}, {y1:4d}, {x2:4d}, {y2:4d}] "
                      f"conf={conf:.2f} coverage={coverage:5.1f}%")
            else:
                print(f"Frame {frame_idx:3d}: {detection_type:12s} (no bbox)")

    cap.release()

    # ============================================================================
    # Analysis
    # ============================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    detection_types = [dt for dt, _ in retinaface_detections]
    bbox_list = [bb for _, bb in retinaface_detections if bb is not None]

    print(f"\nDetection Summary:")
    print(f"  Total frames processed: {len(retinaface_detections)}")

    for dtype in set(detection_types):
        count = detection_types.count(dtype)
        percentage = (count / len(detection_types)) * 100
        print(f"  {dtype}: {count} frames ({percentage:.1f}%)")

    if bbox_list:
        print(f"\nBounding Box Statistics:")

        # Bbox sizes
        bbox_widths = [bb[2] - bb[0] for bb in bbox_list]
        bbox_heights = [bb[3] - bb[1] for bb in bbox_list]
        bbox_areas = [w * h for w, h in zip(bbox_widths, bbox_heights)]

        print(f"  Average bbox size: {np.mean(bbox_widths):.0f} x {np.mean(bbox_heights):.0f}")
        print(f"  Bbox width range: {np.min(bbox_widths):.0f} - {np.max(bbox_widths):.0f}")
        print(f"  Bbox height range: {np.min(bbox_heights):.0f} - {np.max(bbox_heights):.0f}")

        # Frame coverage
        w, h = frame_dimensions[0]
        frame_area = w * h
        coverages = [(area / frame_area) * 100 for area in bbox_areas]

        print(f"\n  Frame coverage:")
        print(f"    Average: {np.mean(coverages):.1f}%")
        print(f"    Range: {np.min(coverages):.1f}% - {np.max(coverages):.1f}%")

        # Confidence
        confidences = [bb[4] for bb in bbox_list]
        print(f"\n  Detection confidence:")
        print(f"    Average: {np.mean(confidences):.3f}")
        print(f"    Range: {np.min(confidences):.3f} - {np.max(confidences):.3f}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if "RETINAFACE" in detection_types:
        print("✗ PROBLEM FOUND: RetinaFace IS running on mirrored videos!")
        print()
        print("  This is causing AU extraction issues because:")
        print("  1. Mirrored videos are already face-aligned from the mirroring pipeline")
        print("  2. Running RetinaFace again crops the face incorrectly")
        print("  3. MTL receives poorly cropped faces → bad AU predictions")
        print()

        if bbox_list:
            avg_coverage = np.mean([(bb[2]-bb[0])*(bb[3]-bb[1]) / (frame_dimensions[0][0]*frame_dimensions[0][1]) * 100
                                   for bb in bbox_list])
            if avg_coverage < 80:
                print(f"  Average frame coverage is only {avg_coverage:.1f}%")
                print("  → RetinaFace is cropping out important facial regions!")

        print()
        print("SOLUTION:")
        print("  In openface_integration.py, line 1031, change:")
        print("    processor = OpenFace3Processor(device=device)")
        print("  to:")
        print("    processor = OpenFace3Processor(device=device, skip_face_detection=True)")
        print()

    elif "FULL_FRAME" in detection_types:
        print("✓ CORRECT: RetinaFace is disabled, using full frame")
        print("  This is the correct behavior for mirrored videos.")
    elif "NO_FACE" in detection_types:
        print("✗ PROBLEM FOUND: RetinaFace cannot detect faces!")
        print("  RetinaFace is running but failing to find faces.")
        print("  Solution: Use skip_face_detection=True for mirrored videos")
    else:
        print("⚠ UNKNOWN: Unexpected detection pattern")

    print("=" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simple RetinaFace diagnostic')
    parser.add_argument('video_path', type=str, help='Path to mirrored video file')
    parser.add_argument('--max-frames', type=int, default=30,
                       help='Number of frames to analyze (default: 30)')

    args = parser.parse_args()

    diagnose_retinaface(args.video_path, args.max_frames)
