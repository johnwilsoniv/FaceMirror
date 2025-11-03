#!/usr/bin/env python3
"""
Test full face mirroring pipeline on problematic patient videos.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory and pyfaceau to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector
from face_mirror import FaceMirror
import config_paths

def test_mirroring_pipeline(video_path, num_frames=30):
    """
    Test full mirroring pipeline on first N frames of video.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to process
    """
    print(f"\n{'='*70}")
    print(f"Testing full pipeline: {Path(video_path).name}")
    print(f"{'='*70}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"Processing first {num_frames} frames...\n")

    # Initialize detector and mirror
    detector = PyFaceAU68LandmarkDetector(
        debug_mode=False,
        skip_face_detection=False,
        use_clnf_refinement=True
    )

    mirror = FaceMirror(detector)

    # Process frames
    output_dir = Path("/tmp/pipeline_test")
    output_dir.mkdir(exist_ok=True)

    frames_processed = 0
    frames_failed = 0

    for frame_idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Get landmarks
        landmarks, _ = detector.get_face_mesh(frame)

        if landmarks is None:
            print(f"Frame {frame_idx}: ❌ No landmarks detected")
            frames_failed += 1
            continue

        # Create mirrored faces
        right_face, left_face = mirror.create_mirrored_faces(frame, landmarks)
        debug_frame = mirror.create_debug_frame(frame, landmarks)

        # Calculate head pose
        yaw = detector.calculate_head_pose(landmarks)
        yaw_str = f"{yaw:.1f}°" if yaw is not None else "N/A"

        print(f"Frame {frame_idx}: ✓ Landmarks detected (yaw: {yaw_str})")
        frames_processed += 1

        # Save some debug frames
        if frame_idx in [0, 5, 10, 15, 20]:
            base_name = Path(video_path).stem
            cv2.imwrite(str(output_dir / f"{base_name}_frame{frame_idx:03d}_debug.jpg"), debug_frame)
            cv2.imwrite(str(output_dir / f"{base_name}_frame{frame_idx:03d}_right.jpg"), right_face)
            cv2.imwrite(str(output_dir / f"{base_name}_frame{frame_idx:03d}_left.jpg"), left_face)

    cap.release()

    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Processed: {frames_processed}/{num_frames} frames")
    print(f"  Failed: {frames_failed}/{num_frames} frames")
    print(f"  Debug frames saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Test both problematic videos
    test_videos = [
        "/Users/johnwilsoniv/Documents/SplitFace Open3/D Facial Paralysis Pts/IMG_8401.MOV",
        "/Users/johnwilsoniv/Documents/SplitFace Open3/D Facial Paralysis Pts/IMG_9330.MOV"
    ]

    for video_path in test_videos:
        if Path(video_path).exists():
            test_mirroring_pipeline(video_path, num_frames=30)
        else:
            print(f"WARNING: Video not found: {video_path}")
