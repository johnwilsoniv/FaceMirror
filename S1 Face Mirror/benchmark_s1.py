#!/usr/bin/env python3
"""
Benchmark S1 Face Mirror performance on a specific video

This script measures FPS and per-frame processing time for AU extraction
using the S1 Face Mirror OpenFace 3.0 pipeline.

Usage:
    python3 benchmark_s1.py --video <path> [--max-frames N]
"""

import argparse
import time
import sys
from pathlib import Path
import cv2

# Apply config settings
import config
config.apply_environment_settings()

from openface_integration import OpenFace3Processor


def benchmark_video(video_path, max_frames=None, num_threads=6):
    """
    Benchmark S1 Face Mirror AU extraction on a video

    Args:
        video_path: Path to input video
        max_frames: Maximum frames to process (None = all)
        num_threads: Number of processing threads

    Returns:
        dict with performance metrics
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return None

    print("=" * 80)
    print("S1 FACE MIRROR PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"Video: {video_path.name}")
    print(f"Threads: {num_threads}")
    print("")

    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"Video Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print("")

    # Initialize processor
    print("Initializing OpenFace 3.0 processor...")
    start_init = time.time()

    processor = OpenFace3Processor(
        device='cpu',  # Use CPU for consistent benchmarking
        weights_dir='weights',
        confidence_threshold=0.5,
        nms_threshold=0.4,
        calculate_landmarks=False,  # Don't need 98-point landmarks for this test
        num_threads=num_threads,
        debug_mode=False,
        skip_face_detection=False
    )

    init_time = time.time() - start_init
    print(f"✓ Initialized in {init_time:.2f}s")
    print("")

    # Create temporary output file
    output_csv = Path("/tmp/benchmark_s1_output.csv")

    # Process video
    print("Processing video...")
    print("-" * 80)

    def progress_callback(update):
        """Print progress updates"""
        if hasattr(update, 'current_frame') and hasattr(update, 'total_frames'):
            percent = (update.current_frame / update.total_frames) * 100
            fps_now = getattr(update, 'fps', 0)
            print(f"Frame {update.current_frame}/{update.total_frames} "
                  f"({percent:.1f}%) - {fps_now:.2f} FPS", end='\r')

    start_process = time.time()

    try:
        # Process the video
        processor.process_video(
            video_path=str(video_path),
            output_csv_path=str(output_csv),
            progress_callback=progress_callback
        )

        process_time = time.time() - start_process

        print()  # New line after progress
        print("-" * 80)
        print("")

        # Calculate metrics
        overall_fps = total_frames / process_time if process_time > 0 else 0
        ms_per_frame = (process_time / total_frames * 1000) if total_frames > 0 else 0

        # Results
        print("=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        print(f"Total frames processed: {total_frames}")
        print(f"Total time: {process_time:.2f}s")
        print(f"Overall FPS: {overall_fps:.2f}")
        print(f"Time per frame: {ms_per_frame:.1f}ms")
        print("")

        # Comparison to goals
        print("Performance vs Goals:")
        print(f"  Minimum goal (30 FPS): {'✓ ACHIEVED' if overall_fps >= 30 else f'✗ {overall_fps:.2f} FPS ({30/overall_fps:.2f}x slower)'}")
        print(f"  Stretch goal (50 FPS): {'✓ ACHIEVED' if overall_fps >= 50 else f'✗ {overall_fps:.2f} FPS ({50/overall_fps:.2f}x slower)'}")
        print("")

        # Clean up
        if output_csv.exists():
            output_csv.unlink()

        return {
            'total_frames': total_frames,
            'total_time': process_time,
            'overall_fps': overall_fps,
            'ms_per_frame': ms_per_frame
        }

    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark S1 Face Mirror performance",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--max-frames', type=int, default=200, help='Maximum frames to process (default: 200)')
    parser.add_argument('--threads', type=int, default=6, help='Number of processing threads (default: 6)')

    args = parser.parse_args()

    results = benchmark_video(
        video_path=args.video,
        max_frames=args.max_frames,
        num_threads=args.threads
    )

    if results:
        print("=" * 80)
        print("Benchmark complete!")
        print("=" * 80)
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
