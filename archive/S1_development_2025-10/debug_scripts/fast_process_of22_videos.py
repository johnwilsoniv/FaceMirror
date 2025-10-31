#!/usr/bin/env python3
"""
Fast processing using the optimized OpenFace3Processor with batching and threading
"""

import config
config.apply_environment_settings()

from pathlib import Path
from openface_integration import OpenFace3Processor

def main():
    # Input videos (mirrored by OF2.2)
    videos = [
        ("/Users/johnwilsoniv/Documents/open2GR/1_Face_Mirror/output/IMG_0942_left_mirrored.mp4",
         "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/OP3 v OP 22/IMG_0942_left_mirroredOP22_processedONNXv3.csv"),
        ("/Users/johnwilsoniv/Documents/open2GR/1_Face_Mirror/output/IMG_0942_right_mirrored.mp4",
         "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/OP3 v OP 22/IMG_0942_right_mirroredOP22_processedONNXv3.csv")
    ]

    print("=" * 80)
    print("Fast OF3.0 Processing (with batching + threading)")
    print("=" * 80)

    # Initialize processor ONCE (reuse for both videos)
    print("\nInitializing optimized OpenFace 3.0 processor...")
    print("  Mode: skip_face_detection=True (videos already face-aligned)")
    processor = OpenFace3Processor(
        device='cpu',
        calculate_landmarks=False,
        num_threads=6,  # Parallel processing
        debug_mode=False,
        skip_face_detection=True  # CRITICAL: OF2.2 videos are already face-cropped!
    )
    print("✓ Processor initialized\n")

    # Process both videos
    for video_path, csv_path in videos:
        video_path = Path(video_path)
        csv_path = Path(csv_path)

        print(f"\nProcessing: {video_path.name}")
        print(f"Output: {csv_path.name}")

        def progress(current, total, fps):
            if current % 100 == 0 or current == total:
                pct = (current / total * 100) if total > 0 else 0
                print(f"  Frame {current:>4}/{total} ({pct:>5.1f}%) @ {fps:.1f} fps")

        try:
            frame_count = processor.process_video(
                video_path,
                csv_path,
                progress_callback=progress
            )
            print(f"✓ Processed {frame_count} frames\n")
        except Exception as e:
            print(f"✗ Error: {e}\n")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
