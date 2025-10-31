#!/usr/bin/env python3
"""Single-threaded test to avoid deadlocks"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from openface_integration import OpenFace3Processor

video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
output_csv = "./test_output_fixed/IMG_0942_left_mirrored.csv"

print("Processing with single-threaded mode...")

processor = OpenFace3Processor(
    device='cpu',
    skip_face_detection=True,
    num_threads=1,  # Single-threaded to avoid deadlocks
    debug_mode=False
)

print(f"Config: skip_face_detection={processor.skip_face_detection}, threads={processor.num_threads}")

processor.process_video(video_path, output_csv)
print(f"\n✓ Complete! CSV saved to: {output_csv}")
