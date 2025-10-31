#!/usr/bin/env python3
"""
Simple test of OpenFace 3.0 ONNX pipeline on a single video
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from openface_integration import OpenFace3Processor

video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/D Normal Pts/IMG_0942.MOV"
output_csv = Path.home() / "Desktop/IMG_0942_ONNX_test.csv"

print("Initializing OpenFace 3.0 with ONNX...")
processor = OpenFace3Processor(
    device='cpu',
    calculate_landmarks=True,
    num_threads=6,
    debug_mode=True,
    skip_face_detection=False
)

print(f"\nProcessing: {video_path}")
frame_count = processor.process_video(video_path, output_csv)
print(f"\n✓ Processed {frame_count} frames")
print(f"✓ Output: {output_csv}")
