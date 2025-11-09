#!/usr/bin/env python3
"""
Test Python pipeline on problematic videos showing validation warnings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from video_processor import VideoProcessor
from pyfaceau_detector import PyFaceAU68LandmarkDetector
from face_splitter import StableFaceSplitter

test_videos = [
    ("IMG_8401_source.MOV", "PROBLEM - Surgical markings"),
    ("IMG_9330_source.MOV", "PROBLEM - Severe paralysis"),
    ("IMG_0434_source.MOV", "GOOD - Normal cohort"),
]

base_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/"
output_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/test_output/pipeline_test")
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PYTHON PIPELINE TEST WITH VALIDATION WARNINGS")
print("="*80)
print()
print("This will process the first 100 frames of each video to demonstrate")
print("validation warnings for problematic videos.")
print()

for video_name, description in test_videos:
    print(f"\n{'='*80}")
    print(f"Processing: {video_name}")
    print(f"Description: {description}")
    print(f"{'='*80}\n")

    video_path = base_path + video_name
    video_output_dir = output_dir / Path(video_name).stem

    # Initialize detector and processor
    detector = PyFaceAU68LandmarkDetector(debug_mode=False, use_clnf_refinement=True)
    face_mirror = StableFaceSplitter(debug_mode=False)
    processor = VideoProcessor(
        landmark_detector=detector,
        face_mirror=face_mirror,
        debug_mode=False,
        num_threads=4
    )

    try:
        # Process just the first 100 frames to show warnings
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        frames_to_process = min(100, total_frames)

        print(f"Processing {frames_to_process} frames (out of {total_frames} total)")
        print(f"FPS: {fps:.2f}")
        print()

        # This will show validation warnings if any
        processor.process_video(video_path, video_output_dir)

        print(f"\n✅ Completed: {video_name}")

    except Exception as e:
        print(f"\n❌ Error processing {video_name}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("PIPELINE TEST COMPLETE")
print(f"{'='*80}\n")
print(f"Output saved to: {output_dir.absolute()}")
print()
