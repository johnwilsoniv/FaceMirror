#!/usr/bin/env python3
"""
Demonstrate validation warnings that will appear during S1 processing.
"""

import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector

test_videos = [
    ("IMG_8401_source.MOV", "PROBLEM - Surgical markings"),
    ("IMG_9330_source.MOV", "PROBLEM - Severe paralysis"),
    ("IMG_0434_source.MOV", "GOOD - Normal cohort"),
    ("IMG_0942_source.MOV", "GOOD - Normal cohort"),
]

base_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/"

print("="*80)
print("VALIDATION WARNING DEMONSTRATION")
print("="*80)
print()
print("Simulating what users will see during S1 Face Mirror processing:")
print()

detector = PyFaceAU68LandmarkDetector(debug_mode=False, use_clnf_refinement=True)

for video_name, description in test_videos:
    print(f"\n{'='*80}")
    print(f"Video: {video_name}")
    print(f"Category: {description}")
    print(f"{'='*80}\n")

    video_path = base_path + video_name

    # Simulate processing frames 0, 300, 600 (where warnings print)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in [0, 300, 600]:
        if frame_idx >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # This simulates what happens in video_processor.py
        landmarks, validation_info = detector.get_face_mesh(frame)

        if validation_info is not None:
            used_fallback = validation_info.get('used_fallback', False)
            validation_failed = not validation_info.get('validation_passed', True)

            # This is the exact warning logic from video_processor.py
            if validation_failed or used_fallback:
                reason = validation_info.get('reason', 'Unknown')

                if used_fallback:
                    print(f"⚠️  WARNING [Frame {frame_idx}]: Primary face detector failed, using MTCNN fallback")
                    print(f"    Primary detector failure reason: {reason}")
                    print(f"    Results may be less accurate. Consider reviewing this video manually.\n")
                else:
                    print(f"⚠️  WARNING [Frame {frame_idx}]: Landmark detection validation failed")
                    print(f"    Reason: {reason}")
                    print(f"    Processing will continue but results may be inaccurate.\n")
                sys.stdout.flush()

        detector.reset_tracking_history()

    cap.release()

    if used_fallback or validation_failed:
        print(f"→ This video would show warnings every 10 seconds during processing")
    else:
        print(f"→ This video processes silently with no warnings")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}\n")
print("Problematic videos (IMG_8401, IMG_9330):")
print("  - Show warnings at frames 0, 300, 600, ... (every 10 seconds)")
print("  - Indicate MTCNN fallback was used")
print("  - Advise user to review manually")
print()
print("Normal videos (IMG_0434, IMG_0942):")
print("  - Process silently with no warnings")
print("  - Use primary RetinaFace detector successfully")
print()
