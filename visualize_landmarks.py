#!/usr/bin/env python3
"""Visualize landmarks on a few frames to verify quality."""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector

video_path = Path(__file__).parent / "Patient Data" / "Normal Cohort" / "IMG_0942.MOV"
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)

print("Initializing detector...")
detector = PyFaceAU68LandmarkDetector(
    debug_mode=False,
    use_clnf_refinement=True,
    skip_redetection=True
)

# Test frames: 0, 50, 100
test_frames = [0, 50, 100]

cap = cv2.VideoCapture(str(video_path))

for frame_idx in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        continue

    # Detect landmarks
    landmarks, _ = detector.get_face_mesh(frame)

    if landmarks is not None:
        # Draw landmarks
        vis = frame.copy()
        for i, (x, y) in enumerate(landmarks):
            # Different colors for different regions
            if i < 17:  # Jaw
                color = (0, 255, 0)  # Green
            elif i < 27:  # Eyebrows
                color = (255, 0, 0)  # Blue
            elif i < 36:  # Nose
                color = (0, 255, 255)  # Yellow
            elif i < 48:  # Eyes
                color = (255, 0, 255)  # Magenta
            else:  # Mouth
                color = (0, 128, 255)  # Orange

            cv2.circle(vis, (int(x), int(y)), 3, color, -1)

        # Get midline
        glabella, chin = detector.get_facial_midline(landmarks)

        # Draw midline
        cv2.line(vis, (int(glabella[0]), int(glabella[1])),
                (int(chin[0]), int(chin[1])), (0, 0, 255), 2)

        # Save
        output_path = output_dir / f"landmarks_frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(output_path), vis)
        print(f"✓ Frame {frame_idx}: {len(landmarks)} landmarks detected")
        print(f"  Saved to: {output_path}")

cap.release()
print("\nVisualization complete!")
print(f"Check {output_dir} for results")
