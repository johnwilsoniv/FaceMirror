#!/usr/bin/env python3
"""
Debug visualization to see what's actually being detected.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector

# Test on IMG_8401_source
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV"
output_dir = Path(__file__).parent / "test_output" / "debug"
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading frame 50 from IMG_8401_source...")
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

print(f"Frame shape: {frame.shape}")

# Initialize detector
print("Initializing detector...")
detector = PyFaceAU68LandmarkDetector(
    debug_mode=False,
    use_clnf_refinement=True,
    skip_redetection=False
)

# Detect
print("Detecting landmarks...")
landmarks, _ = detector.get_face_mesh(frame)

if landmarks is None:
    print("ERROR: No landmarks detected!")
    sys.exit(1)

print(f"Got {len(landmarks)} landmarks")
print(f"Bbox: {detector.cached_bbox}")

# Create visualization
vis = frame.copy()

# Draw bbox
if detector.cached_bbox is not None:
    x1, y1, x2, y2 = [int(v) for v in detector.cached_bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 3)
    cv2.putText(vis, f"Bbox: [{x1},{y1}] to [{x2},{y2}]", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

# Draw landmarks with colors
for i, (x, y) in enumerate(landmarks):
    if i < 17:  # Jaw
        color = (0, 255, 0)
    elif i < 27:  # Eyebrows
        color = (255, 0, 0)
    elif i < 36:  # Nose
        color = (0, 255, 255)
    elif i < 48:  # Eyes
        color = (255, 0, 255)
    else:  # Mouth
        color = (0, 128, 255)

    cv2.circle(vis, (int(x), int(y)), 5, color, -1)

    # Label some key points
    if i in [0, 8, 16, 27, 30, 33, 36, 39, 42, 45, 48, 54]:
        cv2.putText(vis, str(i), (int(x)+7, int(y)-7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Midline
glabella, chin = detector.get_facial_midline(landmarks)
if glabella is not None and chin is not None:
    cv2.line(vis, (int(glabella[0]), int(glabella[1])),
            (int(chin[0]), int(chin[1])), (0, 0, 255), 3)

# Info
x_range = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
y_range = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])

info_text = [
    "DEBUG: IMG_8401_source Frame 50",
    f"Spread: {x_range:.0f}x{y_range:.0f}px",
    f"Landmarks: {len(landmarks)} points"
]

y_offset = 40
for text in info_text:
    cv2.putText(vis, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (255, 255, 255), 3)
    cv2.putText(vis, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (0, 0, 0), 1)
    y_offset += 35

# Print landmark coordinates
print("\nFirst 10 landmarks (x, y):")
for i in range(10):
    print(f"  {i}: ({landmarks[i][0]:.0f}, {landmarks[i][1]:.0f})")

print(f"\nLandmark range:")
print(f"  X: [{np.min(landmarks[:, 0]):.0f}, {np.max(landmarks[:, 0]):.0f}]")
print(f"  Y: [{np.min(landmarks[:, 1]):.0f}, {np.max(landmarks[:, 1]):.0f}]")

# Save
output_path = output_dir / "IMG_8401_debug.jpg"
cv2.imwrite(str(output_path), vis)
print(f"\nSaved: {output_path}")
