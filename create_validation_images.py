#!/usr/bin/env python3
"""
Create validation images showing which detector was used (RetinaFace vs MTCNN).
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector

test_videos = [
    ("IMG_8401_source", "Paralysis - Surgical Markings", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV"),
    ("IMG_9330_source", "Paralysis - Severe", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV"),
    ("IMG_0434_source", "Normal Cohort", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0434_source.MOV"),
    ("IMG_0942_source", "Normal Cohort", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0942_source.MOV"),
]

print("="*80)
print("CREATING VALIDATION IMAGES - SHOWING DETECTOR USED")
print("="*80)
print()

output_dir = Path("test_output")
output_dir.mkdir(exist_ok=True)

# Track which videos used MTCNN
mtcnn_used = []
retinaface_used = []

for name, category, path in test_videos:
    print(f"\nProcessing: {name} ({category})")

    # Create a custom detector with debug mode to see which detector is used
    class DebugDetector(PyFaceAU68LandmarkDetector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.used_mtcnn = False

        def get_face_mesh(self, frame, detection_interval=2):
            # Call parent method
            result = super().get_face_mesh(frame, detection_interval)
            return result

    detector = DebugDetector(debug_mode=True, use_clnf_refinement=True)

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"  ❌ Failed to read frame")
        continue

    # Capture stdout to detect which detector was used
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        landmarks, _ = detector.get_face_mesh(frame)

    output = f.getvalue()
    used_mtcnn_fallback = "MTCNN fallback succeeded" in output

    bbox = detector.cached_bbox.copy() if detector.cached_bbox is not None else None
    detector.reset_tracking_history()

    if landmarks is None or bbox is None:
        print(f"  ❌ No detection")
        continue

    # Track which detector was used
    detector_name = "MTCNN (fallback)" if used_mtcnn_fallback else "RetinaFace"
    if used_mtcnn_fallback:
        mtcnn_used.append(name)
    else:
        retinaface_used.append(name)

    # Validate
    temp_det = PyFaceAU68LandmarkDetector(debug_mode=False, use_clnf_refinement=False)
    is_valid, reason, confidence = temp_det._validate_landmarks(landmarks, bbox, frame.shape)

    # Create visualization
    vis = frame.copy()

    # Draw bbox (green for valid)
    bbox_color = (0, 255, 0) if is_valid else (0, 0, 255)
    cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, 3)

    # Draw all 68 landmarks (blue dots)
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(vis, (int(x), int(y)), 3, (255, 0, 0), -1)

    # Add text annotations
    # Line 1: Video name + detector used
    detector_color = (0, 165, 255) if used_mtcnn_fallback else (0, 255, 0)  # Orange for MTCNN, green for RetinaFace
    cv2.putText(vis, f"{name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Line 2: Detector used
    cv2.putText(vis, f"Detector: {detector_name}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, detector_color, 2)

    # Line 3: Validation result
    status_text = f"Valid (conf={confidence:.2f})" if is_valid else f"{reason} (conf={confidence:.2f})"
    cv2.putText(vis, status_text, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

    # Save
    output_file = output_dir / f"validation_{name}.jpg"
    cv2.imwrite(str(output_file), vis)
    print(f"  ✅ Saved: {output_file}")
    print(f"     Detector: {detector_name}")
    print(f"     Status: {status_text}")

print()
print("="*80)
print("SUMMARY")
print("="*80)
print()
print(f"RetinaFace (primary detector):")
for name in retinaface_used:
    print(f"  ✅ {name}")
print()
print(f"MTCNN (fallback detector):")
for name in mtcnn_used:
    print(f"  🔄 {name}")
print()
print("All visualizations saved to: test_output/")
print()
