#!/usr/bin/env python3
"""
Visualize ALL RetinaFace detections to see which is the correct face.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))
from pyfaceau.detectors.retinaface import ONNXRetinaFaceDetector

# Initialize RetinaFace
detector = ONNXRetinaFaceDetector(
    str(Path(__file__).parent / 'S1 Face Mirror/weights/retinaface_mobilenet025_coreml.onnx'),
    use_coreml=True,
    confidence_threshold=0.5,
    nms_threshold=0.4
)

output_dir = Path(__file__).parent / "test_output" / "all_detections"
output_dir.mkdir(parents=True, exist_ok=True)

test_videos = [
    ("IMG_0942_source", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0942_source.MOV"),
    ("IMG_8401_source", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV"),
    ("IMG_9330_source", "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV"),
]

colors = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
]

for name, path in test_videos:
    print(f"Processing {name}...")

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    # Detect all faces
    detections, landmarks = detector.detect_faces(frame)

    if detections is not None and len(detections) > 0:
        vis = frame.copy()

        # Draw all detections with different colors
        for i, det in enumerate(detections):
            bbox = det[:4].astype(int)
            conf = det[4]
            color = colors[i % len(colors)]

            # Draw bbox
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 4)

            # Label
            label = f"Det {i}: conf={conf:.3f}"
            cv2.putText(vis, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

            # Draw size
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            size_label = f"{bbox_w}x{bbox_h}px"
            cv2.putText(vis, size_label, (bbox[0], bbox[3] + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Title
        title = f"{name}: {len(detections)} face(s) detected"
        cv2.putText(vis, title, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
        cv2.putText(vis, title, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        # Save
        output_path = output_dir / f"{name}_all_detections.jpg"
        cv2.imwrite(str(output_path), vis)
        print(f"  Saved: {output_path}")
    else:
        print(f"  No faces detected!")

print("\nDone! Check test_output/all_detections/")
