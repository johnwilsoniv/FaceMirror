#!/usr/bin/env python3
"""
Test script to generate Python CCNF debug output for comparison with C++.
"""

import sys
import numpy as np
import cv2

# Add paths
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))

from pymtcnn import MTCNN
from pyclnf import CLNF

def main():
    # Load the same image used for C++ testing
    image_path = "/tmp/test_frame.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return

    print(f"Image shape: {image.shape}")

    # Initialize face detector and CLNF with eye refinement enabled
    detector = MTCNN()
    model_dir = "pyclnf/models"
    clnf = CLNF(model_dir, use_eye_refinement=True)

    # Detect face
    bboxes, landmarks = detector.detect(image)
    if bboxes is None or len(bboxes) == 0:
        print("No faces detected!")
        return

    # Use first detection (bboxes format: x1, y1, x2, y2, score)
    face_bbox = bboxes[0, :4].astype(np.float32)

    # Reset debug call counter
    if hasattr(clnf, 'eye_model') and clnf.eye_model is not None:
        clnf.eye_model._ccnf_debug_call = 0

    # Detect landmarks
    print(f"Face bbox: {face_bbox}")
    print("Detecting landmarks...")
    landmarks, info = clnf.fit(image, face_bbox)

    if landmarks is not None:
        print(f"Detected {len(landmarks)} landmarks")

        # Check if debug files were created
        import os
        for i in range(5):
            debug_file = f'/tmp/python_ccnf_neuron_debug_call{i}.txt'
            if os.path.exists(debug_file):
                print(f"Created: {debug_file}")

        # Print some key landmarks
        print("\nEye landmarks (36-41, 42-47):")
        for i in range(36, 48):
            print(f"  {i}: ({landmarks[i, 0]:.4f}, {landmarks[i, 1]:.4f})")
    else:
        print("No landmarks detected!")

if __name__ == "__main__":
    main()
