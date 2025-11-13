#!/usr/bin/env python3
"""
Debug ONet scores to see why all faces are being filtered out.
"""

import cv2
import numpy as np
from pure_python_mtcnn_detector import PurePythonMTCNNDetector

# Load image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')

# Create detector with VERY LOW thresholds to see raw scores
detector = PurePythonMTCNNDetector()
detector.onet_threshold = 0.0  # Accept everything to see raw scores

print("="*80)
print("DEBUGGING ONET SCORES")
print("="*80)

# Temporarily patch the detect method to print ONet scores
orig_detect = detector.detect

def debug_detect(img, debug=False):
    """Patched detect that prints ONet scores"""
    img_h, img_w = img.shape[:2]
    img_float = img.astype(np.float32)

    # ... all the PNet/RNet code would go here ...
    # For now, let's just manually run ONet on a cropped face region

    # Use a rough estimate of where the face should be (from previous tests)
    # C++ MTCNN detected: x=331.6, y=753.5, w=367.9, h=422.8
    x, y, w, h = 331, 753, 368, 423

    # Extract and resize to 48x48 for ONet
    face = img_float[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face_data = detector._preprocess(face)

    print(f"\nTesting ONet on face region: x={x}, y={y}, w={w}, h={h}")
    print(f"Face data shape: {face_data.shape}")
    print(f"Face data range: [{face_data.min():.3f}, {face_data.max():.3f}]")

    # Run ONet
    output = detector.onet(face_data)
    output = output[-1]  # Take last layer

    print(f"\nONet output shape: {output.shape}")
    print(f"ONet output: {output}")

    # Calculate score
    logit_not_face = output[0]
    logit_face = output[1]
    score = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

    print(f"\nLogit not-face: {logit_not_face:.6f}")
    print(f"Logit face: {logit_face:.6f}")
    print(f"Difference (not-face - face): {logit_not_face - logit_face:.6f}")
    print(f"Score (sigmoid): {score:.6f}")
    print(f"ONet threshold: {detector.onet_threshold}")
    print(f"Pass threshold: {score > detector.onet_threshold}")

    # Show output breakdown
    print(f"\nOutput breakdown:")
    print(f"  [0:2] Classification logits: {output[0:2]}")
    print(f"  [2:6] Bbox regression: {output[2:6]}")
    print(f"  [6:16] Landmark coords: {output[6:16]}")

    return np.empty((0, 4)), np.empty((0, 5, 2))

detector.detect = debug_detect
detector.detect(img)

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("\nIf score < 0.7 (default threshold), ONet filters it out.")
print("This could be due to:")
print("  1. Weight loading issue (weights corrupted/wrong order)")
print("  2. Input preprocessing mismatch")
print("  3. Layer computation error")
print("  4. Output interpretation wrong (maybe logits are flipped?)")
