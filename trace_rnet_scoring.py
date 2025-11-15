"""
Trace RNet Scoring - Check for Threshold Issues

Even though divergence starts at PNet, user said "issue was in RNet".
Let's check RNet scores to see if there's a threshold problem there too.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

from pymtcnn.backends.coreml_backend import CoreMLMTCNN
from pymtcnn.backends.onnx_backend import ONNXMTCNN

TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"

class DebugCoreML(CoreMLMTCNN):
    """CoreML with RNet score capture"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rnet_scores = []

    def _run_rnet(self, img_data):
        output = super()._run_rnet(img_data)
        # output is (6,): [not_face_logit, face_logit, dx1, dy1, dx2, dy2]
        face_prob = 1.0 / (1.0 + np.exp(output[0] - output[1]))
        self.rnet_scores.append(face_prob)
        return output

class DebugONNX(ONNXMTCNN):
    """ONNX with RNet score capture"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rnet_scores = []

    def _run_rnet(self, img_data):
        output = super()._run_rnet(img_data)
        # output is (6,): [not_face_logit, face_logit, dx1, dy1, dx2, dy2]
        face_prob = 1.0 / (1.0 + np.exp(output[0] - output[1]))
        self.rnet_scores.append(face_prob)
        return output

print("="*80)
print("RNet Scoring Analysis - Find Threshold Issues")
print("="*80)

# Load image
img = cv2.imread(TEST_IMAGE)

# Initialize debug detectors
print("\nInitializing debug detectors...")
coreml_detector = DebugCoreML(verbose=False)
onnx_detector = DebugONNX(verbose=False)

# Run detection
print("Running CoreML...")
coreml_bboxes, coreml_landmarks = coreml_detector.detect(img)

print("Running ONNX...")
onnx_bboxes, onnx_landmarks = onnx_detector.detect(img)

# Analyze RNet scores
print("\n" + "="*80)
print("RNet Score Distribution")
print("="*80)

coreml_scores = np.array(coreml_detector.rnet_scores)
onnx_scores = np.array(onnx_detector.rnet_scores)

print(f"\nCoreML RNet:")
print(f"  Total boxes tested: {len(coreml_scores)}")
print(f"  Scores: min={coreml_scores.min():.6f}, max={coreml_scores.max():.6f}, mean={coreml_scores.mean():.6f}")
print(f"  Passing (>0.7): {(coreml_scores > 0.7).sum()}")
print(f"  Borderline (0.68-0.72): {((coreml_scores > 0.68) & (coreml_scores < 0.72)).sum()}")

print(f"\nONNX RNet:")
print(f"  Total boxes tested: {len(onnx_scores)}")
print(f"  Scores: min={onnx_scores.min():.6f}, max={onnx_scores.max():.6f}, mean={onnx_scores.mean():.6f}")
print(f"  Passing (>0.7): {(onnx_scores > 0.7).sum()}")
print(f"  Borderline (0.68-0.72): {((onnx_scores > 0.68) & (onnx_scores < 0.72)).sum()}")

# Check for borderline scores
print("\n" + "="*80)
print("Borderline RNet Scores (0.65-0.75)")
print("="*80)

coreml_borderline = coreml_scores[(coreml_scores > 0.65) & (coreml_scores < 0.75)]
onnx_borderline = onnx_scores[(onnx_scores > 0.65) & (onnx_scores < 0.75)]

if len(coreml_borderline) > 0:
    print(f"\nCoreML borderline scores:")
    for i, score in enumerate(sorted(coreml_borderline)):
        status = "PASS" if score > 0.7 else "FAIL"
        print(f"  Box {i}: {score:.6f} ({status})")

if len(onnx_borderline) > 0:
    print(f"\nONNX borderline scores:")
    for i, score in enumerate(sorted(onnx_borderline)):
        status = "PASS" if score > 0.7 else "FAIL"
        print(f"  Box {i}: {score:.6f} ({status})")

# Check scores near threshold that flip
print("\n" + "="*80)
print("Scores Within 0.02 of Threshold (0.68-0.72)")
print("="*80)

coreml_near_threshold = coreml_scores[(coreml_scores > 0.68) & (coreml_scores < 0.72)]
onnx_near_threshold = onnx_scores[(onnx_scores > 0.68) & (onnx_scores < 0.72)]

print(f"\nCoreML: {len(coreml_near_threshold)} boxes near threshold")
if len(coreml_near_threshold) > 0:
    for i, score in enumerate(sorted(coreml_near_threshold)):
        status = "PASS" if score > 0.7 else "FAIL"
        margin = abs(score - 0.7)
        print(f"  Box {i}: {score:.6f} ({status}) - margin: {margin:.6f}")

print(f"\nONNX: {len(onnx_near_threshold)} boxes near threshold")
if len(onnx_near_threshold) > 0:
    for i, score in enumerate(sorted(onnx_near_threshold)):
        status = "PASS" if score > 0.7 else "FAIL"
        margin = abs(score - 0.7)
        print(f"  Box {i}: {score:.6f} ({status}) - margin: {margin:.6f}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

print(f"""
RNet Testing Summary:
  CoreML: {len(coreml_scores)} boxes tested, {(coreml_scores > 0.7).sum()} passed
  ONNX:   {len(onnx_scores)} boxes tested, {(onnx_scores > 0.7).sum()} passed

Borderline Cases:
  CoreML: {len(coreml_borderline)} boxes in 0.65-0.75 range
  ONNX:   {len(onnx_borderline)} boxes in 0.65-0.75 range

Critical Finding:
  Different number of PNet boxes ({len(coreml_scores)} vs {len(onnx_scores)}) means
  RNet is testing DIFFERENT boxes, not the same boxes with different scores.

If we find scores very close to 0.7 threshold (within 0.01):
  → RNet threshold sensitivity is contributing to the divergence
  → Small model differences cause boxes to flip between pass/fail

This compounds the PNet divergence:
  1. PNet produces different boxes (99 vs 100)
  2. RNet filters them with scores near threshold
  3. Small score differences cause different filtering
  4. Different boxes reach ONet
  5. ONet produces 2 boxes for same face
  6. NMS doesn't merge them (IoU=0.36 < 0.7)
""")

print("\n" + "="*80)
print("Final Results")
print("="*80)
print(f"\nCoreML: {len(coreml_bboxes)} face(s) detected")
print(f"ONNX:   {len(onnx_bboxes)} face(s) detected")

if len(onnx_bboxes) > len(coreml_bboxes):
    print(f"\n✗ ONNX over-detected: {len(onnx_bboxes)} faces instead of {len(coreml_bboxes)}")
    print("  → This is the bug we need to fix")
