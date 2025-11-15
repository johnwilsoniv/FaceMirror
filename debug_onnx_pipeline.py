"""
Debug ONNX Pipeline - Trace Stage-by-Stage to Find Divergence

Compares intermediate outputs between CoreML (working) and ONNX (2 detections vs 1)
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

from pymtcnn.backends.coreml_backend import CoreMLMTCNN
from pymtcnn.backends.onnx_backend import ONNXMTCNN

TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"

print("="*80)
print("ONNX Pipeline Debug - Find Divergence Point")
print("="*80)

# Load image
img = cv2.imread(TEST_IMAGE)
print(f"\nTest image: {img.shape}")

# Initialize both detectors
print("\nInitializing detectors...")
coreml_detector = CoreMLMTCNN(verbose=False)
onnx_detector = ONNXMTCNN(verbose=False)

# Compare parameters
print("\n" + "="*80)
print("MTCNN Parameters")
print("="*80)
print(f"CoreML thresholds: {coreml_detector.thresholds}")
print(f"ONNX thresholds:   {onnx_detector.thresholds}")
print(f"CoreML min_face_size: {coreml_detector.min_face_size}")
print(f"ONNX min_face_size:   {onnx_detector.min_face_size}")
print(f"CoreML factor: {coreml_detector.factor}")
print(f"ONNX factor:   {onnx_detector.factor}")

# Test PNet on first scale
print("\n" + "="*80)
print("Stage 1: PNet - First Scale")
print("="*80)

img_h, img_w = img.shape[:2]
img_float = img.astype(np.float32)
min_face_size = 60
m = 12.0 / min_face_size
scale = m

hs = int(np.ceil(img_h * scale))
ws = int(np.ceil(img_w * scale))

print(f"\nScale: {scale:.6f}, Scaled size: {hs}x{ws}")

# Preprocess
img_scaled = cv2.resize(img_float, (ws, hs), interpolation=cv2.INTER_LINEAR)
img_data_coreml = coreml_detector._preprocess(img_scaled, flip_bgr_to_rgb=True)
img_data_onnx = onnx_detector._preprocess(img_scaled, flip_bgr_to_rgb=True)

print(f"CoreML preprocessed: shape={img_data_coreml.shape}, range=[{img_data_coreml.min():.4f}, {img_data_coreml.max():.4f}]")
print(f"ONNX preprocessed:   shape={img_data_onnx.shape}, range=[{img_data_onnx.min():.4f}, {img_data_onnx.max():.4f}]")

# Check if preprocessing is identical
preprocess_diff = np.abs(img_data_coreml - img_data_onnx).max()
print(f"Preprocessing diff: {preprocess_diff:.10f} {'✓ IDENTICAL' if preprocess_diff < 1e-6 else '✗ DIFFERENT'}")

# Run PNet
coreml_pnet_out = coreml_detector._run_pnet(img_data_coreml)
onnx_pnet_out = onnx_detector._run_pnet(img_data_onnx)

print(f"\nCoreML PNet output: shape={coreml_pnet_out.shape}, range=[{coreml_pnet_out.min():.6f}, {coreml_pnet_out.max():.6f}]")
print(f"ONNX PNet output:   shape={onnx_pnet_out.shape}, range=[{onnx_pnet_out.min():.6f}, {onnx_pnet_out.max():.6f}]")

# Compare PNet outputs
pnet_diff = np.abs(coreml_pnet_out - onnx_pnet_out).max()
print(f"PNet output diff: {pnet_diff:.10f} {'✓ IDENTICAL' if pnet_diff < 1e-5 else '✗ DIFFERENT'}")

# Extract logits and compute probabilities
coreml_pnet_chw = coreml_pnet_out[0].transpose(1, 2, 0)
onnx_pnet_chw = onnx_pnet_out[0].transpose(1, 2, 0)

coreml_logit_face = coreml_pnet_chw[:, :, 1]
onnx_logit_face = onnx_pnet_chw[:, :, 1]

coreml_prob_face = 1.0 / (1.0 + np.exp(coreml_pnet_chw[:, :, 0] - coreml_pnet_chw[:, :, 1]))
onnx_prob_face = 1.0 / (1.0 + np.exp(onnx_pnet_chw[:, :, 0] - onnx_pnet_chw[:, :, 1]))

print(f"\nCoreML face prob: range=[{coreml_prob_face.min():.6f}, {coreml_prob_face.max():.6f}], >0.6: {(coreml_prob_face > 0.6).sum()}")
print(f"ONNX face prob:   range=[{onnx_prob_face.min():.6f}, {onnx_prob_face.max():.6f}], >0.6: {(onnx_prob_face > 0.6).sum()}")

# Full pipeline with debug info
print("\n" + "="*80)
print("Full Pipeline Comparison")
print("="*80)

print("\nRunning CoreML with debug...")
coreml_bboxes, coreml_landmarks, coreml_debug = coreml_detector.detect_with_debug(img)

print(f"\nCoreML Results:")
print(f"  PNet: {coreml_debug['pnet']['num_boxes']} boxes")
print(f"  RNet: {coreml_debug['rnet']['num_boxes']} boxes")
print(f"  ONet: {coreml_debug['onet']['num_boxes']} boxes")
print(f"  Final: {coreml_debug['final']['num_boxes']} boxes")

print("\nRunning ONNX (regular detect)...")
onnx_bboxes, onnx_landmarks = onnx_detector.detect(img)

print(f"\nONNX Results:")
print(f"  Final: {len(onnx_bboxes)} boxes")

if len(onnx_bboxes) > 0:
    print(f"\nONNX Bounding Boxes:")
    for i, bbox in enumerate(onnx_bboxes):
        print(f"  Box {i}: x={bbox[0]:.1f}, y={bbox[1]:.1f}, w={bbox[2]:.1f}, h={bbox[3]:.1f}")

if len(coreml_bboxes) > 0:
    print(f"\nCoreML Bounding Boxes:")
    for i, bbox in enumerate(coreml_bboxes):
        print(f"  Box {i}: x={bbox[0]:.1f}, y={bbox[1]:.1f}, w={bbox[2]:.1f}, h={bbox[3]:.1f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if len(onnx_bboxes) == len(coreml_bboxes):
    print("✓ Both backends detect the same number of faces")
else:
    print(f"✗ ONNX detects {len(onnx_bboxes)} faces, CoreML detects {len(coreml_bboxes)} faces")
    print("   → Need to trace through RNet and ONet stages to find divergence")
