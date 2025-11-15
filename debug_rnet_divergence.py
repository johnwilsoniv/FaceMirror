"""
Debug RNet Stage Divergence - Find Why ONNX=6 boxes, CoreML=9 boxes

Trace which PNet boxes are filtered differently by RNet in ONNX vs CoreML.
Both use identical pipeline code, so differences must be from model scores.
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
print("RNet Stage Divergence Analysis")
print("="*80)

# Load image
img = cv2.imread(TEST_IMAGE)
print(f"\nTest image: {img.shape}")

# Initialize detectors
print("\nInitializing detectors...")
coreml_detector = CoreMLMTCNN(verbose=False)
onnx_detector = ONNXMTCNN(verbose=False)

# Run full detection with debug
print("\nRunning CoreML with debug...")
coreml_bboxes, coreml_landmarks, coreml_debug = coreml_detector.detect_with_debug(img)

print("Running ONNX with debug...")
onnx_bboxes, onnx_landmarks, onnx_debug = onnx_detector.detect_with_debug(img)

# Compare stage counts
print("\n" + "="*80)
print("Stage Counts Summary")
print("="*80)
print(f"\nPNet:  CoreML={coreml_debug['pnet']['num_boxes']}, ONNX={onnx_debug['pnet']['num_boxes']}")
print(f"RNet:  CoreML={coreml_debug['rnet']['num_boxes']}, ONNX={onnx_debug['rnet']['num_boxes']}")
print(f"ONet:  CoreML={coreml_debug['onet']['num_boxes']}, ONNX={onnx_debug['onet']['num_boxes']}")
print(f"Final: CoreML={coreml_debug['final']['num_boxes']}, ONNX={onnx_debug['final']['num_boxes']}")

# Analyze PNet outputs (inputs to RNet)
print("\n" + "="*80)
print("PNet Output Analysis")
print("="*80)

pnet_coreml_count = coreml_debug['pnet']['num_boxes']
pnet_onnx_count = onnx_debug['pnet']['num_boxes']

print(f"\nPNet detected same number of boxes: {pnet_coreml_count == pnet_onnx_count}")

if pnet_coreml_count == pnet_onnx_count:
    print(f"  → Both PNet outputs have {pnet_coreml_count} boxes")
    print(f"  → Divergence happens IN RNet filtering (same inputs, different scores)")
else:
    print(f"  → PNet outputs differ: CoreML={pnet_coreml_count}, ONNX={onnx_debug['pnet']['num_boxes']}")
    print(f"  → Divergence starts BEFORE RNet")

# We need to manually run PNet and RNet to get scores
print("\n" + "="*80)
print("Detailed RNet Score Analysis (Manual Trace)")
print("="*80)

# Run PNet stage manually for both backends
img_h, img_w = img.shape[:2]
img_float = img.astype(np.float32)

# Stage 1: PNet multi-scale detection
min_face_size = 60
m = 12.0 / min_face_size
min_len = min(img_h, img_w)
factor = 0.709

scales = []
scale = m
while scale * min_len >= 12:
    scales.append(scale)
    scale *= factor

print(f"\nPNet will run on {len(scales)} scales: {[f'{s:.6f}' for s in scales[:3]]}...")

# For detailed analysis, let's focus on the first scale where most detections come from
scale = scales[0]
hs = int(np.ceil(img_h * scale))
ws = int(np.ceil(img_w * scale))

print(f"\nFirst scale: {scale:.6f}, Scaled size: {hs}x{ws}")

# Preprocess for PNet
img_scaled = cv2.resize(img_float, (ws, hs), interpolation=cv2.INTER_LINEAR)
img_data_coreml = coreml_detector._preprocess(img_scaled, flip_bgr_to_rgb=True)
img_data_onnx = onnx_detector._preprocess(img_scaled, flip_bgr_to_rgb=True)

# Run PNet on first scale
coreml_pnet_out = coreml_detector._run_pnet(img_data_coreml)
onnx_pnet_out = onnx_detector._run_pnet(img_data_onnx)

# Extract face probabilities
coreml_pnet_chw = coreml_pnet_out[0].transpose(1, 2, 0)
onnx_pnet_chw = onnx_pnet_out[0].transpose(1, 2, 0)

# Compute face probabilities
coreml_prob_face = 1.0 / (1.0 + np.exp(coreml_pnet_chw[:, :, 0] - coreml_pnet_chw[:, :, 1]))
onnx_prob_face = 1.0 / (1.0 + np.exp(onnx_pnet_chw[:, :, 0] - onnx_pnet_chw[:, :, 1]))

print(f"\nFirst scale PNet results:")
print(f"  CoreML: max prob={coreml_prob_face.max():.6f}, cells >0.6: {(coreml_prob_face > 0.6).sum()}")
print(f"  ONNX:   max prob={onnx_prob_face.max():.6f}, cells >0.6: {(onnx_prob_face > 0.6).sum()}")

# Check probability differences
prob_diff = np.abs(coreml_prob_face - onnx_prob_face)
print(f"  Max prob diff: {prob_diff.max():.6f}")

# Find cells where one backend passes threshold but other doesn't
coreml_pass = coreml_prob_face > 0.6
onnx_pass = onnx_prob_face > 0.6
only_coreml = coreml_pass & ~onnx_pass
only_onnx = onnx_pass & ~coreml_pass

print(f"\n  Cells passing only in CoreML: {only_coreml.sum()}")
print(f"  Cells passing only in ONNX:   {only_onnx.sum()}")

if only_coreml.sum() > 0:
    coords = np.where(only_coreml)
    for i in range(min(3, len(coords[0]))):
        y, x = coords[0][i], coords[1][i]
        coreml_p = coreml_prob_face[y, x]
        onnx_p = onnx_prob_face[y, x]
        print(f"    [{y},{x}]: CoreML={coreml_p:.6f} (PASS), ONNX={onnx_p:.6f} (FAIL)")

if only_onnx.sum() > 0:
    coords = np.where(only_onnx)
    for i in range(min(3, len(coords[0]))):
        y, x = coords[0][i], coords[1][i]
        coreml_p = coreml_prob_face[y, x]
        onnx_p = onnx_prob_face[y, x]
        print(f"    [{y},{x}]: CoreML={coreml_p:.6f} (FAIL), ONNX={onnx_p:.6f} (PASS)")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

print(f"""
Key Findings:
1. PNet outputs: CoreML={pnet_coreml_count}, ONNX={pnet_onnx_count}
2. RNet outputs: CoreML={coreml_debug['rnet']['num_boxes']}, ONNX={onnx_debug['rnet']['num_boxes']}
3. ONet outputs: CoreML={coreml_debug['onet']['num_boxes']}, ONNX={onnx_debug['onet']['num_boxes']}

If PNet counts match:
  → Same boxes enter RNet, but RNet scores differ due to model numerical differences
  → Some boxes are borderline (~0.7 threshold) and pass in one backend but not the other

If PNet counts differ:
  → Divergence starts at PNet (some boxes pass 0.6 threshold in one backend but not other)
  → PNet model numerical differences cause different initial detections
  → This cascades through RNet and ONet

Current Status:
  → RNet divergence ({coreml_debug['rnet']['num_boxes']} vs {onnx_debug['rnet']['num_boxes']}) leads to ONet getting different boxes
  → ONet produces 2 boxes for ONNX vs 1 for CoreML
  → These 2 boxes have low overlap (IoU=0.36) so NMS keeps both
  → But they're the SAME face - this is the bug
""")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

print("""
The ~2% model numerical differences between CoreML (FP32) and ONNX are causing
borderline detections to pass/fail differently at EACH stage threshold:

- PNet threshold: 0.6
- RNet threshold: 0.7
- ONet threshold: 0.7

Boxes with scores near these thresholds will be filtered differently, leading
to different downstream detections.

Solutions:
1. Accept this as expected behavior for different model formats (NOT ACCEPTABLE)
2. Adjust thresholds to be more conservative (e.g., 0.72 instead of 0.7)
3. Investigate why models produce different scores (already verified models are correct)
4. Use platform-specific backends (CoreML on macOS, ONNX elsewhere)

Recommended: Investigate why PNet/RNet produce different scores despite bit-for-bit
accuracy on synthetic inputs. The issue may be in how real image patches are cropped
and preprocessed before being fed to RNet.
""")
