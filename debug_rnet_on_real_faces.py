#!/usr/bin/env python3
"""
Debug Pure Python RNet on real face crops from PNet stage.
Capture actual 24x24 face inputs and analyze RNet behavior.
"""

import cv2
import numpy as np
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2
from cpp_cnn_loader import CPPCNN
import os

print("=" * 80)
print("RNET DEBUG ON REAL FACE CROPS")
print("=" * 80)

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
print(f"\nTest image: {img.shape}")

# Create modified detector that captures RNet inputs
class DebugMTCNN(PurePythonMTCNN_V2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rnet_inputs = []  # Store all RNet inputs
        self.rnet_outputs = []  # Store all RNet outputs

    def _run_rnet(self, img_data):
        """Run RNet and capture inputs/outputs for debugging."""
        self.rnet_inputs.append(img_data.copy())
        outputs = self.rnet(img_data)
        output = outputs[-1]
        self.rnet_outputs.append(output.copy())
        return output

# Run detection with debug mode
print("\n" + "=" * 80)
print("STAGE 1: Run MTCNN and capture RNet inputs")
print("=" * 80)

detector = DebugMTCNN()
detector.thresholds = [0.6, 0.6, 0.6]  # Use lowered thresholds
bboxes, landmarks = detector.detect(img)

print(f"\nCaptured {len(detector.rnet_inputs)} RNet face crops")
print(f"Final detections: {len(bboxes)} faces")

if len(detector.rnet_inputs) == 0:
    print("\n⚠️  No faces passed PNet stage!")
    print("Try lowering PNet threshold or check PNet operation.")
    exit(1)

# Analyze all RNet inputs
print("\n" + "=" * 80)
print("STAGE 2: Analyze RNet inputs and outputs")
print("=" * 80)

for i, (face_crop, rnet_output) in enumerate(zip(detector.rnet_inputs, detector.rnet_outputs)):
    print(f"\n--- Face {i+1}/{len(detector.rnet_inputs)} ---")

    # Input stats
    print(f"Input shape: {face_crop.shape}")
    print(f"Input range: [{face_crop.min():.4f}, {face_crop.max():.4f}]")
    print(f"Input mean: {face_crop.mean():.4f}")
    print(f"Input std: {face_crop.std():.4f}")

    # Output stats
    logit_not_face = rnet_output[0]
    logit_face = rnet_output[1]
    score = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

    print(f"\nRNet output:")
    print(f"  Logit not-face: {logit_not_face:.4f}")
    print(f"  Logit face: {logit_face:.4f}")
    print(f"  Score: {score:.4f} {'✓ PASS' if score > 0.6 else '✗ FAIL'} (threshold=0.6)")
    print(f"  Regression: [{rnet_output[2]:.4f}, {rnet_output[3]:.4f}, {rnet_output[4]:.4f}, {rnet_output[5]:.4f}]")

    # Save the first few face crops for manual inspection
    if i < 5:
        # Denormalize for visualization
        face_vis = face_crop.copy()
        face_vis = (face_vis / 0.0078125 + 127.5).clip(0, 255).astype(np.uint8)
        face_vis = np.transpose(face_vis, (1, 2, 0))  # CHW -> HWC
        face_vis = cv2.cvtColor(face_vis, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f'debug_rnet_face_{i+1}_score{score:.3f}.jpg', face_vis)
        print(f"  💾 Saved visualization: debug_rnet_face_{i+1}_score{score:.3f}.jpg")

# Find the highest scoring face
if len(detector.rnet_outputs) > 0:
    scores = []
    for output in detector.rnet_outputs:
        score = 1.0 / (1.0 + np.exp(output[0] - output[1]))
        scores.append(score)

    best_idx = np.argmax(scores)
    best_score = scores[best_idx]

    print("\n" + "=" * 80)
    print("STAGE 3: Detailed analysis of highest-scoring face")
    print("=" * 80)

    print(f"\nHighest RNet score: {best_score:.4f} (Face {best_idx+1})")

    if best_score < 0.7:
        gap = 0.7 - best_score
        print(f"\n⚠️  Best score is {gap:.4f} below official RNet threshold (0.7)")
        print(f"This is why Pure Python MTCNN fails with official thresholds.")

    # Run layer-by-layer analysis on best face
    print(f"\n--- Layer-by-layer analysis of Face {best_idx+1} ---")

    best_input = detector.rnet_inputs[best_idx]
    current = best_input

    model_dir = os.path.expanduser(
        "~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/"
        "face_detection/mtcnn/convert_to_cpp/"
    )
    rnet = CPPCNN(os.path.join(model_dir, "RNet.dat"))

    for layer_idx, layer in enumerate(rnet.layers):
        print(f"\nLayer {layer_idx}: {layer.__class__.__name__}")

        # Show layer details
        if hasattr(layer, 'num_kernels'):
            print(f"  Conv: {layer.num_in_maps}→{layer.num_kernels}, kernel={layer.kernel_h}x{layer.kernel_w}")
        elif hasattr(layer, 'slopes'):
            print(f"  PReLU: {len(layer.slopes)} channels")
            print(f"  Slopes range: [{layer.slopes.min():.4f}, {layer.slopes.max():.4f}]")
        elif hasattr(layer, 'kernel_size'):
            print(f"  MaxPool: kernel={layer.kernel_size}, stride={layer.stride}")
        elif hasattr(layer, 'weights'):
            print(f"  FC: {layer.weights.shape[1]}→{layer.weights.shape[0]}")

        print(f"  Input: shape={current.shape}, range=[{current.min():.4f}, {current.max():.4f}]")

        current = layer.forward(current)

        print(f"  Output: shape={current.shape}, range=[{current.min():.4f}, {current.max():.4f}]")

        # Check for issues
        if np.isnan(current).any():
            print(f"  ⚠️  NaN detected!")
        if np.isinf(current).any():
            print(f"  ⚠️  Inf detected!")

print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

print(f"\n1. PNet passed {len(detector.rnet_inputs)} faces to RNet")
print(f"2. RNet best score: {max(scores):.4f}")
print(f"3. Official RNet threshold: 0.7")
print(f"4. Gap to official threshold: {0.7 - max(scores):.4f}")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

print("\n1. Compare saved face crops to C++ RNet input expectations")
print("2. Run C++ MTCNN and extract its RNet inputs for comparison")
print("3. Check if preprocessing/normalization differs from C++")
print("4. Verify weight loading is correct (compare first layer conv output)")
