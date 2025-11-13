#!/usr/bin/env python3
"""
Debug PNet output format in V2.
"""

import cv2
import numpy as np
from cpp_cnn_loader import CPPCNN
import os

# Load PNet
model_dir = os.path.expanduser(
    "~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/"
    "face_detection/mtcnn/convert_to_cpp/"
)

pnet = CPPCNN(os.path.join(model_dir, "PNet.dat"))

# Load and preprocess image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
img_float = img.astype(np.float32)

# Resize to first scale
scale = 0.3
hs = int(np.ceil(img.shape[0] * scale))
ws = int(np.ceil(img.shape[1] * scale))
img_scaled = cv2.resize(img_float, (ws, hs))

# Preprocess (no batch dimension)
img_norm = (img_scaled - 127.5) * 0.0078125
img_chw = np.transpose(img_norm, (2, 0, 1))  # (3, H, W)

print(f"Input shape: {img_chw.shape}")
print(f"Input range: [{img_chw.min():.3f}, {img_chw.max():.3f}]")

# Run PNet
outputs = pnet(img_chw)

print(f"\nNumber of layer outputs: {len(outputs)}")
for i, out in enumerate(outputs):
    if isinstance(out, np.ndarray):
        print(f"  Layer {i}: shape={out.shape}, dtype={out.dtype}")
    else:
        print(f"  Layer {i}: type={type(out)}")

# Check final output
final_output = outputs[-1]
print(f"\nFinal output shape: {final_output.shape}")
print(f"Final output range: [{final_output.min():.3f}, {final_output.max():.3f}]")

# Expected: (6, H_out, W_out)
if final_output.ndim == 3:
    C, H, W = final_output.shape
    print(f"\nInterpreting as (C={C}, H={H}, W={W})")

    # Try to interpret as PNet output
    if C == 6:
        print("✓ Correct number of channels (6)")
        logits = final_output[:2, :, :]
        reg = final_output[2:6, :, :]
        print(f"  Logits shape: {logits.shape}")
        print(f"  Regression shape: {reg.shape}")

        # Calculate probabilities
        logit_not_face = final_output[0, :, :]
        logit_face = final_output[1, :, :]
        prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

        print(f"\n  Prob face range: [{prob_face.min():.4f}, {prob_face.max():.4f}]")
        print(f"  Scores > 0.6: {(prob_face > 0.6).sum()}")
        print(f"  Scores > 0.7: {(prob_face > 0.7).sum()}")
        print(f"  Scores > 0.8: {(prob_face > 0.8).sum()}")
    else:
        print(f"❌ Wrong number of channels ({C}, expected 6)")
else:
    print(f"❌ Wrong number of dimensions ({final_output.ndim}, expected 3)")
