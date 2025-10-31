#!/usr/bin/env python3
"""
Test if ONNX model outputs match PyTorch model outputs on identical inputs
"""

import numpy as np
import torch
import cv2
import onnxruntime as ort
from openface.model.MTL import MTL

# Load PyTorch model
print("="*80)
print("LOADING PYTORCH MODEL")
print("="*80)
pytorch_model = MTL()
state_dict = torch.load('weights/MTL_backbone.pth', map_location='cpu')
pytorch_model.load_state_dict(state_dict)
pytorch_model.eval()
print("✓ PyTorch model loaded\n")

# Load ONNX model
print("="*80)
print("LOADING ONNX MODEL")
print("="*80)
onnx_session = ort.InferenceSession('weights/mtl_efficientnet_b0_coreml.onnx', providers=['CPUExecutionProvider'])
print("✓ ONNX model loaded\n")

# Create identical test input
print("="*80)
print("CREATING TEST INPUT")
print("="*80)
test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
print(f"Input shape: {test_input.shape}")
print(f"Input range: [{test_input.min():.4f}, {test_input.max():.4f}]\n")

# PyTorch inference
print("="*80)
print("PYTORCH INFERENCE")
print("="*80)
with torch.no_grad():
    test_input_torch = torch.from_numpy(test_input)
    emotion_pt, gaze_pt, au_pt = pytorch_model(test_input_torch)

print(f"Emotion output shape: {emotion_pt.shape}")
print(f"Gaze output shape: {gaze_pt.shape}")
print(f"AU output shape: {au_pt.shape}")
print(f"\nAU outputs (PyTorch):")
print(f"  {au_pt[0].numpy()}\n")

# ONNX inference
print("="*80)
print("ONNX INFERENCE")
print("="*80)
outputs_onnx = onnx_session.run(None, {'input_face': test_input})
emotion_onnx, gaze_onnx, au_onnx = outputs_onnx

print(f"Emotion output shape: {emotion_onnx.shape}")
print(f"Gaze output shape: {gaze_onnx.shape}")
print(f"AU output shape: {au_onnx.shape}")
print(f"\nAU outputs (ONNX):")
print(f"  {au_onnx[0]}\n")

# Compare outputs
print("="*80)
print("COMPARISON")
print("="*80)

# AU comparison
au_diff = np.abs(au_pt.numpy() - au_onnx)
print(f"\nAU Outputs:")
print(f"  Absolute differences: {au_diff[0]}")
print(f"  Mean abs diff: {au_diff.mean():.6f}")
print(f"  Max abs diff: {au_diff.max():.6f}")

if au_diff.max() < 1e-4:
    print(f"  ✓ EXCELLENT - Models are identical")
elif au_diff.max() < 1e-2:
    print(f"  ✓ GOOD - Minor numerical differences")
elif au_diff.max() < 0.1:
    print(f"  ⚠ MODERATE - Noticeable differences")
else:
    print(f"  ✗ POOR - Significant differences")
    print(f"\n  This suggests the ONNX export has issues!")

# Emotion comparison
emotion_diff = np.abs(emotion_pt.numpy() - emotion_onnx)
print(f"\nEmotion Outputs:")
print(f"  Mean abs diff: {emotion_diff.mean():.6f}")
print(f"  Max abs diff: {emotion_diff.max():.6f}")

# Gaze comparison
gaze_diff = np.abs(gaze_pt.numpy() - gaze_onnx)
print(f"\nGaze Outputs:")
print(f"  Mean abs diff: {gaze_diff.mean():.6f}")
print(f"  Max abs diff: {gaze_diff.max():.6f}")

print("\n" + "="*80)
