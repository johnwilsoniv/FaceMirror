#!/usr/bin/env python3
"""
Compare C++ .dat ONet output vs ONNX ONet output on identical input.
"""

import numpy as np
import onnxruntime as ort

# Load the C++ ONet input tensor (48x48x3 float32, HWC format)
cpp_input = np.fromfile('/tmp/cpp_onet_input.bin', dtype=np.float32)
cpp_input = cpp_input.reshape((48, 48, 3))

print("="*80)
print("COMPARING C++ .DAT vs ONNX ONET OUTPUTS")
print("="*80)

print(f"\nLoaded C++ ONet input tensor")
print(f"  Shape: {cpp_input.shape}")
print(f"  Dtype: {cpp_input.dtype}")
print(f"  Sample [0,0,:]: {cpp_input[0, 0, :]}")
print(f"  Range: [{cpp_input.min():.6f}, {cpp_input.max():.6f}]")

# C++ .dat model output (from debug log)
print(f"\nC++ .dat ONet Output:")
print(f"  logit[0] (not_face): -3.41372")
print(f"  logit[1] (face):      3.41272")
print(f"  prob: 0.998916")

# Now test ONNX model
print(f"\nLoading ONNX ONet model...")
onet_session = ort.InferenceSession(
    "cpp_mtcnn_onnx/onet.onnx",
    providers=['CPUExecutionProvider']
)

# ONNX expects (N, C, H, W), C++ provides (H, W, C)
# So we need to transpose
onnx_input = np.transpose(cpp_input, (2, 0, 1))  # HWC -> CHW
onnx_input = np.expand_dims(onnx_input, 0)  # Add batch dimension

print(f"\nONNX input tensor prepared:")
print(f"  Shape: {onnx_input.shape}")  # Should be (1, 3, 48, 48)
print(f"  Sample [0,0,0,:3]: {onnx_input[0, 0, 0, :3]}")
print(f"  Sample [0,1,0,:3]: {onnx_input[0, 1, 0, :3]}")
print(f"  Sample [0,2,0,:3]: {onnx_input[0, 2, 0, :3]}")

# Run ONNX inference
onnx_output = onet_session.run(None, {'input': onnx_input})[0]

print(f"\nONNX ONet Output:")
print(f"  Raw output shape: {onnx_output.shape}")
print(f"  Raw output: {onnx_output[0]}")
print(f"  logit[0] (not_face): {onnx_output[0, 0]:.6f}")
print(f"  logit[1] (face):     {onnx_output[0, 1]:.6f}")

# Calculate probability
logit_diff = onnx_output[0, 0] - onnx_output[0, 1]
prob = 1.0 / (1.0 + np.exp(logit_diff))
print(f"  prob: {prob:.6f}")

# Compare
print(f"\n{'='*80}")
print(f"COMPARISON")
print(f"{'='*80}")
print(f"  logit[0] difference: {-3.41372 - onnx_output[0, 0]:.6f}")
print(f"  logit[1] difference: {3.41272 - onnx_output[0, 1]:.6f}")
print(f"  prob difference: {0.998916 - prob:.6f}")

if abs(-3.41372 - onnx_output[0, 0]) < 0.01 and abs(3.41272 - onnx_output[0, 1]) < 0.01:
    print(f"\n✓ OUTPUTS MATCH! ONNX model is correct.")
else:
    print(f"\n❌ OUTPUTS DIFFER! This explains the detection failures.")
    print(f"\nPossible causes:")
    print(f"  1. ONNX model weights differ from .dat model")
    print(f"  2. ONNX conversion introduced errors")
    print(f"  3. Channel ordering mismatch (BGR vs RGB)")
