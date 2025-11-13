#!/usr/bin/env python3
"""
Test if multiplying all weights by 2 fixes the output scaling.
"""

import numpy as np
import onnxruntime as ort
import onnx
from onnx import numpy_helper
from pathlib import Path

print("="*80)
print("TESTING 2X WEIGHT SCALING HYPOTHESIS")
print("="*80)

# Load the original ONNX model
onnx_path = "cpp_mtcnn_onnx/onet.onnx"
model = onnx.load(onnx_path)

# Multiply all weight/bias initializers by 2
for initializer in model.graph.initializer:
    tensor = numpy_helper.to_array(initializer)
    tensor_scaled = tensor * 2.0

    # Replace the initializer
    new_initializer = numpy_helper.from_array(tensor_scaled, initializer.name)
    initializer.CopyFrom(new_initializer)
    print(f"  Scaled {initializer.name}: shape {tensor.shape}")

# Save scaled model
scaled_path = "cpp_mtcnn_onnx/onet_2x.onnx"
onnx.save(model, scaled_path)
print(f"\nSaved 2x scaled model to {scaled_path}")

# Test with the same input
cpp_input = np.fromfile('/tmp/cpp_onet_input.bin', dtype=np.float32).reshape((48, 48, 3))
onnx_input = np.transpose(cpp_input, (2, 0, 1))  # HWC -> CHW
onnx_input = np.expand_dims(onnx_input, 0)  # Add batch dimension

# Original model
print(f"\n{'='*80}")
print("ORIGINAL MODEL")
print(f"{'='*80}")
session_orig = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
output_orig = session_orig.run(None, {'input': onnx_input})[0]
print(f"  logit[0]: {output_orig[0, 0]:.6f}")
print(f"  logit[1]: {output_orig[0, 1]:.6f}")

# Scaled model
print(f"\n{'='*80}")
print("2X SCALED MODEL")
print(f"{'='*80}")
session_scaled = ort.InferenceSession(scaled_path, providers=['CPUExecutionProvider'])
output_scaled = session_scaled.run(None, {'input': onnx_input})[0]
print(f"  logit[0]: {output_scaled[0, 0]:.6f}")
print(f"  logit[1]: {output_scaled[0, 1]:.6f}")

# C++ reference
print(f"\n{'='*80}")
print("C++ REFERENCE")
print(f"{'='*80}")
cpp_logit0 = -3.41372
cpp_logit1 = 3.41272
print(f"  logit[0]: {cpp_logit0:.6f}")
print(f"  logit[1]: {cpp_logit1:.6f}")

# Compare
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")
print(f"Original vs C++:")
print(f"  logit[0] diff: {abs(output_orig[0, 0] - cpp_logit0):.6f}")
print(f"  logit[1] diff: {abs(output_orig[0, 1] - cpp_logit1):.6f}")

print(f"\n2x Scaled vs C++:")
print(f"  logit[0] diff: {abs(output_scaled[0, 0] - cpp_logit0):.6f}")
print(f"  logit[1] diff: {abs(output_scaled[0, 1] - cpp_logit1):.6f}")

if abs(output_scaled[0, 0] - cpp_logit0) < 0.01 and abs(output_scaled[0, 1] - cpp_logit1) < 0.01:
    print("\n🎯 2X SCALING FIXES IT!")
    print("ROOT CAUSE: All weights need to be multiplied by 2")
else:
    print("\n❌ 2x scaling doesn't fix it")
    print(f"   Scaling factor: {cpp_logit1 / output_orig[0, 1]:.4f}")
