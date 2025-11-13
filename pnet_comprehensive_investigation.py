#!/usr/bin/env python3
"""
Comprehensive PNet Investigation - Steps 1-4:
1. Compare PNet C++ vs Python implementation
2. Check preprocessing (RGB vs BGR, etc.)
3. Visualize PNet output maps at each scale
4. Test on synthetic inputs

This script systematically investigates why Pure Python PNet doesn't detect faces
at appropriate scales.
"""

import cv2
import numpy as np
from cpp_cnn_loader import CPPCNN
import os
import matplotlib.pyplot as plt

print("=" * 80)
print("COMPREHENSIVE PNET INVESTIGATION")
print("=" * 80)

#  ===========================================================================
# STEP 1: Load and Compare C++ vs Python PNet Outputs
# ===========================================================================

print("\n" + "=" * 80)
print("STEP 1: Compare C++ vs Python PNet Outputs")
print("=" * 80)

# Load C++ PNet outputs from previous run
cpp_input = np.fromfile('/tmp/cpp_pnet_input_scale0.bin', dtype=np.float32)
cpp_logit0 = np.fromfile('/tmp/cpp_pnet_logit0_scale0.bin', dtype=np.float32)
cpp_logit1 = np.fromfile('/tmp/cpp_pnet_logit1_scale0.bin', dtype=np.float32)

print(f"\nC++ PNet Debug Files:")
print(f"  Input size: {cpp_input.shape}")
print(f"  Logit0 size: {cpp_logit0.shape}")
print(f"  Logit1 size: {cpp_logit1.shape}")

# The C++ code dumps scale 0, need to figure out dimensions
# From C++ code: w_pyr × h_pyr input → output through PNet
# For our test image (1920×1080) at scale 0 (m=0.3): 576×324 input
# PNet output will be smaller due to convolutions

# Try to reshape (guess dimensions from file size)
output_size = int(np.sqrt(len(cpp_logit0)))
if output_size * output_size == len(cpp_logit0):
    cpp_logit0 = cpp_logit0.reshape(output_size, output_size)
    cpp_logit1 = cpp_logit1.reshape(output_size, output_size)
    print(f"\nC++ PNet output shape: {output_size}×{output_size}")
else:
    # Not square, try aspect ratio similar to input
    # Input was 576×324, so try proportional output
    h_out = int(np.sqrt(len(cpp_logit0) * 324 / 576))
    w_out = len(cpp_logit0) // h_out
    if h_out * w_out == len(cpp_logit0):
        cpp_logit0 = cpp_logit0.reshape(h_out, w_out)
        cpp_logit1 = cpp_logit1.reshape(h_out, w_out)
        print(f"\nC++ PNet output shape: {h_out}×{w_out}")
    else:
        print(f"\n⚠️  Cannot determine C++ output shape from {len(cpp_logit0)} elements")

# Compute C++ probabilities
cpp_prob = 1.0 / (1.0 + np.exp(cpp_logit0 - cpp_logit1))

print(f"\nC++ PNet Output Statistics:")
print(f"  Logit0 (not-face): [{cpp_logit0.min():.3f}, {cpp_logit0.max():.3f}], mean={cpp_logit0.mean():.3f}")
print(f"  Logit1 (face): [{cpp_logit1.min():.3f}, {cpp_logit1.max():.3f}], mean={cpp_logit1.mean():.3f}")
print(f"  Probability: [{cpp_prob.min():.3f}, {cpp_prob.max():.3f}], mean={cpp_prob.mean():.3f}")

# Now run Python PNet on the same input
print("\n" + "-" * 80)
print("Running Python PNet on same input...")
print("-" * 80)

# Load PNet model
model_dir = os.path.expanduser(
    "~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/"
    "face_detection/mtcnn/convert_to_cpp/"
)
pnet = CPPCNN(os.path.join(model_dir, "PNet.dat"))

# Load test image and prepare input at scale 0
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
img_h, img_w = img.shape[:2]
print(f"\nOriginal image: {img_w}×{img_h}")

# Scale 0 parameters (matching C++ calculation)
min_face_size = 40
m = 12.0 / min_face_size  # 0.3
hs = int(np.ceil(img_h * m))
ws = int(np.ceil(img_w * m))

print(f"Scale 0: m={m:.4f}, scaled size: {ws}×{hs}")

img_float = img.astype(np.float32)
img_scaled = cv2.resize(img_float, (ws, hs))

# Preprocess
def preprocess(img):
    img_norm = (img - 127.5) * 0.0078125
    img_chw = np.transpose(img_norm, (2, 0, 1))
    return img_chw

img_data = preprocess(img_scaled)

print(f"Input to PNet: {img_data.shape}")

# Run PNet
py_output = pnet(img_data)
py_output = py_output[-1]  # Get final output
py_output = py_output[np.newaxis, :, :, :]  # Add batch dimension
py_output = py_output[0].transpose(1, 2, 0)  # (H, W, C)

print(f"Python PNet output shape: {py_output.shape}")

py_logit0 = py_output[:, :, 0]
py_logit1 = py_output[:, :, 1]
py_prob = 1.0 / (1.0 + np.exp(py_logit0 - py_logit1))

print(f"\nPython PNet Output Statistics:")
print(f"  Logit0 (not-face): [{py_logit0.min():.3f}, {py_logit0.max():.3f}], mean={py_logit0.mean():.3f}")
print(f"  Logit1 (face): [{py_logit1.min():.3f}, {py_logit1.max():.3f}], mean={py_logit1.mean():.3f}")
print(f"  Probability: [{py_prob.min():.3f}, {py_prob.max():.3f}], mean={py_prob.mean():.3f}")

# Compare
if cpp_logit0.shape == py_logit0.shape:
    print("\n" + "-" * 80)
    print("C++ vs Python Comparison:")
    print("-" * 80)

    logit0_diff = np.abs(cpp_logit0 - py_logit0)
    logit1_diff = np.abs(cpp_logit1 - py_logit1)
    prob_diff = np.abs(cpp_prob - py_prob)

    print(f"\nAbsolute Differences:")
    print(f"  Logit0: max={logit0_diff.max():.4f}, mean={logit0_diff.mean():.4f}")
    print(f"  Logit1: max={logit1_diff.max():.4f}, mean={logit1_diff.mean():.4f}")
    print(f"  Probability: max={prob_diff.max():.4f}, mean={prob_diff.mean():.4f}")

    # Check correlation
    corr0 = np.corrcoef(cpp_logit0.flatten(), py_logit0.flatten())[0, 1]
    corr1 = np.corrcoef(cpp_logit1.flatten(), py_logit1.flatten())[0, 1]
    print(f"\nCorrelations:")
    print(f"  Logit0: {corr0:.4f}")
    print(f"  Logit1: {corr1:.4f}")

    if corr1 < 0:
        print(f"\n⚠️  WARNING: Negative correlation in logit1 ({corr1:.4f})!")
        print(f"  This matches the previous finding of Channel 1 divergence")
else:
    print(f"\n⚠️  Shape mismatch: C++={cpp_logit0.shape}, Python={py_logit0.shape}")

# ===========================================================================
# STEP 2: Check Preprocessing Variations
# ===========================================================================

print("\n" + "=" * 80)
print("STEP 2: Test Preprocessing Variations")
print("=" * 80)

# Test different preprocessing approaches
preprocessing_variants = {
    'Standard (current)': lambda img: (img - 127.5) * 0.0078125,
    'RGB→BGR swap': lambda img: (img[:, :, ::-1] - 127.5) * 0.0078125,
    'Different normalization': lambda img: img / 255.0,
    'Mean subtraction': lambda img: img - np.array([103.939, 116.779, 123.68])
}

print(f"\nTesting {len(preprocessing_variants)} preprocessing variants...")

for name, preproc_fn in preprocessing_variants.items():
    img_variant = preproc_fn(img_scaled.copy())
    img_variant_chw = np.transpose(img_variant, (2, 0, 1))

    output_variant = pnet(img_variant_chw)[-1]
    output_variant = output_variant[np.newaxis, :, :, :][0].transpose(1, 2, 0)

    logit1_variant = output_variant[:, :, 1]
    prob_variant = 1.0 / (1.0 + np.exp(output_variant[:, :, 0] - logit1_variant))

    print(f"\n{name}:")
    print(f"  Max probability: {prob_variant.max():.4f}")
    print(f"  Mean probability: {prob_variant.mean():.4f}")
    if cpp_logit0.shape == output_variant.shape[:2]:
        corr = np.corrcoef(cpp_logit1.flatten(), logit1_variant.flatten())[0, 1]
        print(f"  Correlation with C++: {corr:.4f}")

# ===========================================================================
# STEP 3: Visualize PNet Output Maps
# ===========================================================================

print("\n" + "=" * 80)
print("STEP 3: Visualize PNet Output Maps at Multiple Scales")
print("=" * 80)

# Test multiple pyramid scales
scales_to_test = [0, 1, 2, 3]  # First 4 scales
factor = 0.709

fig, axes = plt.subplots(2, len(scales_to_test), figsize=(20, 10))

for idx, scale_idx in enumerate(scales_to_test):
    scale = m * (factor ** scale_idx)
    hs = int(np.ceil(img_h * scale))
    ws = int(np.ceil(img_w * scale))

    face_size = min_face_size / scale

    img_scaled = cv2.resize(img_float, (ws, hs))
    img_data = preprocess(img_scaled)

    output = pnet(img_data)[-1]
    output = output[np.newaxis, :, :, :][0].transpose(1, 2, 0)

    prob = 1.0 / (1.0 + np.exp(output[:, :, 0] - output[:, :, 1]))

    print(f"\nScale {scale_idx}: {ws}×{hs} (detects faces ≥{face_size:.0f}px)")
    print(f"  Output: {prob.shape}")
    print(f"  Max prob: {prob.max():.4f}, Mean: {prob.mean():.4f}")
    print(f"  Prob > 0.6: {(prob > 0.6).sum()} pixels")

    # Plot input
    ax = axes[0, idx]
    ax.imshow(cv2.cvtColor(img_scaled.astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax.set_title(f'Scale {scale_idx}: {ws}×{hs}\\nDetects ≥{face_size:.0f}px faces')
    ax.axis('off')

    # Plot probability map
    ax = axes[1, idx]
    im = ax.imshow(prob, cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'Probability Map\\nMax: {prob.max():.3f}')
    ax.axis('off')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('pnet_pyramid_visualization.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization to: pnet_pyramid_visualization.png")

# ===========================================================================
# STEP 4: Test on Synthetic Inputs
# ===========================================================================

print("\n" + "=" * 80)
print("STEP 4: Test PNet on Synthetic Inputs")
print("=" * 80)

# Test 1: Uniform images
print("\nTest 1: Uniform color images")
for color_val in [0, 127, 255]:
    synthetic = np.full((48, 48, 3), color_val, dtype=np.float32)
    synth_data = preprocess(synthetic)
    output = pnet(synth_data)[-1]
    output = output[np.newaxis, :, :, :][0].transpose(1, 2, 0)
    prob = 1.0 / (1.0 + np.exp(output[:, :, 0] - output[:, :, 1]))
    print(f"  Uniform {color_val}: max_prob={prob.max():.4f}, mean_prob={prob.mean():.4f}")

# Test 2: Checkerboard pattern
print("\nTest 2: Checkerboard pattern")
checkerboard = np.indices((48, 48)).sum(axis=0) % 2
checkerboard = np.stack([checkerboard * 255] * 3, axis=2).astype(np.float32)
synth_data = preprocess(checkerboard)
output = pnet(synth_data)[-1]
output = output[np.newaxis, :, :, :][0].transpose(1, 2, 0)
prob = 1.0 / (1.0 + np.exp(output[:, :, 0] - output[:, :, 1]))
print(f"  Checkerboard: max_prob={prob.max():.4f}, mean_prob={prob.mean():.4f}")

# Test 3: Random noise
print("\nTest 3: Random noise")
np.random.seed(42)
noise = np.random.randint(0, 256, (48, 48, 3)).astype(np.float32)
synth_data = preprocess(noise)
output = pnet(synth_data)[-1]
output = output[np.newaxis, :, :, :][0].transpose(1, 2, 0)
prob = 1.0 / (1.0 + np.exp(output[:, :, 0] - output[:, :, 1]))
print(f"  Random noise: max_prob={prob.max():.4f}, mean_prob={prob.mean():.4f}")

# Test 4: Gradient
print("\nTest 4: Horizontal gradient")
gradient = np.linspace(0, 255, 48).reshape(1, 48, 1)
gradient = np.repeat(gradient, 48, axis=0)
gradient = np.repeat(gradient, 3, axis=2).astype(np.float32)
synth_data = preprocess(gradient)
output = pnet(synth_data)[-1]
output = output[np.newaxis, :, :, :][0].transpose(1, 2, 0)
prob = 1.0 / (1.0 + np.exp(output[:, :, 0] - output[:, :, 1]))
print(f"  Gradient: max_prob={prob.max():.4f}, mean_prob={prob.mean():.4f}")

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)

print("\nKey Findings to Review:")
print("1. C++ vs Python logit correlation (especially Channel 1)")
print("2. Which preprocessing variant matches C++ best")
print("3. At which pyramid scale do we get highest face probabilities")
print("4. How synthetic inputs behave vs real images")
print("\nCheck 'pnet_pyramid_visualization.png' for visual analysis")
