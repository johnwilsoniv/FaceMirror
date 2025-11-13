#!/usr/bin/env python3
"""
Implement the exact C++ im2col + BLAS convolution technique for PNet.

Based on CNN_utils.cpp:445-540 (im2col_multimap + convolution_direct_blas)
and FaceDetectorMTCNN.cpp:499-523 (weight matrix construction).
"""

import numpy as np
import torch
import torch.nn.functional as F

def im2col_multimap_cpp(inputs, kernel_h, kernel_w):
    """
    Implement C++'s im2col_multimap exactly.

    From CNN_utils.cpp:445-495:
    - Each row represents one spatial position
    - Column ordering: colIdx = xx*height + yy + in_maps * stride
    - Last column is 1.0 (for bias)

    Args:
        inputs: list of cv::Mat_<float> (num_channels, H, W)
        kernel_h, kernel_w: kernel dimensions

    Returns:
        im2col matrix (num_positions, num_features+1)
    """
    num_maps = len(inputs)
    H, W = inputs[0].shape

    # Output spatial dimensions
    yB = H - kernel_h + 1
    xB = W - kernel_w + 1
    num_positions = yB * xB

    stride = kernel_h * kernel_w
    num_features = num_maps * stride

    # Allocate output (include bias column)
    im2col = np.ones((num_positions, num_features + 1), dtype=np.float32)

    # Iterate over spatial positions
    for i in range(yB):
        for j in range(xB):
            row_idx = i * xB + j

            # Extract patch for this position
            for yy in range(kernel_h):
                for in_map_idx in range(num_maps):
                    for xx in range(kernel_w):
                        # C++ column ordering
                        col_idx = xx * kernel_h + yy + in_map_idx * stride
                        im2col[row_idx, col_idx] = inputs[in_map_idx][i + yy, j + xx]

    return im2col


def create_weight_matrix_cpp(weights, biases):
    """
    Create weight matrix exactly as C++ does.

    From FaceDetectorMTCNN.cpp:499-523:
    1. Transpose each kernel: kernels_rearr[k][i].t()
    2. Flatten and arrange
    3. Transpose weight matrix
    4. Add bias column
    5. Final transpose

    Args:
        weights: (num_kernels, num_in_maps, kernel_h, kernel_w)
        biases: (num_kernels,)

    Returns:
        Weight matrix ready for BLAS multiplication
    """
    num_kernels, num_in_maps, kernel_h, kernel_w = weights.shape
    stride = kernel_h * kernel_w

    # Step 1: Create weight_matrix
    # Shape: (num_in_maps * kernel_h * kernel_w, num_kernels)
    weight_matrix = np.zeros((num_in_maps * stride, num_kernels), dtype=np.float32)

    for k in range(num_kernels):
        for i in range(num_in_maps):
            # Transpose the kernel (line 506)
            k_flat = weights[k, i, :, :].T  # kernels_rearr[k][i].t()

            # Flatten and transpose (line 507)
            k_flat = k_flat.reshape(1, -1).T  # k_flat.reshape(0, 1).t()

            # Copy to weight matrix (line 508)
            start_row = i * stride
            end_row = start_row + stride
            weight_matrix[start_row:end_row, k] = k_flat[:, 0]

    # Step 2: Transpose the weight matrix (line 513)
    weight_matrix = weight_matrix.T

    # Step 3: Add bias column (lines 516-522)
    num_rows = weight_matrix.shape[0]
    num_cols = weight_matrix.shape[1]
    W = np.ones((num_rows, num_cols + 1), dtype=np.float32)

    # Copy weight_matrix
    W[:, :num_cols] = weight_matrix

    # Set bias column (last column)
    W[:, num_cols] = biases

    # Step 4: Final transpose (line 523)
    W = W.T

    return W


def convolution_direct_blas_cpp(inputs, weight_matrix, kernel_h, kernel_w):
    """
    Implement C++'s convolution_direct_blas exactly.

    From CNN_utils.cpp:500-540:
    1. im2col on inputs
    2. Matrix multiply: out = im2col @ weight_matrix.T
    3. Transpose output
    4. Reshape to spatial maps

    Args:
        inputs: list of (H, W) arrays
        weight_matrix: from create_weight_matrix_cpp (includes bias)
        kernel_h, kernel_w: kernel dimensions

    Returns:
        list of output feature maps
    """
    # Step 1: im2col
    im2col = im2col_multimap_cpp(inputs, kernel_h, kernel_w)

    # im2col shape: (num_positions, num_features + 1)
    # weight_matrix shape: (num_features + 1, num_kernels)

    # Step 2: Matrix multiplication (line 528)
    # sgemm: out = im2col * weight_matrix (no transpose needed since already transposed)
    out = im2col @ weight_matrix  # (num_positions, num_kernels)

    # Step 3: Transpose (line 532)
    out = out.T  # (num_kernels, num_positions)

    # Step 4: Reshape to spatial maps (lines 535-538)
    H, W = inputs[0].shape
    yB = H - kernel_h + 1
    xB = W - kernel_w + 1

    outputs = []
    for k in range(out.shape[0]):
        feature_map = out[k, :].reshape(yB, xB)
        outputs.append(feature_map)

    return outputs


# Test against C++ PNet layer 0
print("="*80)
print("IMPLEMENTING C++ IM2COL + BLAS TECHNIQUE FOR PNET")
print("="*80)

# Load C++ PNet input (HWC format)
cpp_input = np.fromfile('/tmp/cpp_pnet_input_scale0.bin', dtype=np.float32)
cpp_input = cpp_input.reshape(384, 216, 3)

print(f"\nC++ PNet Input:")
print(f"  Shape: {cpp_input.shape}")
print(f"  Sample pixel [0,0]: {cpp_input[0,0,:]}")

# Load PNet layer 0 weights
pnet_weights = np.load('cpp_mtcnn_weights/pnet/pnet_layer00_conv_weights.npy')
pnet_biases = np.load('cpp_mtcnn_weights/pnet/pnet_layer00_conv_biases.npy')

print(f"\nPNet Layer 0 Weights:")
print(f"  Shape: {pnet_weights.shape}")  # (10, 3, 3, 3)
print(f"  Bias shape: {pnet_biases.shape}")  # (10,)

# Convert input to channel-first for processing
# CRITICAL: OpenCV uses BGR order, so C++ inputs are [B, G, R] = [ch2, ch1, ch0]
input_maps = [cpp_input[:, :, c] for c in [2, 1, 0]]  # BGR order!

print(f"\nInput maps prepared:")
for i, inp_map in enumerate(input_maps):
    print(f"  Channel {i}: shape {inp_map.shape}")

# Create weight matrix using C++ technique
print(f"\nCreating weight matrix using C++ technique...")
weight_matrix = create_weight_matrix_cpp(pnet_weights, pnet_biases)
print(f"  Weight matrix shape: {weight_matrix.shape}")

# Perform convolution using C++ technique
print(f"\nPerforming convolution using C++ im2col + BLAS...")
outputs_cpp_style = convolution_direct_blas_cpp(input_maps, weight_matrix, 3, 3)

print(f"\nC++ Style Convolution Output:")
print(f"  Number of outputs: {len(outputs_cpp_style)}")
print(f"  Output[0] shape: {outputs_cpp_style[0].shape}")
print(f"  Value at [0,0,0]: {outputs_cpp_style[0][0, 0]}")

# Load actual C++ output for comparison
with open('/tmp/cpp_pnet_layer0_after_conv_output.bin', 'rb') as f:
    num_channels = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    cpp_output = np.fromfile(f, dtype=np.float32).reshape(num_channels, height, width)

print(f"\nC++ Actual Output (GOLD STANDARD):")
print(f"  Shape: {cpp_output.shape}")
print(f"  Value at [0,0,0]: {cpp_output[0, 0, 0]}")

# Convert Python outputs to numpy array
py_output = np.array(outputs_cpp_style)

# Compare
diff = np.abs(cpp_output - py_output)

print(f"\n{'='*80}")
print(f"COMPARISON: C++ STYLE IM2COL VS C++ ACTUAL")
print(f"{'='*80}")
print(f"Max difference: {diff.max():.10f}")
print(f"Mean difference: {diff.mean():.10f}")
print(f"Value at [0,0,0] - C++: {cpp_output[0,0,0]:.10f}, Python: {py_output[0,0,0]:.10f}")

# Show distribution
print(f"\nDifference distribution:")
print(f"  < 1e-7: {(diff < 1e-7).sum()} / {diff.size} = {100*(diff < 1e-7).sum()/diff.size:.1f}%")
print(f"  < 1e-6: {(diff < 1e-6).sum()} / {diff.size} = {100*(diff < 1e-6).sum()/diff.size:.1f}%")
print(f"  < 1e-5: {(diff < 1e-5).sum()} / {diff.size} = {100*(diff < 1e-5).sum()/diff.size:.1f}%")

# Channel-wise analysis
print(f"\nChannel-wise max differences:")
for ch in range(min(10, num_channels)):
    ch_max_diff = diff[ch].max()
    ch_mean_diff = diff[ch].mean()
    print(f"  Channel {ch}: max={ch_max_diff:.10f}, mean={ch_mean_diff:.10f}")

print(f"\n{'='*80}")
print(f"CONCLUSION:")
print(f"{'='*80}")
if diff.max() < 1e-5:
    print(f"✅ C++ IM2COL TECHNIQUE MATCHES C++ GOLD STANDARD!")
    print(f"   We've successfully replicated the C++ implementation!")
elif diff.max() < 1e-3:
    print(f"⚠️  Close match (max diff < 1e-3)")
else:
    print(f"❌ Still diverges (max diff = {diff.max():.6f})")
    print(f"   Need to investigate weight loading more carefully.")
