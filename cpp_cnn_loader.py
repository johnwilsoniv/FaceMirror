#!/usr/bin/env python3
"""
Pure Python CNN that loads C++ binary models (.dat files) for bit-for-bit matching.

This implementation reads the exact binary format used by OpenFace's C++ CNN_utils.cpp
and implements inference with exact C++ behavior, including:
- BGR channel ordering
- C++ im2col technique
- Exact PReLU: output = x if x >= 0 else x * slope
- C++ matrix multiply ordering
- C++ pooling and fully connected layers
"""

import numpy as np
import struct
from typing import List, Tuple, Optional


class CPPCNNLayer:
    """Base class for CNN layers"""
    def __init__(self, layer_type: int):
        self.layer_type = layer_type
        # Layer types from C++:
        # 0 = Convolutional
        # 1 = MaxPooling
        # 2 = FullyConnected
        # 3 = PReLU
        # 4 = Sigmoid

    def forward(self, x):
        raise NotImplementedError


class ConvLayer(CPPCNNLayer):
    """Convolutional layer matching C++ CNN_utils.cpp:445-549"""
    def __init__(self, num_in_maps: int, num_kernels: int, kernel_h: int, kernel_w: int,
                 kernels: np.ndarray, biases: np.ndarray):
        super().__init__(0)
        self.num_in_maps = num_in_maps
        self.num_kernels = num_kernels
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.kernels = kernels  # Shape: (num_kernels, num_in_maps, kernel_h, kernel_w)
        self.biases = biases    # Shape: (num_kernels,)

        # Precompute weight matrix in C++ format for efficient BLAS-style matmul
        self.weight_matrix = self._create_weight_matrix()

    def _create_weight_matrix(self) -> np.ndarray:
        """
        Create weight matrix in C++ format: (num_in_maps * stride + 1, num_kernels)

        Matches C++ FaceDetectorMTCNN.cpp:526-549:
        - kernels_rearr[k][i].t() transposes each kernel
        - Flattens to column vector
        - Adds bias as last row
        """
        stride = self.kernel_h * self.kernel_w

        # Create weight matrix
        weight_matrix = np.zeros((self.num_in_maps * stride, self.num_kernels), dtype=np.float32)

        for k in range(self.num_kernels):
            for i in range(self.num_in_maps):
                # Transpose kernel (C++ .t() operation)
                k_flat = self.kernels[k, i, :, :].T
                k_flat = k_flat.reshape(1, -1).T

                start_row = i * stride
                end_row = start_row + stride
                weight_matrix[start_row:end_row, k] = k_flat[:, 0]

        # Add bias row and transpose to (num_kernels, stride+1)
        weight_matrix = weight_matrix.T
        W = np.ones((self.num_kernels, self.num_in_maps * stride + 1), dtype=np.float32)
        W[:, :self.num_in_maps * stride] = weight_matrix
        W[:, -1] = self.biases

        return W.T  # (num_in_maps * stride + 1, num_kernels)

    def _im2col_multimap_cpp(self, inputs: List[np.ndarray]) -> np.ndarray:
        """
        Convert image to column format matching C++ CNN_utils.cpp:445-495

        Args:
            inputs: List of input feature maps [H, W]

        Returns:
            im2col matrix of shape (num_positions, num_features + 1)
        """
        num_maps = len(inputs)
        H, W = inputs[0].shape
        yB = H - self.kernel_h + 1
        xB = W - self.kernel_w + 1
        num_positions = yB * xB
        stride = self.kernel_h * self.kernel_w
        num_features = num_maps * stride

        # Initialize with ones for bias column
        im2col = np.ones((num_positions, num_features + 1), dtype=np.float32)

        # C++ column ordering: xx*height + yy + in_maps * stride
        for i in range(yB):
            for j in range(xB):
                row_idx = i * xB + j
                for yy in range(self.kernel_h):
                    for in_map_idx in range(num_maps):
                        for xx in range(self.kernel_w):
                            # C++ column index formula
                            col_idx = xx * self.kernel_h + yy + in_map_idx * stride
                            im2col[row_idx, col_idx] = inputs[in_map_idx][i + yy, j + xx]

        return im2col

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass using C++ BLAS convolution technique

        Args:
            x: Input of shape (num_in_maps, H, W) or (H, W, num_in_maps)

        Returns:
            Output of shape (num_kernels, out_h, out_w)
        """
        # Handle both channel-first and channel-last
        if x.ndim == 3:
            if x.shape[2] == self.num_in_maps:
                # Channel-last (H, W, C) → Channel-first (C, H, W)
                x = np.transpose(x, (2, 0, 1))

        # Split into list of 2D feature maps
        input_maps = [x[c, :, :] for c in range(self.num_in_maps)]

        # im2col
        im2col = self._im2col_multimap_cpp(input_maps)

        # Matrix multiply: im2col @ weight_matrix
        output = im2col @ self.weight_matrix  # (num_positions, num_kernels)

        # Reshape to (num_kernels, out_h, out_w)
        H, W = input_maps[0].shape
        out_h = H - self.kernel_h + 1
        out_w = W - self.kernel_w + 1
        output = output.T.reshape(self.num_kernels, out_h, out_w)

        return output


class MaxPoolLayer(CPPCNNLayer):
    """Max pooling layer matching C++ CNN_utils.cpp"""
    def __init__(self, kernel_size: int, stride: int = None):
        super().__init__(1)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Max pooling forward pass

        Args:
            x: Input of shape (num_maps, H, W)

        Returns:
            Output of shape (num_maps, out_h, out_w)
        """
        num_maps, H, W = x.shape
        # Use ROUND like C++ CNN_utils.cpp line 169-170
        out_h = int(round((H - self.kernel_size) / self.stride)) + 1
        out_w = int(round((W - self.kernel_size) / self.stride)) + 1

        output = np.zeros((num_maps, out_h, out_w), dtype=np.float32)

        for c in range(num_maps):
            for i in range(out_h):
                for j in range(out_w):
                    y = i * self.stride
                    x_pos = j * self.stride
                    window = x[c, y:y+self.kernel_size, x_pos:x_pos+self.kernel_size]
                    output[c, i, j] = window.max()

        return output


class FullyConnectedLayer(CPPCNNLayer):
    """Fully connected layer matching C++ CNN_utils.cpp

    Supports both modes:
    - Regular FC: (C, H, W) -> (output_size,) for fixed-size inputs
    - Fully convolutional: (C, H, W) -> (output_size, H, W) for PNet
    """
    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        super().__init__(2)
        self.weights = weights  # Shape: (output_size, input_size)
        self.biases = biases    # Shape: (output_size,)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Fully connected forward pass with support for spatial dimensions.

        Args:
            x: Input shape (C, H, W) or (features,)

        Returns:
            - If C*H*W matches expected input_size: Regular FC output (output_size,)
            - Otherwise (PNet fully convolutional): output (output_size, H, W)
        """
        # Handle 1D input (from previous FC layer)
        if x.ndim == 1:
            return self.weights @ x + self.biases

        C, H, W = x.shape
        expected_input_size = self.weights.shape[1]
        actual_flat_size = C * H * W

        # Check if input size matches what FC layer expects
        if actual_flat_size == expected_input_size:
            # Regular FC mode (for RNet, ONet with fixed input size)
            # Flatten in C-major order (C, H, W) -> flatten
            x_flat = x.flatten()
            return self.weights @ x_flat + self.biases
        elif H > 1 or W > 1:
            # Fully convolutional mode (for PNet)
            # Treat FC as 1x1 convolution applied to each spatial position
            output = np.zeros((self.biases.shape[0], H, W), dtype=np.float32)

            for i in range(H):
                for j in range(W):
                    # Extract feature vector at position (i, j)
                    x_vec = x[:, i, j]  # Shape: (C,)
                    # Apply FC layer
                    output[:, i, j] = self.weights @ x_vec + self.biases

            return output
        else:
            # H=W=1, just flatten
            x_flat = x.flatten()
            return self.weights @ x_flat + self.biases


class PReLULayer(CPPCNNLayer):
    """PReLU layer matching C++ CNN_utils.cpp:56-63 EXACTLY"""
    def __init__(self, slopes: np.ndarray):
        super().__init__(3)
        self.slopes = slopes  # Shape: (num_channels,)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        PReLU forward pass: output = x if x >= 0 else x * slope

        Matches C++ exactly:
        ```cpp
        float neg_mult = prelu_weights.at<float>(k);
        for (unsigned int i = 0; i < size_in; ++i)
        {
            float in_val = *iter;
            *iter++ = in_val >= 0 ? in_val : in_val * neg_mult;
        }
        ```

        Args:
            x: Input of shape (num_channels, H, W) or (num_features,)

        Returns:
            Output of same shape as input
        """
        output = np.zeros_like(x)

        if x.ndim == 1:
            # Handle 1D input (from FC layers)
            num_features = x.shape[0]
            for k in range(num_features):
                neg_mult = self.slopes[k]
                output[k] = x[k] if x[k] >= 0 else x[k] * neg_mult
        else:
            # Handle 3D input (from Conv layers)
            num_channels = x.shape[0]
            for k in range(num_channels):
                neg_mult = self.slopes[k]
                channel_data = x[k, :, :]
                # Apply PReLU element-wise
                output[k, :, :] = np.where(channel_data >= 0, channel_data, channel_data * neg_mult)

        return output


class SigmoidLayer(CPPCNNLayer):
    """Sigmoid layer"""
    def __init__(self):
        super().__init__(4)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid forward pass"""
        return 1.0 / (1.0 + np.exp(-x))


class CPPCNN:
    """
    CNN that loads C++ binary models for bit-for-bit matching.

    Binary format from CNN_utils.cpp:454-575:
    - int32: network_depth (number of layers)
    - For each layer:
      - int32: layer_type (0=conv, 1=pool, 2=FC, 3=PReLU, 4=sigmoid)
      - Layer-specific parameters (see _read_layer)
    """

    def __init__(self, model_path: str):
        """
        Load CNN from C++ binary .dat file

        Args:
            model_path: Path to .dat file (e.g., PNet.dat, RNet.dat, ONet.dat)
        """
        self.layers: List[CPPCNNLayer] = []
        self._load_from_binary(model_path)

    def _read_int32(self, f) -> int:
        """Read single int32 from binary file"""
        return struct.unpack('<i', f.read(4))[0]

    def _read_float32(self, f) -> float:
        """Read single float32 from binary file"""
        return struct.unpack('<f', f.read(4))[0]

    def _read_float32_array(self, f, count: int) -> np.ndarray:
        """Read array of float32 values"""
        return np.array(struct.unpack(f'<{count}f', f.read(4 * count)), dtype=np.float32)

    def _read_matrix_bin(self, f) -> np.ndarray:
        """
        Read matrix written by MATLAB writeMatrixBin function

        Format:
        - uint32: rows
        - uint32: cols
        - uint32: type (5 = float32)
        - float32[rows*cols]: transposed data (need to transpose back)
        """
        rows = self._read_int32(f)
        cols = self._read_int32(f)
        mat_type = self._read_int32(f)

        # Read data (stored transposed)
        data = self._read_float32_array(f, rows * cols)

        # Reshape and transpose back to original form
        # MATLAB writes M' (transposed), so we need to transpose back
        matrix = data.reshape(rows, cols).T

        return matrix

    def _read_conv_layer(self, f) -> ConvLayer:
        """
        Read convolutional layer from binary file

        MATLAB format (Write_CNN_to_binary.m:17-43):
        - uint32: num_in_maps
        - uint32: num_kernels
        - float32[num_kernels]: biases
        - For each input map (k=1 to num_in_maps):
          - For each output kernel (k2=1 to num_kernels):
            - writeMatrixBin(W) where W is the kernel matrix
        """
        num_in_maps = self._read_int32(f)
        num_kernels = self._read_int32(f)

        # Read biases (raw floats, not writeMatrixBin)
        biases = self._read_float32_array(f, num_kernels)

        # Read kernels using writeMatrixBin format
        kernel_h = None
        kernel_w = None
        kernels = []

        for i in range(num_in_maps):
            kernel_map = []
            for k in range(num_kernels):
                # Read kernel using writeMatrixBin format
                kernel = self._read_matrix_bin(f)

                if kernel_h is None:
                    kernel_h, kernel_w = kernel.shape

                kernel_map.append(kernel)
            kernels.append(kernel_map)

        # Rearrange to (num_kernels, num_in_maps, h, w)
        kernels_array = np.zeros((num_kernels, num_in_maps, kernel_h, kernel_w), dtype=np.float32)
        for i in range(num_in_maps):
            for k in range(num_kernels):
                kernels_array[k, i, :, :] = kernels[i][k]

        return ConvLayer(num_in_maps, num_kernels, kernel_h, kernel_w, kernels_array, biases)

    def _read_pool_layer(self, f) -> MaxPoolLayer:
        """
        Read max pooling layer from binary file

        MATLAB format (Write_CNN_to_binary.m:54-60):
        - uint32: kernel_size_x
        - uint32: kernel_size_y
        - uint32: stride_x
        - uint32: stride_y
        """
        kernel_size_x = self._read_int32(f)
        kernel_size_y = self._read_int32(f)
        stride_x = self._read_int32(f)
        stride_y = self._read_int32(f)

        # Assuming square kernels and uniform stride for simplicity
        # (MTCNN uses square kernels)
        return MaxPoolLayer(kernel_size_x, stride_x)

    def _read_fc_layer(self, f) -> FullyConnectedLayer:
        """
        Read fully connected layer from binary file

        MATLAB format (Write_CNN_to_binary.m:44-52):
        - writeMatrixBin(biases)
        - writeMatrixBin(weights)
        """
        # Read biases first (MATLAB writes biases before weights for FC)
        biases_matrix = self._read_matrix_bin(f)
        biases = biases_matrix.flatten()

        # Read weights
        weights = self._read_matrix_bin(f)

        return FullyConnectedLayer(weights, biases)

    def _read_prelu_layer(self, f) -> PReLULayer:
        """
        Read PReLU layer from binary file

        MATLAB format (Write_CNN_to_binary.m:62-64):
        - writeMatrixBin(weights)
        """
        slopes_matrix = self._read_matrix_bin(f)
        slopes = slopes_matrix.flatten()
        return PReLULayer(slopes)

    def _read_sigmoid_layer(self, f) -> SigmoidLayer:
        """Read sigmoid layer (no parameters)"""
        return SigmoidLayer()

    def _read_layer(self, f, layer_type: int) -> CPPCNNLayer:
        """Read layer based on type"""
        if layer_type == 0:
            return self._read_conv_layer(f)
        elif layer_type == 1:
            return self._read_pool_layer(f)
        elif layer_type == 2:
            return self._read_fc_layer(f)
        elif layer_type == 3:
            return self._read_prelu_layer(f)
        elif layer_type == 4:
            return self._read_sigmoid_layer(f)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    def _load_from_binary(self, model_path: str):
        """Load CNN from C++ binary .dat file"""
        with open(model_path, 'rb') as f:
            # Read network depth
            network_depth = self._read_int32(f)

            print(f"Loading CNN from {model_path}")
            print(f"Network depth: {network_depth} layers")

            # Read each layer
            for i in range(network_depth):
                layer_type = self._read_int32(f)
                layer = self._read_layer(f, layer_type)
                self.layers.append(layer)

                # Print layer info
                layer_names = ["Conv", "MaxPool", "FC", "PReLU", "Sigmoid"]
                print(f"  Layer {i}: {layer_names[layer_type]}", end="")

                if isinstance(layer, ConvLayer):
                    print(f" ({layer.num_in_maps}→{layer.num_kernels}, {layer.kernel_h}x{layer.kernel_w})")
                elif isinstance(layer, MaxPoolLayer):
                    print(f" ({layer.kernel_size}x{layer.kernel_size}, stride={layer.stride})")
                elif isinstance(layer, FullyConnectedLayer):
                    print(f" ({layer.weights.shape[1]}→{layer.weights.shape[0]})")
                elif isinstance(layer, PReLULayer):
                    print(f" ({len(layer.slopes)} channels)")
                else:
                    print()

            print(f"Successfully loaded {network_depth} layers!")

    def forward(self, x: np.ndarray) -> List[np.ndarray]:
        """
        Forward pass through network

        Args:
            x: Input tensor

        Returns:
            List of outputs (multiple heads for MTCNN)
        """
        outputs = []
        current = x

        for i, layer in enumerate(self.layers):
            current = layer.forward(current)

            # MTCNN networks output intermediate results
            # We'll return all outputs after FC/Sigmoid layers
            if isinstance(layer, (FullyConnectedLayer, SigmoidLayer)):
                outputs.append(current.copy())

        return outputs if outputs else [current]

    def __call__(self, x: np.ndarray) -> List[np.ndarray]:
        """Convenience method for forward pass"""
        return self.forward(x)


if __name__ == "__main__":
    # Test loading C++ binary models
    import os

    # Path to C++ binary models
    model_dir = os.path.expanduser("~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/face_detection/mtcnn/convert_to_cpp/")

    print("="*80)
    print("Testing C++ CNN Loader")
    print("="*80)

    # Load PNet
    pnet_path = os.path.join(model_dir, "PNet.dat")
    if os.path.exists(pnet_path):
        print("\nLoading PNet...")
        pnet = CPPCNN(pnet_path)
        print(f"✓ PNet loaded successfully with {len(pnet.layers)} layers")
    else:
        print(f"✗ PNet not found at {pnet_path}")

    # Load RNet
    rnet_path = os.path.join(model_dir, "RNet.dat")
    if os.path.exists(rnet_path):
        print("\nLoading RNet...")
        rnet = CPPCNN(rnet_path)
        print(f"✓ RNet loaded successfully with {len(rnet.layers)} layers")
    else:
        print(f"✗ RNet not found at {rnet_path}")

    # Load ONet
    onet_path = os.path.join(model_dir, "ONet.dat")
    if os.path.exists(onet_path):
        print("\nLoading ONet...")
        onet = CPPCNN(onet_path)
        print(f"✓ ONet loaded successfully with {len(onet.layers)} layers")
    else:
        print(f"✗ ONet not found at {onet_path}")

    print("\n" + "="*80)
    print("Ready to implement pure Python MTCNN detector!")
    print("="*80)
