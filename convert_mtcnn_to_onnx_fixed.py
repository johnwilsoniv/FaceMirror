#!/usr/bin/env python3
"""
Convert Extracted C++ MTCNN Weights to ONNX

Builds PyTorch models from extracted weights and exports to ONNX format
for use with onnxruntime (CoreML acceleration on ARM Mac).
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json


class PReLU(nn.Module):
    """PReLU activation from MTCNN (per-channel)."""
    def __init__(self, num_channels, for_fc=False):
        super().__init__()
        # For conv layers: (1, num_channels, 1, 1)
        # For FC layers: (num_channels,)
        if for_fc:
            self.weight = nn.Parameter(torch.zeros(num_channels))
        else:
            self.weight = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        return torch.max(x, torch.zeros_like(x)) + self.weight * torch.min(x, torch.zeros_like(x))


class PNet(nn.Module):
    """Proposal Network (PNet) from MTCNN."""
    def __init__(self):
        super().__init__()

        # Layer 0: Conv 3->10, 3x3
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=0, bias=True)
        self.prelu1 = PReLU(10)

        # Layer 2: MaxPool 2x2, stride 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # Layer 3: Conv 10->16, 3x3
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=0, bias=True)
        self.prelu2 = PReLU(16)

        # Layer 5: Conv 16->32, 3x3
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, bias=True)
        self.prelu3 = PReLU(32)

        # Layer 7: FC (implemented as 1x1 conv for fully convolutional)
        # Output: 2 (classification) + 4 (bbox regression) = 6
        self.conv4 = nn.Conv2d(32, 6, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = self.conv4(x)
        return x


class RNet(nn.Module):
    """Refinement Network (RNet) from MTCNN."""
    def __init__(self):
        super().__init__()

        # Layer 0: Conv 3->28, 3x3
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3, stride=1, padding=0, bias=True)
        self.prelu1 = PReLU(28)

        # Layer 2: MaxPool 3x3, stride 2
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Layer 3: Conv 28->48, 3x3
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3, stride=1, padding=0, bias=True)
        self.prelu2 = PReLU(48)

        # Layer 5: MaxPool 3x3, stride 2
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Layer 6: Conv 48->64, 2x2
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2, stride=1, padding=0, bias=True)
        self.prelu3 = PReLU(64)

        # Layer 8: FC 576->128 (64 * 3 * 3 = 576 after convolutions)
        self.fc1 = nn.Linear(576, 128, bias=True)
        self.prelu4 = PReLU(128, for_fc=True)  # FC layer needs 1D PReLU

        # Layer 10: FC 128->6
        self.fc2 = nn.Linear(128, 6, bias=True)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)
        x = self.prelu3(self.conv3(x))
        # CRITICAL: C++ transposes each feature map before flattening (Pitfall #4)
        # x shape: (batch, C, H, W) → transpose to (batch, C, W, H) → flatten
        x = x.transpose(2, 3).contiguous()  # Transpose H and W
        x = x.view(x.size(0), -1)  # Flatten
        x = self.prelu4(self.fc1(x))
        x = self.fc2(x)
        return x


class ONet(nn.Module):
    """Output Network (ONet) from MTCNN."""
    def __init__(self):
        super().__init__()

        # Layer 0: Conv 3->32, 3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=True)
        self.prelu1 = PReLU(32)

        # Layer 2: MaxPool 3x3, stride 2
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Layer 3: Conv 32->64, 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=True)
        self.prelu2 = PReLU(64)

        # Layer 5: MaxPool 3x3, stride 2
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Layer 6: Conv 64->64, 3x3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=True)
        self.prelu3 = PReLU(64)

        # Layer 8: MaxPool 2x2, stride 2
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # Layer 9: Conv 64->128, 2x2
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0, bias=True)
        self.prelu4 = PReLU(128)

        # Layer 11: FC 1152->256 (128 * 3 * 3 = 1152 after convolutions for 48x48 input)
        self.fc1 = nn.Linear(1152, 256, bias=True)
        self.prelu5 = PReLU(256, for_fc=True)  # FC layer needs 1D PReLU

        # Layer 13: FC 256->16 (2 classification + 4 bbox + 10 landmarks = 16)
        self.fc2 = nn.Linear(256, 16, bias=True)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)
        x = self.prelu3(self.conv3(x))
        x = self.pool3(x)
        x = self.prelu4(self.conv4(x))
        # CRITICAL: C++ transposes each feature map before flattening (Pitfall #4)
        # x shape: (batch, C, H, W) → transpose to (batch, C, W, H) → flatten
        x = x.transpose(2, 3).contiguous()  # Transpose H and W
        x = x.view(x.size(0), -1)  # Flatten
        x = self.prelu5(self.fc1(x))
        x = self.fc2(x)
        return x


def load_weights_to_model(model, weights_dir, net_name):
    """Load extracted weights into PyTorch model."""
    weights_dir = Path(weights_dir)
    structure_file = weights_dir / 'structure.json'

    with open(structure_file, 'r') as f:
        structure = json.load(f)

    print(f"  Loading {net_name} weights...")

    # Map layer indices to model parameters
    conv_idx = 0
    fc_idx = 0
    prelu_idx = 0

    for i, layer_info in enumerate(structure['layers']):
        layer_type = layer_info['type']
        prefix = f"{net_name.lower()}_layer{i:02d}_{layer_type}"

        if layer_type == 'conv':
            # Load conv weights and biases
            weights_file = weights_dir / f"{prefix}_weights.npy"
            biases_file = weights_dir / f"{prefix}_biases.npy"

            weights = np.load(weights_file)
            biases = np.load(biases_file)

            # Get the corresponding conv layer in the model
            if conv_idx == 0:
                conv_layer = model.conv1
            elif conv_idx == 1:
                conv_layer = model.conv2
            elif conv_idx == 2:
                conv_layer = model.conv3
            elif conv_idx == 3:
                conv_layer = model.conv4 if hasattr(model, 'conv4') else None

            if conv_layer:
                # PyTorch expects (out_channels, in_channels, height, width)
                conv_layer.weight.data = torch.from_numpy(weights)
                conv_layer.bias.data = torch.from_numpy(biases)
                print(f"    Layer {i}: Conv {weights.shape} loaded")

            conv_idx += 1

        elif layer_type == 'fc':
            # Load fc weights and biases
            weights_file = weights_dir / f"{prefix}_weights.npy"
            bias_file = weights_dir / f"{prefix}_bias.npy"

            weights = np.load(weights_file)  # Shape: (out_features, in_features)
            bias = np.load(bias_file)

            # Special case: PNet layer 7 is FC in C++ but implemented as 1x1 conv in PyTorch
            if fc_idx == 0 and hasattr(model, 'conv4') and not hasattr(model, 'fc1'):
                # PNet: Reshape FC (6, 32) to Conv (6, 32, 1, 1)
                weights_conv = weights.reshape(weights.shape[0], weights.shape[1], 1, 1)
                model.conv4.weight.data = torch.from_numpy(weights_conv)
                model.conv4.bias.data = torch.from_numpy(bias)
                print(f"    Layer {i}: FC→Conv {weights.shape} → {weights_conv.shape} loaded into conv4")
            else:
                # Get the corresponding fc layer
                if fc_idx == 0:
                    fc_layer = model.fc1 if hasattr(model, 'fc1') else None
                elif fc_idx == 1:
                    fc_layer = model.fc2 if hasattr(model, 'fc2') else None
                else:
                    fc_layer = None

                if fc_layer:
                    # PyTorch Linear expects (out_features, in_features)
                    fc_layer.weight.data = torch.from_numpy(weights)
                    fc_layer.bias.data = torch.from_numpy(bias)
                    print(f"    Layer {i}: FC {weights.shape} loaded")

            fc_idx += 1

        elif layer_type == 'prelu':
            # Load prelu weights
            weights_file = weights_dir / f"{prefix}_weights.npy"
            weights = np.load(weights_file)

            # Get the corresponding prelu layer
            if prelu_idx == 0:
                prelu_layer = model.prelu1
            elif prelu_idx == 1:
                prelu_layer = model.prelu2
            elif prelu_idx == 2:
                prelu_layer = model.prelu3
            elif prelu_idx == 3:
                prelu_layer = model.prelu4 if hasattr(model, 'prelu4') else None
            elif prelu_idx == 4:
                prelu_layer = model.prelu5 if hasattr(model, 'prelu5') else None
            else:
                prelu_layer = None

            if prelu_layer:
                # Check if this is for FC layer or conv layer based on expected weight shape
                if len(prelu_layer.weight.shape) == 1:
                    # FC layer PReLU: (num_channels,)
                    weights_reshaped = weights.flatten()
                else:
                    # Conv layer PReLU: (1, num_channels, 1, 1)
                    weights_reshaped = weights.reshape(1, -1, 1, 1)
                prelu_layer.weight.data = torch.from_numpy(weights_reshaped)
                print(f"    Layer {i}: PReLU {weights.shape} loaded")

            prelu_idx += 1

    print(f"  ✓ All weights loaded for {net_name}")


def export_to_onnx(model, model_name, input_size, output_path):
    """Export PyTorch model to ONNX with dynamic input dimensions."""
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, *input_size)

    # Export with dynamic axes for batch_size, height, width
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"  ✓ Exported {model_name} to {output_path} (dynamic input size)")


def main():
    """Convert extracted MTCNN weights to ONNX."""
    print("="*80)
    print("CONVERT C++ MTCNN WEIGHTS TO ONNX")
    print("="*80)

    weights_base = Path("cpp_mtcnn_weights")
    output_dir = Path("cpp_mtcnn_onnx")
    output_dir.mkdir(exist_ok=True)

    # PNet: 12x12 minimum input (fully convolutional - variable size)
    print("\n[1/3] PNet (Proposal Network)")
    pnet = PNet()
    load_weights_to_model(pnet, weights_base / "pnet", "pnet")

    # PNet is fully convolutional, export with dynamic spatial dimensions
    pnet.eval()
    dummy_input = torch.randn(1, 3, 12, 12)
    torch.onnx.export(
        pnet,
        dummy_input,
        output_dir / "pnet.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'out_height', 3: 'out_width'}
        }
    )
    print(f"  ✓ Exported PNet to {output_dir / 'pnet.onnx'} (fully convolutional)")

    # RNet: 24x24 fixed input (has FC layers)
    print("\n[2/3] RNet (Refinement Network)")
    rnet = RNet()
    load_weights_to_model(rnet, weights_base / "rnet", "rnet")

    # RNet has FC layers - only batch dimension is dynamic
    rnet.eval()
    dummy_input = torch.randn(1, 3, 24, 24)
    torch.onnx.export(
        rnet,
        dummy_input,
        output_dir / "rnet.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"  ✓ Exported RNet to {output_dir / 'rnet.onnx'} (fixed 24x24, dynamic batch)")

    # ONet: 48x48 fixed input (has FC layers)
    print("\n[3/3] ONet (Output Network)")
    onet = ONet()
    load_weights_to_model(onet, weights_base / "onet", "onet")

    # ONet has FC layers - only batch dimension is dynamic
    onet.eval()
    dummy_input = torch.randn(1, 3, 48, 48)
    torch.onnx.export(
        onet,
        dummy_input,
        output_dir / "onet.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"  ✓ Exported ONet to {output_dir / 'onet.onnx'} (fixed 48x48, dynamic batch)")

    print("\n" + "="*80)
    print("✅ ONNX CONVERSION COMPLETE")
    print("="*80)
    print(f"\nONNX models saved to: {output_dir.absolute()}")
    print(f"\nNext step: Use these ONNX models with onnxruntime for C++-identical MTCNN detection")


if __name__ == "__main__":
    main()
