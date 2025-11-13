#!/usr/bin/env python3
"""
Test PyTorch ONet directly (no ONNX) to find next divergence point.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class ONet(nn.Module):
    """PyTorch ONet matching C++ structure."""

    def __init__(self, weights_dir='cpp_mtcnn_weights/onet'):
        super().__init__()
        self.weights_dir = Path(weights_dir)

        # Layer 0: Conv 3x32x3x3
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 0)
        self._load_conv(self.conv1, 'onet_layer00_conv')

        # Layer 1: PReLU
        self.prelu1 = nn.PReLU(32)
        self._load_prelu(self.prelu1, 'onet_layer01_prelu')

        # Layer 2: MaxPool 3x3 stride 2
        self.pool1 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)

        # Layer 3: Conv 32x64x3x3
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0)
        self._load_conv(self.conv2, 'onet_layer03_conv')

        # Layer 4: PReLU
        self.prelu2 = nn.PReLU(64)
        self._load_prelu(self.prelu2, 'onet_layer04_prelu')

        # Layer 5: MaxPool 3x3 stride 2
        self.pool2 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)

        # Layer 6: Conv 64x64x3x3
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self._load_conv(self.conv3, 'onet_layer06_conv')

        # Layer 7: PReLU
        self.prelu3 = nn.PReLU(64)
        self._load_prelu(self.prelu3, 'onet_layer07_prelu')

        # Layer 8: MaxPool 2x2 stride 2
        self.pool3 = nn.MaxPool2d(2, 2, 0, ceil_mode=True)

        # Layer 9: Conv 64x128x2x2
        self.conv4 = nn.Conv2d(64, 128, 2, 1, 0)
        self._load_conv(self.conv4, 'onet_layer09_conv')

        # Layer 10: PReLU
        self.prelu4 = nn.PReLU(128)
        self._load_prelu(self.prelu4, 'onet_layer10_prelu')

        # Layer 11: FC 1152→256
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self._load_fc(self.fc1, 'onet_layer11_fc')

        # Layer 12: PReLU
        self.prelu5 = nn.PReLU(256)
        self._load_prelu(self.prelu5, 'onet_layer12_prelu')

        # Layer 13: FC 256→16 (2 classification + 4 bbox + 10 landmarks)
        self.fc2 = nn.Linear(256, 16)
        self._load_fc(self.fc2, 'onet_layer13_fc')

    def _load_conv(self, layer, prefix):
        """Load conv layer weights."""
        w = np.load(self.weights_dir / f'{prefix}_weights.npy')
        b = np.load(self.weights_dir / f'{prefix}_biases.npy')
        layer.weight.data = torch.from_numpy(w)
        layer.bias.data = torch.from_numpy(b)

    def _load_fc(self, layer, prefix):
        """Load FC layer weights."""
        w = np.load(self.weights_dir / f'{prefix}_weights.npy')
        b = np.load(self.weights_dir / f'{prefix}_bias.npy')
        layer.weight.data = torch.from_numpy(w)
        layer.bias.data = torch.from_numpy(b)

    def _load_prelu(self, layer, prefix):
        """Load PReLU weights."""
        w = np.load(self.weights_dir / f'{prefix}_weights.npy')
        layer.weight.data = torch.from_numpy(w)

    def forward(self, x):
        """Forward pass returning all intermediate outputs."""
        outputs = {}

        # Layer 0: Conv 3→32
        x = self.conv1(x)
        outputs['layer0_conv'] = x.clone()

        # Layer 1: PReLU
        x = self.prelu1(x)
        outputs['layer1_prelu'] = x.clone()

        # Layer 2: MaxPool
        x = self.pool1(x)
        outputs['layer2_pool'] = x.clone()

        # Layer 3: Conv 32→64
        x = self.conv2(x)
        outputs['layer3_conv'] = x.clone()

        # Layer 4: PReLU
        x = self.prelu2(x)
        outputs['layer4_prelu'] = x.clone()

        # Layer 5: MaxPool
        x = self.pool2(x)
        outputs['layer5_pool'] = x.clone()

        # Layer 6: Conv 64→64
        x = self.conv3(x)
        outputs['layer6_conv'] = x.clone()

        # Layer 7: PReLU
        x = self.prelu3(x)
        outputs['layer7_prelu'] = x.clone()

        # Layer 8: MaxPool
        x = self.pool3(x)
        outputs['layer8_pool'] = x.clone()

        # Layer 9: Conv 64→128
        x = self.conv4(x)
        outputs['layer9_conv'] = x.clone()

        # Layer 10: PReLU
        x = self.prelu4(x)
        outputs['layer10_prelu'] = x.clone()

        # Layer 11: Flatten + FC 1152→256
        x = x.view(x.size(0), -1)
        outputs['layer11_flatten'] = x.clone()
        x = self.fc1(x)
        outputs['layer11_fc'] = x.clone()

        # Layer 12: PReLU
        x = self.prelu5(x)
        outputs['layer12_prelu'] = x.clone()

        # Layer 13: FC 256→16 (combined output)
        combined = self.fc2(x)
        outputs['layer13_combined'] = combined.clone()

        # Split into classification, bbox, landmarks
        cls = combined[:, :2]
        bbox = combined[:, 2:6]
        landmarks = combined[:, 6:16]

        return cls, bbox, landmarks, outputs


def main():
    print("="*80)
    print("PYTORCH ONET TEST - Finding Next Divergence Point")
    print("="*80)

    # Load C++ reference input
    print("\nLoading C++ reference input...")
    with open('/tmp/cpp_onet_input.bin', 'rb') as f:
        input_data = np.frombuffer(f.read(), dtype=np.float32)

    # C++ saves as HWC, convert to CHW for PyTorch
    input_hwc = input_data.reshape(48, 48, 3)
    input_chw = input_hwc.transpose(2, 0, 1).copy()  # Make writable copy

    # Add batch dimension
    input_tensor = torch.from_numpy(input_chw).unsqueeze(0)
    print(f"Input shape: {input_tensor.shape}")

    # Load C++ reference outputs
    print("\nLoading C++ reference outputs...")
    cpp_outputs = {}

    # Layer 0 output (32 channels) - skip 12-byte header (3 uint32 dimensions)
    with open('/tmp/cpp_layer0_after_conv_output.bin', 'rb') as f:
        import struct
        channels = struct.unpack('<I', f.read(4))[0]
        height = struct.unpack('<I', f.read(4))[0]
        width = struct.unpack('<I', f.read(4))[0]
        layer0_data = np.frombuffer(f.read(), dtype=np.float32)
    cpp_outputs['layer0'] = layer0_data.reshape(channels, height, width)
    print(f"C++ Layer 0 shape: {cpp_outputs['layer0'].shape}")

    # Create PyTorch model
    print("\nBuilding PyTorch ONet...")
    model = ONet()
    model.eval()

    # Run forward pass
    print("\nRunning PyTorch forward pass...")
    with torch.no_grad():
        cls, bbox, landmarks, outputs = model(input_tensor)

    print("\n" + "="*80)
    print("LAYER-BY-LAYER COMPARISON")
    print("="*80)

    # Compare Layer 0 (we know this should match now)
    print("\nLayer 0: Conv 3→32")
    py_layer0 = outputs['layer0_conv'][0].numpy()
    cpp_layer0 = cpp_outputs['layer0']

    diff = np.abs(py_layer0 - cpp_layer0)
    print(f"  PyTorch output [0,0,0]: {py_layer0[0,0,0]:.6f}")
    print(f"  C++ output [0,0,0]:     {cpp_layer0[0,0,0]:.6f}")
    print(f"  Max difference: {diff.max():.2e}")
    print(f"  Mean difference: {diff.mean():.2e}")

    if diff.max() < 1e-5:
        print("  ✅ MATCH!")
    else:
        print("  ❌ DIVERGENCE FOUND!")
        return

    # Compare final classification output
    print("\nFinal Classification Output:")
    py_cls = cls[0].numpy()
    cpp_cls = np.array([-3.250, 3.249])  # From /tmp/cpp_onet_debug.txt

    print(f"  PyTorch logits: [{py_cls[0]:.3f}, {py_cls[1]:.3f}]")
    print(f"  C++ logits:     [{cpp_cls[0]:.3f}, {cpp_cls[1]:.3f}]")

    diff = np.abs(py_cls - cpp_cls)
    print(f"  Absolute difference: [{diff[0]:.3f}, {diff[1]:.3f}]")
    print(f"  Relative error: {100 * diff.max() / np.abs(cpp_cls).max():.1f}%")

    if diff.max() < 0.01:
        print("  ✅ MATCH!")
    else:
        print("  ❌ Still some divergence")
        print("\n  Need to check intermediate layers to find where error accumulates...")

    # Show all intermediate layer shapes
    print("\n" + "="*80)
    print("ALL LAYER OUTPUTS")
    print("="*80)
    for name, tensor in outputs.items():
        print(f"  {name:25s} {list(tensor.shape)}")


if __name__ == '__main__':
    main()
