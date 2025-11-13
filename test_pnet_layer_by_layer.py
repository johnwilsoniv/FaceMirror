#!/usr/bin/env python3
"""
Layer-by-layer PNet test to find divergence point.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class PNetDebug(nn.Module):
    """PNet with layer-by-layer output capture."""

    def __init__(self, weights_dir='cpp_mtcnn_weights/pnet'):
        super().__init__()
        self.weights_dir = Path(weights_dir)

        # Layer 0: Conv 3->10, 3x3
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=0, bias=True)
        self._load_conv(self.conv1, 'pnet_layer00_conv')

        # Layer 1: PReLU
        self.prelu1 = nn.PReLU(10)
        self._load_prelu(self.prelu1, 'pnet_layer01_prelu')

        # Layer 2: MaxPool 2x2, stride 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # Layer 3: Conv 10->16, 3x3
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=0, bias=True)
        self._load_conv(self.conv2, 'pnet_layer03_conv')

        # Layer 4: PReLU
        self.prelu2 = nn.PReLU(16)
        self._load_prelu(self.prelu2, 'pnet_layer04_prelu')

        # Layer 5: Conv 16->32, 3x3
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, bias=True)
        self._load_conv(self.conv3, 'pnet_layer05_conv')

        # Layer 6: PReLU
        self.prelu3 = nn.PReLU(32)
        self._load_prelu(self.prelu3, 'pnet_layer06_prelu')

        # Layer 7: FC implemented as 1x1 conv for fully convolutional
        self.conv4 = nn.Conv2d(32, 6, kernel_size=1, stride=1, padding=0, bias=True)
        self._load_fc_as_conv(self.conv4, 'pnet_layer07_fc')

    def _load_conv(self, layer, prefix):
        """Load conv layer weights."""
        w = np.load(self.weights_dir / f'{prefix}_weights.npy')
        b = np.load(self.weights_dir / f'{prefix}_biases.npy')
        layer.weight.data = torch.from_numpy(w)
        layer.bias.data = torch.from_numpy(b)
        print(f"  Loaded {prefix}: weights {w.shape}, bias {b.shape}")

    def _load_fc_as_conv(self, layer, prefix):
        """Load FC layer as 1x1 conv."""
        w = np.load(self.weights_dir / f'{prefix}_weights.npy')  # (6, 32)
        b = np.load(self.weights_dir / f'{prefix}_bias.npy')
        # Reshape FC to conv: (6, 32) -> (6, 32, 1, 1)
        w_conv = w.reshape(w.shape[0], w.shape[1], 1, 1)
        layer.weight.data = torch.from_numpy(w_conv)
        layer.bias.data = torch.from_numpy(b)
        print(f"  Loaded {prefix}: FC {w.shape} -> Conv {w_conv.shape}, bias {b.shape}")

    def _load_prelu(self, layer, prefix):
        """Load PReLU weights."""
        w = np.load(self.weights_dir / f'{prefix}_weights.npy')
        layer.weight.data = torch.from_numpy(w)
        print(f"  Loaded {prefix}: {w.shape}")

    def forward(self, x):
        """Forward pass capturing all intermediate outputs."""
        outputs = {}

        # Layer 0: Conv 3->10
        x = self.conv1(x)
        outputs['layer0_conv'] = x.clone()

        # Layer 1: PReLU
        x = self.prelu1(x)
        outputs['layer1_prelu'] = x.clone()

        # Layer 2: MaxPool
        x = self.pool1(x)
        outputs['layer2_pool'] = x.clone()

        # Layer 3: Conv 10->16
        x = self.conv2(x)
        outputs['layer3_conv'] = x.clone()

        # Layer 4: PReLU
        x = self.prelu2(x)
        outputs['layer4_prelu'] = x.clone()

        # Layer 5: Conv 16->32
        x = self.conv3(x)
        outputs['layer5_conv'] = x.clone()

        # Layer 6: PReLU
        x = self.prelu3(x)
        outputs['layer6_prelu'] = x.clone()

        # Layer 7: FC as 1x1 conv
        x = self.conv4(x)
        outputs['layer7_fc'] = x.clone()

        return x, outputs


def main():
    print("="*80)
    print("PNET LAYER-BY-LAYER DEBUGGING")
    print("="*80)

    # Load C++ input
    print("\nLoading C++ PNet input...")
    cpp_input = np.fromfile('/tmp/cpp_pnet_input_scale0.bin', dtype=np.float32)
    cpp_input = cpp_input.reshape(384, 216, 3)  # HWC
    print(f"  C++ input shape: {cpp_input.shape}")
    print(f"  Sample pixel [0,0]: {cpp_input[0,0,:]}")

    # Convert to PyTorch (HWC -> CHW)
    py_input = torch.from_numpy(cpp_input).permute(2, 0, 1).unsqueeze(0).float()
    print(f"  PyTorch input shape: {py_input.shape}")

    # Load C++ final outputs
    print("\nLoading C++ PNet final outputs...")
    cpp_logit0 = np.fromfile('/tmp/cpp_pnet_logit0_scale0.bin', dtype=np.float32)
    cpp_logit1 = np.fromfile('/tmp/cpp_pnet_logit1_scale0.bin', dtype=np.float32)
    cpp_logit0 = cpp_logit0.reshape(187, 103)
    cpp_logit1 = cpp_logit1.reshape(187, 103)
    print(f"  C++ logit0 shape: {cpp_logit0.shape}")
    print(f"  C++ logit0[0,0]: {cpp_logit0[0,0]:.6f}")
    print(f"  C++ logit1[0,0]: {cpp_logit1[0,0]:.6f}")

    # Create and run Python PNet
    print("\nBuilding Python PNet...")
    model = PNetDebug()
    model.eval()

    print("\nRunning Python PNet forward pass...")
    with torch.no_grad():
        output, layer_outputs = model(py_input)

    print(f"\nPython PNet final output shape: {output.shape}")

    # Extract classification logits (channels 0 and 1)
    py_logit0 = output[0, 0, :, :].numpy()
    py_logit1 = output[0, 1, :, :].numpy()
    print(f"  Python logit0 shape: {py_logit0.shape}")
    print(f"  Python logit0[0,0]: {py_logit0[0,0]:.6f}")
    print(f"  Python logit1[0,0]: {py_logit1[0,0]:.6f}")

    # Compare final outputs
    print(f"\n{'='*80}")
    print(f"FINAL OUTPUT COMPARISON:")
    print(f"{'='*80}")
    diff0 = np.abs(cpp_logit0 - py_logit0)
    diff1 = np.abs(cpp_logit1 - py_logit1)
    print(f"Logit0 differences: mean={diff0.mean():.6f}, max={diff0.max():.6f}")
    print(f"Logit1 differences: mean={diff1.mean():.6f}, max={diff1.max():.6f}")

    # Show all layer shapes
    print(f"\n{'='*80}")
    print(f"ALL LAYER OUTPUTS:")
    print(f"{'='*80}")
    for name, tensor in layer_outputs.items():
        print(f"  {name:20s} {list(tensor.shape)}")

    # Check specific layer outputs
    print(f"\n{'='*80}")
    print(f"LAYER 0 (First Conv) CHECK:")
    print(f"{'='*80}")
    layer0_out = layer_outputs['layer0_conv'][0].numpy()
    print(f"  Shape: {layer0_out.shape}")
    print(f"  Sample [0,0,0]: {layer0_out[0,0,0]:.6f}")
    print(f"  Stats: min={layer0_out.min():.6f}, max={layer0_out.max():.6f}, mean={layer0_out.mean():.6f}")

    print(f"\n{'='*80}")
    print(f"ASSESSMENT:")
    print(f"{'='*80}")
    if diff0.max() < 0.01 and diff1.max() < 0.01:
        print(f"✅ PNet outputs MATCH! (diff < 0.01)")
    elif diff0.max() < 0.1 and diff1.max() < 0.1:
        print(f"⚠️  PNet outputs MOSTLY match (diff < 0.1)")
        print(f"   Some numerical precision issues")
    else:
        print(f"❌ PNet outputs DO NOT MATCH!")
        print(f"   Need to check intermediate layers to find divergence")
        print(f"   Consider adding C++ layer 0 logging to compare directly")


if __name__ == '__main__':
    main()
