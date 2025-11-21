#!/usr/bin/env python3
"""
Export CEN patch experts to CoreML format for Apple Neural Engine acceleration.

This script converts the existing CEN patch expert models (neural networks)
from the OpenFace binary format to CoreML .mlpackage format.

Usage:
    python pyclnf/backends/export_to_coreml.py

Output:
    pyclnf/models/coreml_cen/
        cen_lm00_scale0.25.mlpackage/
        cen_lm00_scale0.35.mlpackage/
        ...
        cen_lm67_scale1.0.mlpackage/

Requirements:
    pip install coremltools torch torchvision
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyclnf.core.cen_patch_expert import CENPatchExperts

try:
    import coremltools as ct
except ImportError:
    print("Error: coremltools not installed")
    print("Install with: pip install coremltools")
    sys.exit(1)


class CENPatchExpertPyTorch(nn.Module):
    """
    PyTorch wrapper for CEN patch expert (needed for CoreML conversion).
    """

    def __init__(self, expert):
        super().__init__()
        self.expert = expert

        # Convert weights and biases to PyTorch parameters
        self.layers = nn.ModuleList()

        for i in range(len(expert.weights)):
            layer = nn.Linear(
                expert.weights[i].shape[1],
                expert.weights[i].shape[0],
                bias=True
            )

            # Copy weights (note: OpenFace stores as output x input, PyTorch expects input x output)
            layer.weight.data = torch.from_numpy(expert.weights[i]).float()
            layer.bias.data = torch.from_numpy(expert.biases[i]).flatten().float()

            self.layers.append(layer)

        # Activation functions
        self.activations = expert.activation_function

    def forward(self, x):
        """
        Forward pass matching CEN patch expert computation.

        Input: (B, 1, H, W) - batch of grayscale patches
        Output: (B, 1, response_h, response_w) - response maps
        """
        batch_size, channels, height, width = x.shape

        # Extract patches using unfold (im2col equivalent)
        patch_size = 11
        patches = nn.functional.unfold(x, kernel_size=patch_size, stride=1)

        # patches shape: (B, patch_size*patch_size, num_patches)
        # Transpose to (B, num_patches, patch_size*patch_size)
        patches = patches.transpose(1, 2)

        # Add bias column
        bias_col = torch.ones(batch_size, patches.shape[1], 1, device=x.device, dtype=x.dtype)
        patches = torch.cat([bias_col, patches], dim=2)

        # Contrast normalization (per patch)
        # Skip first column (bias)
        data = patches[:, :, 1:]
        mean = data.mean(dim=2, keepdim=True)
        std = data.std(dim=2, keepdim=True) + 1e-10
        data = (data - mean) / std
        patches = torch.cat([bias_col, data], dim=2)

        # Forward through layers
        out = patches
        for i, layer in enumerate(self.layers):
            out = layer(out)

            # Apply activation (matches CEN activation_function mapping)
            if self.activations[i] == 0:  # Sigmoid
                out = torch.sigmoid(out)
            elif self.activations[i] == 1:  # Tanh
                out = torch.tanh(out)
            elif self.activations[i] == 2:  # ReLU
                out = torch.relu(out)

        # Reshape output to response map
        response_h = height - patch_size + 1
        response_w = width - patch_size + 1
        response = out.view(batch_size, 1, response_h, response_w)

        return response


def export_to_coreml(model_dir: str = "pyclnf/models", scales: list = None):
    """
    Export CEN patch experts to CoreML format.

    Args:
        model_dir: Directory containing CEN patch expert .dat files
        scales: List of scales to export (default: [0.25, 0.35, 0.5, 1.0])
    """
    if scales is None:
        scales = [0.25, 0.35, 0.5, 1.0]

    output_dir = Path(model_dir) / "coreml_cen"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("CEN PATCH EXPERTS → COREML EXPORT")
    print("="*80)
    print(f"Source: {model_dir}")
    print(f"Output: {output_dir}")
    print(f"Scales: {scales}")
    print()

    # Load CEN patch experts
    print("Loading CEN patch experts...")
    cen_experts = CENPatchExperts(model_dir)

    total_exported = 0
    total_skipped = 0

    for scale_idx, scale in enumerate(scales):
        if scale not in cen_experts.patch_scaling:
            print(f"Warning: Scale {scale} not available in CEN models")
            continue

        print(f"\n[{scale_idx+1}/{len(scales)}] Exporting scale {scale}...")

        experts = cen_experts.patch_experts[scale_idx]

        for landmark_idx in range(68):
            expert = experts[landmark_idx]

            if expert.is_empty:
                total_skipped += 1
                continue

            try:
                # Create PyTorch wrapper
                pytorch_model = CENPatchExpertPyTorch(expert)
                pytorch_model.eval()

                # Determine typical input size
                # CEN expects patches of size (patch_dim + response_dim - 1)
                # For 11x11 patch expert generating ~11x11 response, input is ~21x21
                input_size = 21

                # Create example input
                example_input = torch.randn(1, 1, input_size, input_size)

                # Trace the model
                traced_model = torch.jit.trace(pytorch_model, example_input)

                # Convert to CoreML
                mlmodel = ct.convert(
                    traced_model,
                    inputs=[ct.TensorType(name="input", shape=(1, 1, input_size, input_size))],
                    outputs=[ct.TensorType(name="response")],
                    convert_to="mlprogram",  # Use ML Program format for best performance
                    compute_units=ct.ComputeUnit.ALL,  # Use Neural Engine + GPU + CPU
                    minimum_deployment_target=ct.target.macOS13
                )

                # Add metadata
                mlmodel.user_defined_metadata['landmark_idx'] = str(landmark_idx)
                mlmodel.user_defined_metadata['scale'] = str(scale)
                mlmodel.user_defined_metadata['confidence'] = str(expert.confidence)
                mlmodel.short_description = f"CEN Patch Expert for landmark {landmark_idx} at scale {scale}"

                # Save model
                output_path = output_dir / f"cen_lm{landmark_idx:02d}_scale{scale:.2f}.mlpackage"
                mlmodel.save(str(output_path))

                total_exported += 1

                if (landmark_idx + 1) % 10 == 0:
                    print(f"  Progress: {landmark_idx+1}/68 landmarks exported")

            except Exception as e:
                print(f"  Error exporting landmark {landmark_idx}: {e}")
                total_skipped += 1
                continue

    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print(f"Exported: {total_exported} models")
    print(f"Skipped: {total_skipped} models (empty landmarks)")
    print(f"Output directory: {output_dir}")
    print()
    print("To use CoreML backend:")
    print("  from pyclnf.backends import CoreMLCENBackend")
    print("  backend = CoreMLCENBackend()")
    print("  backend.load_models('pyclnf/models')")
    print("="*80)


if __name__ == "__main__":
    export_to_coreml()
