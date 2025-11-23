#!/usr/bin/env python3
"""
Export OpenFace eye models to Python format.

This script exports the hierarchical eye models (PDM and CCNF patches)
from OpenFace C++ format to Python-compatible NumPy format.
"""

import numpy as np
from pathlib import Path
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from openface_loader import PDMLoader, CCNFPatchExpertLoader


def export_eye_pdm(openface_model_dir: str, output_dir: str):
    """
    Export eye PDM models from OpenFace format.

    Args:
        openface_model_dir: Path to OpenFace model_eye directory
        output_dir: Output directory for exported models
    """
    model_eye_dir = Path(openface_model_dir)
    output_path = Path(output_dir)

    # Eye PDM files
    # Note: Both eyes use the same PDM structure but may have different files
    pdm_files = {
        'left': model_eye_dir / 'pdms' / 'pdm_28_l_eye_3D_closed.txt',
        'right': model_eye_dir / 'pdms' / 'pdm_28_eye_3D_closed.txt'  # Right eye uses different file
    }

    for eye_side, pdm_file in pdm_files.items():
        if not pdm_file.exists():
            print(f"Warning: {pdm_file} not found, skipping {eye_side} eye PDM")
            continue

        print(f"Loading {eye_side} eye PDM from {pdm_file}...")
        loader = PDMLoader(str(pdm_file))

        # Create output directory
        eye_output_dir = output_path / f'exported_eye_pdm_{eye_side}'
        eye_output_dir.mkdir(parents=True, exist_ok=True)

        # Save components
        np.save(eye_output_dir / 'mean_shape.npy', loader.mean_shape)
        np.save(eye_output_dir / 'eigenvectors.npy', loader.princ_comp)
        np.save(eye_output_dir / 'eigenvalues.npy', loader.eigen_values)

        print(f"  Exported {eye_side} eye PDM:")
        print(f"    mean_shape: {loader.mean_shape.shape}")
        print(f"    eigenvectors: {loader.princ_comp.shape}")
        print(f"    eigenvalues: {loader.eigen_values.shape}")
        print(f"    Saved to: {eye_output_dir}")


def export_eye_ccnf(openface_model_dir: str, output_dir: str):
    """
    Export eye CCNF patch expert models from OpenFace format.

    Args:
        openface_model_dir: Path to OpenFace model_eye directory
        output_dir: Output directory for exported models
    """
    model_eye_dir = Path(openface_model_dir)
    output_path = Path(output_dir)

    # Eye CCNF patch expert files
    # Scales: 1.00 and 1.50
    ccnf_files = {
        'left': {
            1.00: model_eye_dir / 'patch_experts' / 'left_ccnf_patches_1.00_synth_lid_.txt',
            1.50: model_eye_dir / 'patch_experts' / 'left_ccnf_patches_1.50_synth_lid_.txt',
        },
        'right': {
            1.00: model_eye_dir / 'patch_experts' / 'ccnf_patches_1.00_synth_lid_.txt',
            1.50: model_eye_dir / 'patch_experts' / 'ccnf_patches_1.50_synth_lid_.txt',
        }
    }

    for eye_side, scale_files in ccnf_files.items():
        eye_output_dir = output_path / f'exported_eye_ccnf_{eye_side}'
        eye_output_dir.mkdir(parents=True, exist_ok=True)

        for scale, ccnf_file in scale_files.items():
            if not ccnf_file.exists():
                print(f"Warning: {ccnf_file} not found, skipping {eye_side} eye scale {scale}")
                continue

            print(f"Loading {eye_side} eye CCNF scale {scale} from {ccnf_file}...")

            try:
                # Eye models have 28 landmarks, not 68
                loader = CCNFPatchExpertLoader(str(ccnf_file), num_landmarks=28)

                # Use built-in save method
                scale_dir = eye_output_dir / f'scale_{scale:.2f}'
                loader.save_numpy(str(scale_dir))

                print(f"  Exported {eye_side} eye CCNF scale {scale} to {scale_dir}")

            except Exception as e:
                print(f"  Error loading {ccnf_file}: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Main export function."""
    # OpenFace model directory
    openface_model_dir = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model/model_eye"

    # Output directory
    output_dir = "/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models"

    print("=" * 60)
    print("Exporting Eye Models from OpenFace")
    print("=" * 60)
    print(f"Source: {openface_model_dir}")
    print(f"Output: {output_dir}")
    print()

    # Export PDM
    print("\n--- Exporting Eye PDM ---")
    export_eye_pdm(openface_model_dir, output_dir)

    # Export CCNF
    print("\n--- Exporting Eye CCNF ---")
    export_eye_ccnf(openface_model_dir, output_dir)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
