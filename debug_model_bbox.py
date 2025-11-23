#!/usr/bin/env python3
"""
Debug model bounding box computation for fit_to_landmarks.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from pyclnf.core.pdm import PDM

def test_model_bbox():
    """Test how model bbox is computed."""

    pdm = PDM("pyclnf/models/exported_pdm")
    print(f"PDM: {pdm.n_points} landmarks, {pdm.n_modes} modes")

    # Neutral params: scale=1, rotation=0, translation=0, shape=0
    neutral_params = np.zeros(pdm.n_params, dtype=np.float32)
    neutral_params[0] = 1.0

    # Get model landmarks at neutral pose
    neutral_lm = pdm.params_to_landmarks_2d(neutral_params)

    print(f"\nNeutral pose landmarks:")
    print(f"  X range: [{neutral_lm[:, 0].min():.4f}, {neutral_lm[:, 0].max():.4f}]")
    print(f"  Y range: [{neutral_lm[:, 1].min():.4f}, {neutral_lm[:, 1].max():.4f}]")

    model_width = neutral_lm[:, 0].max() - neutral_lm[:, 0].min()
    model_height = neutral_lm[:, 1].max() - neutral_lm[:, 1].min()

    print(f"  Width: {model_width:.4f}")
    print(f"  Height: {model_height:.4f}")

    # Check model center
    model_center_x = (neutral_lm[:, 0].min() + neutral_lm[:, 0].max()) / 2.0
    model_center_y = (neutral_lm[:, 1].min() + neutral_lm[:, 1].max()) / 2.0
    print(f"  Center: ({model_center_x:.4f}, {model_center_y:.4f})")

    # Sample landmark 36 at neutral
    print(f"\n  Landmark 36 at neutral: ({neutral_lm[36, 0]:.4f}, {neutral_lm[36, 1]:.4f})")

    # Test with a typical face bbox
    print("\n" + "="*60)
    print("Test scaling computation")
    print("="*60)

    # Typical face bbox dimensions
    input_width = 150.0
    input_height = 200.0

    scaling = ((input_width / model_width) + (input_height / model_height)) / 2.0
    print(f"\nFor input bbox {input_width}x{input_height}:")
    print(f"  width ratio:  {input_width / model_width:.4f}")
    print(f"  height ratio: {input_height / model_height:.4f}")
    print(f"  Average scaling: {scaling:.4f}")


if __name__ == "__main__":
    test_model_bbox()
