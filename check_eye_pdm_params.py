#!/usr/bin/env python3
"""
Check eye PDM parameter structure.
"""

import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))

from pyclnf.core.eye_pdm import EyePDM

# Load eye PDM
model_dir = Path("pyclnf/models/exported_eye_pdm")
eye_pdm = EyePDM(str(model_dir))

print("Eye PDM Structure:")
print("=" * 80)

print(f"Number of landmarks: {eye_pdm.n_points}")
print(f"Number of modes: {eye_pdm.n_modes}")

# Initialize some params
params = eye_pdm.get_initial_params()
print(f"\nParameter count: {len(params)}")
print(f"  Expected: 10 (6 global + 4 local)")
print(f"  Actual: {len(params)}")

if len(params) > 6:
    print(f"\nParameter breakdown:")
    print(f"  Global (0-5): scale, rot_x, rot_y, rot_z, trans_x, trans_y")
    print(f"  Local (6-{len(params)-1}): {len(params) - 6} shape parameters")

# Generate landmarks to test
landmarks = eye_pdm.params_to_landmarks_2d(params)
print(f"\nGenerated landmarks shape: {landmarks.shape}")

# Check if params match expected structure
if len(params) == 10:
    print("\n✓ Eye PDM has correct 10-parameter structure")
else:
    print(f"\n⚠️ Eye PDM has {len(params)} parameters instead of expected 10")
    print("  This may affect convergence analysis")