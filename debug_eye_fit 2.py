#!/usr/bin/env python3
"""
Debug the actual eye refinement + fit_to_landmarks scenario.

The issue: After eye refinement, Python's fit_to_landmarks moves landmark 36
in the wrong X direction compared to C++ CalcParams.

From the debug doc:
- C++ Eye_8 moved: (-1.019, 3.862)
- Python Eye_8 moved: (-0.707, 3.604)
- But after fit_to_landmarks, C++ landmark 36 ends up at (391.5, 830.2)
  while Python ends up at (390.4, 831.2)
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from pyclnf.core.pdm import PDM
from pyclnf.clnf import CLNF

def test_eye_refinement_fit():
    """Test fit_to_landmarks with eye refinement scenario."""

    print("="*70)
    print("DEBUG: Eye Refinement + fit_to_landmarks")
    print("="*70)

    # Load PDM
    pdm = PDM("pyclnf/models/exported_pdm")
    print(f"PDM: {pdm.n_points} landmarks, {pdm.n_modes} modes")

    # Create landmarks that simulate the situation BEFORE eye refinement
    # Using values from the debug progress doc

    # Initial main model params (before eye refinement)
    # These produce landmarks that then get refined by eye model
    init_params = np.zeros(pdm.n_params, dtype=np.float32)
    init_params[0] = 3.372791  # scale
    init_params[1] = -0.117327  # pitch
    init_params[2] = 0.175093   # yaw
    init_params[3] = -0.099452  # roll
    init_params[4] = 425.039772 # tx
    init_params[5] = 820.113798 # ty

    # Get initial landmarks from main model
    initial_landmarks = pdm.params_to_landmarks_2d(init_params)

    print(f"\nInitial main model landmark 36: ({initial_landmarks[36, 0]:.4f}, {initial_landmarks[36, 1]:.4f})")

    # Simulate eye refinement output
    # From debug doc: Eye_8 (which maps to landmark 36) moved by (-1.019, 3.862) in C++
    # Let's apply similar movement to simulate what eye model does

    refined_landmarks = initial_landmarks.copy()

    # Eye landmarks are 36-41 (left eye) and 42-47 (right eye)
    # Apply movements similar to what eye model produces
    # The eye model moves landmarks based on CCNF responses

    # From python_eye_model_detailed.txt, after all eye refinement:
    # Eye_8 (landmark 36): (389.9703, 832.0247)
    # This is the position AFTER eye model refinement

    # Let's set landmark 36 to the refined position
    refined_landmarks[36] = [389.9703, 832.0247]

    # Also set other eye landmarks from the debug output
    # Eye_10 -> landmark 37, Eye_12 -> landmark 38, etc.
    refined_landmarks[37] = [405.3045, 813.9448]
    refined_landmarks[38] = [436.9733, 809.1592]
    refined_landmarks[39] = [458.7169, 822.9713]
    refined_landmarks[40] = [438.4949, 833.6870]
    refined_landmarks[41] = [410.4356, 838.8382]

    print(f"Refined landmark 36 (after eye model): ({refined_landmarks[36, 0]:.4f}, {refined_landmarks[36, 1]:.4f})")

    # Now call fit_to_landmarks (like C++ CalcParams)
    # This projects the eye refinement through the main model constraints

    print("\n" + "="*70)
    print("Calling fit_to_landmarks (C++ CalcParams equivalent)")
    print("="*70)

    # Pass the current rotation as initial guess
    rotation_init = init_params[1:4]

    fitted_params = pdm.fit_to_landmarks(refined_landmarks, rotation=rotation_init)
    fitted_landmarks = pdm.params_to_landmarks_2d(fitted_params)

    print(f"\nFitted landmark 36: ({fitted_landmarks[36, 0]:.4f}, {fitted_landmarks[36, 1]:.4f})")

    # Compute movement
    initial_pos = initial_landmarks[36]
    refined_pos = refined_landmarks[36]
    fitted_pos = fitted_landmarks[36]

    eye_movement = refined_pos - initial_pos
    fit_movement = fitted_pos - initial_pos

    print(f"\nMovement analysis for landmark 36:")
    print(f"  Initial position:  ({initial_pos[0]:.4f}, {initial_pos[1]:.4f})")
    print(f"  After eye model:   ({refined_pos[0]:.4f}, {refined_pos[1]:.4f})")
    print(f"  After fit:         ({fitted_pos[0]:.4f}, {fitted_pos[1]:.4f})")
    print(f"  Eye model movement: ({eye_movement[0]:+.4f}, {eye_movement[1]:+.4f})")
    print(f"  Final movement:     ({fit_movement[0]:+.4f}, {fit_movement[1]:+.4f})")

    # C++ reference values
    # C++ landmark 36 after CalcParams: (391.5, 830.2)
    # So C++ final movement from initial should be computed

    print(f"\nC++ reference:")
    print(f"  C++ final landmark 36: (391.5, 830.2)")
    cpp_movement = np.array([391.5, 830.2]) - initial_pos
    print(f"  C++ final movement: ({cpp_movement[0]:+.4f}, {cpp_movement[1]:+.4f})")

    print(f"\nComparison:")
    print(f"  Python X movement: {fit_movement[0]:+.4f}")
    print(f"  C++ X movement:    {cpp_movement[0]:+.4f}")

    if np.sign(fit_movement[0]) != np.sign(cpp_movement[0]):
        print(f"  ✗ X DIRECTION INVERTED!")
    else:
        print(f"  ✓ X direction matches")

    # Check parameter differences
    print("\n" + "="*70)
    print("Parameter comparison")
    print("="*70)

    print(f"\nInitial params:")
    print(f"  scale={init_params[0]:.6f}, rot=({init_params[1]:.6f}, {init_params[2]:.6f}, {init_params[3]:.6f})")
    print(f"  tx={init_params[4]:.6f}, ty={init_params[5]:.6f}")

    print(f"\nFitted params:")
    print(f"  scale={fitted_params[0]:.6f}, rot=({fitted_params[1]:.6f}, {fitted_params[2]:.6f}, {fitted_params[3]:.6f})")
    print(f"  tx={fitted_params[4]:.6f}, ty={fitted_params[5]:.6f}")

    param_change = fitted_params[:6] - init_params[:6]
    print(f"\nParameter changes:")
    print(f"  Δscale={param_change[0]:.6f}")
    print(f"  Δrot=({param_change[1]:.6f}, {param_change[2]:.6f}, {param_change[3]:.6f})")
    print(f"  Δtx={param_change[4]:.6f}, Δty={param_change[5]:.6f}")


if __name__ == "__main__":
    test_eye_refinement_fit()
