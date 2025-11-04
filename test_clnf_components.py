#!/usr/bin/env python3
"""Test CLNF components individually to identify segfault source"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

print("="*80)
print("CLNF Component Test")
print("="*80)

# Test 1: PDM
print("\n[Test 1] PDM (Point Distribution Model)")
print("-" * 80)

try:
    from pyfaceau.clnf.pdm import PointDistributionModel

    model_dir = Path("S1 Face Mirror/weights/clnf")
    pdm_path = model_dir / "In-the-wild_aligned_PDM_68.txt"

    pdm = PointDistributionModel(pdm_path)
    print(f"✓ PDM loaded: {pdm.n_landmarks} landmarks, {pdm.n_modes} modes")

    # Test params to landmarks
    test_params = np.zeros(pdm.n_modes, dtype=np.float32)
    test_scale = 1.0
    test_translation = np.array([320.0, 240.0], dtype=np.float32)

    landmarks_2d = pdm.params_to_landmarks_2d(test_params, test_scale, test_translation)
    print(f"✓ params_to_landmarks_2d: {landmarks_2d.shape}")

    # Test landmarks to params
    params, scale, translation = pdm.landmarks_to_params_2d(landmarks_2d)
    print(f"✓ landmarks_to_params_2d: params={params.shape}, scale={scale:.2f}, trans={translation}")

    pdm_ok = True
except Exception as e:
    print(f"✗ PDM test failed: {e}")
    import traceback
    traceback.print_exc()
    pdm_ok = False

if not pdm_ok:
    sys.exit(1)

# Test 2: CEN Patch Experts (just loading, no inference)
print("\n[Test 2] CEN Patch Experts (Loading Only)")
print("-" * 80)

try:
    from pyfaceau.clnf.cen_patch_experts import CENPatchExperts

    patch_experts = CENPatchExperts(model_dir)
    print(f"✓ Patch experts loaded for {len(patch_experts.patch_experts)} scales")

    cen_ok = True
except Exception as e:
    print(f"✗ CEN loading failed: {e}")
    import traceback
    traceback.print_exc()
    cen_ok = False

if not cen_ok:
    sys.exit(1)

# Test 3: NU-RLMS Optimizer (initialization only)
print("\n[Test 3] NU-RLMS Optimizer (Initialization Only)")
print("-" * 80)

try:
    from pyfaceau.clnf.nu_rlms import NURLMSOptimizer

    optimizer = NURLMSOptimizer(
        pdm, patch_experts,
        max_iterations=5,
        convergence_threshold=0.01
    )
    print("✓ Optimizer initialized")

    optimizer_ok = True
except Exception as e:
    print(f"✗ Optimizer init failed: {e}")
    import traceback
    traceback.print_exc()
    optimizer_ok = False

if not optimizer_ok:
    sys.exit(1)

# Test 4: CEN Response Computation (the likely culprit)
print("\n[Test 4] CEN Response Computation")
print("-" * 80)

try:
    import cv2

    # Create test image
    test_image = np.ones((480, 640), dtype=np.float32) * 128.0

    # Create test landmarks in center
    test_landmarks = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        test_landmarks[i, 0] = 320.0 + (i % 17) * 10
        test_landmarks[i, 1] = 240.0 + (i // 17) * 20

    print("  Created test image and landmarks")
    print(f"  Image: {test_image.shape}, dtype={test_image.dtype}")
    print(f"  Landmarks: {test_landmarks.shape}, dtype={test_landmarks.dtype}")

    # Try to compute response (this is where segfault likely occurs)
    print("  Computing CEN responses (this may crash)...")

    scale_idx = 2  # 0.50 scale
    responses, extraction_bounds = patch_experts.response(test_image, test_landmarks, scale_idx)

    print(f"✓ CEN response computed successfully")
    print(f"  Responses: {len(responses)} patches")

    response_ok = True
except Exception as e:
    print(f"✗ CEN response failed: {e}")
    import traceback
    traceback.print_exc()
    response_ok = False

# Final Summary
print("\n" + "="*80)
print("COMPONENT TEST SUMMARY")
print("="*80)

results = {
    "PDM": "✓" if pdm_ok else "✗",
    "CEN Loading": "✓" if cen_ok else "✗",
    "Optimizer Init": "✓" if optimizer_ok else "✗",
    "CEN Response": "✓" if response_ok else "✗"
}

for component, status in results.items():
    print(f"  {status} {component}")

if all(status == "✓" for status in results.values()):
    print("\n✓ All components working!")
else:
    print("\n✗ Some components failed")

print()
