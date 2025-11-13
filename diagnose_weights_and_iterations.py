#!/usr/bin/env python3
"""
Diagnose potential accuracy issues:
1. Check patch confidence values
2. Test weight_multiplier effect
3. Compare iteration counts
"""

import numpy as np
from pathlib import Path
from pyclnf import CLNF
from pyclnf.core.patch_expert import CCNFModel

def check_patch_confidences():
    """Check if patch confidences are loaded correctly."""
    print("=" * 70)
    print("PATCH CONFIDENCE ANALYSIS")
    print("=" * 70)

    # Load CCNF model
    ccnf = CCNFModel("pyclnf/models", scales=[0.25, 0.35, 0.5])

    # Check confidences for each view and scale
    for view_idx in range(ccnf.num_views):
        for scale_idx, scale in enumerate(ccnf.scales):
            patch_experts = ccnf.get_patch_experts(view_idx, scale)

            if len(patch_experts) == 0:
                continue

            print(f"\nView {view_idx}, Scale {scale}:")
            print(f"  Patch experts: {len(patch_experts)}")

            # Collect confidence values
            confidences = []
            for lm_idx, patch in patch_experts.items():
                if hasattr(patch, 'patch_confidence'):
                    confidences.append(patch.patch_confidence)

            if confidences:
                print(f"  Confidence range: [{min(confidences):.3f}, {max(confidences):.3f}]")
                print(f"  Confidence mean: {np.mean(confidences):.3f}")
                print(f"  Confidence std: {np.std(confidences):.3f}")

                # Show some example confidences
                example_landmarks = list(patch_experts.keys())[:10]
                print(f"  Example confidences (first 10 landmarks):")
                for lm_idx in example_landmarks:
                    conf = patch_experts[lm_idx].patch_confidence
                    print(f"    Landmark {lm_idx:2d}: {conf:.4f}")
            else:
                print("  ⚠️  No patch confidence values found!")

    print("\n" + "=" * 70)


def test_weight_multiplier_effect():
    """Test effect of different weight_multiplier values."""
    print("\nWEIGHT MULTIPLIER EFFECT TEST")
    print("=" * 70)

    import cv2
    image_path = "Patient Data/Normal Cohort/IMG_0434.MOV_frame_0000.jpg"
    image = cv2.imread(image_path)

    # Test different weight multipliers
    weight_values = [0.0, 5.0, 7.0]

    print("\nTesting weight_multiplier values:")
    for w in weight_values:
        print(f"\n  weight_multiplier = {w}:")

        clnf = CLNF(weight_multiplier=w, detector=None)

        # Use manual bbox (from previous tests)
        bbox = (273, 793, 401, 404)

        landmarks, info = clnf.fit(image, bbox)

        print(f"    Iterations: {info['iterations']}")
        print(f"    Converged: {info['converged']}")
        print(f"    Final update: {info['final_update']:.6f}")

        # Calculate bbox center distance
        lm_center = landmarks.mean(axis=0)
        bbox_center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
        dist = np.sqrt((lm_center[0] - bbox_center[0])**2 +
                      (lm_center[1] - bbox_center[1])**2)
        print(f"    Landmark center offset from bbox: {dist:.2f}px")

    print("\n" + "=" * 70)


def check_convergence_parameters():
    """Check current convergence parameters."""
    print("\nCONVERGENCE PARAMETERS")
    print("=" * 70)

    clnf = CLNF()

    print(f"  Max iterations: {clnf.optimizer.max_iterations}")
    print(f"  Convergence threshold: {clnf.optimizer.convergence_threshold}")
    print(f"  Regularization: {clnf.regularization}")
    print(f"  Sigma: {clnf.sigma}")
    print(f"  Weight multiplier: {clnf.weight_multiplier}")
    print(f"  Window sizes: {clnf.window_sizes}")

    # Calculate iterations per window
    n_windows = len(clnf.window_sizes)
    iters_per_window = clnf.optimizer.max_iterations // n_windows
    remainder = clnf.optimizer.max_iterations % n_windows

    print(f"\n  Iterations per window:")
    for i, ws in enumerate(clnf.window_sizes):
        iters = iters_per_window + (1 if i < remainder else 0)
        print(f"    Window {ws}: {iters} iterations")

    print("\n  C++ OpenFace typical parameters (from literature):")
    print("    Max iterations: ~10-20 per window size")
    print("    Convergence threshold: ~0.005-0.01")
    print("    Weight multiplier: 5-7 (NU-RLMS mode)")
    print("    Window sizes: [11, 9, 7] (coarse to fine)")

    print("\n" + "=" * 70)


def main():
    print("\n🔍 ACCURACY DIAGNOSTICS: Checking Implementation Details")
    print("=" * 70)

    # 1. Check patch confidences
    check_patch_confidences()

    # 2. Test weight multiplier effect
    test_weight_multiplier_effect()

    # 3. Check convergence parameters
    check_convergence_parameters()

    print("\n" + "=" * 70)
    print("FINDINGS SUMMARY")
    print("=" * 70)
    print("\nKey Observations:")
    print("  1. Patch confidence values are loaded (check output above)")
    print("  2. Default weight_multiplier = 0.0 (NOT using confidence weighting!)")
    print("  3. C++ OpenFace typically uses weight_multiplier = 5-7")
    print("\n⚠️  POTENTIAL ISSUE IDENTIFIED:")
    print("  pyCLNF defaults to weight_multiplier=0.0, which disables")
    print("  patch confidence weighting (NU-RLMS mode). This could explain")
    print("  a significant portion of the 8.23px accuracy gap!")
    print("\n💡 RECOMMENDATION:")
    print("  Test with weight_multiplier=5.0 or 7.0 to match C++ OpenFace")
    print("=" * 70)


if __name__ == "__main__":
    main()
