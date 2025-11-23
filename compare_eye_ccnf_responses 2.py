#!/usr/bin/env python3
"""
Compare Eye CCNF patch expert responses between C++ and Python.

This extracts the same image patch and computes the response in Python,
then compares with C++ response to identify differences.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from pyclnf.core.eye_patch_expert import EyeCCNFModel

# Paths
VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
MODEL_DIR = "pyclnf/models"

def extract_frame(frame_idx=0):
    """Extract a frame from the video."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_idx}")
        return None

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def main():
    print("=" * 60)
    print("Eye CCNF Patch Expert Response Comparison")
    print("=" * 60)

    # Load frame
    gray = extract_frame(0)
    if gray is None:
        return

    print(f"Image shape: {gray.shape}")

    # Load Eye CCNF model
    print("\nLoading Eye CCNF model...")
    eye_ccnf = EyeCCNFModel(MODEL_DIR, 'left')

    print(f"Eye CCNF info: {eye_ccnf.get_info()}")

    # Get patch experts for scale 1.0
    patch_experts = eye_ccnf.get_all_patch_experts(1.0)
    print(f"\nLoaded {len(patch_experts)} patch experts for scale 1.0")

    # Sample landmark position (approximate eye location)
    # These are approximate positions for the left eye in the test video
    test_positions = {
        8: (398, 825),   # Outer corner (maps to main landmark 36)
        10: (417, 806),  # Upper outer (maps to main landmark 37)
        12: (447, 804),  # Upper inner (maps to main landmark 38)
        14: (470, 822),  # Inner corner (maps to main landmark 39)
        16: (446, 829),  # Lower inner (maps to main landmark 40)
        18: (418, 831),  # Lower outer (maps to main landmark 41)
    }

    print("\n" + "=" * 60)
    print("Patch Expert Response Analysis")
    print("=" * 60)

    for lm_idx, (x, y) in test_positions.items():
        if lm_idx not in patch_experts:
            print(f"\nLandmark {lm_idx}: No patch expert")
            continue

        patch_expert = patch_experts[lm_idx]

        print(f"\nLandmark {lm_idx} at ({x}, {y}):")
        print(f"  Patch size: {patch_expert.width}x{patch_expert.height}")
        print(f"  Num neurons: {patch_expert.num_neurons}")
        print(f"  Patch confidence: {patch_expert.patch_confidence:.4f}")

        # Extract patch at this position
        half_w = patch_expert.width // 2
        half_h = patch_expert.height // 2

        y1 = max(0, y - half_h)
        y2 = min(gray.shape[0], y + half_h + 1)
        x1 = max(0, x - half_w)
        x2 = min(gray.shape[1], x + half_w + 1)

        patch = gray[y1:y2, x1:x2]

        # Pad if needed
        if patch.shape[0] < patch_expert.height or patch.shape[1] < patch_expert.width:
            padded = np.zeros((patch_expert.height, patch_expert.width), dtype=np.uint8)
            padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded

        print(f"  Patch extracted: {patch.shape}")
        print(f"  Patch intensity - min: {patch.min()}, max: {patch.max()}, mean: {patch.mean():.1f}")

        # Compute response
        response = patch_expert.compute_response(patch)
        print(f"  Response: {response:.6f}")

        # Also compute a small response map (3x3 window)
        ws = 3
        half_ws = ws // 2
        response_map = np.zeros((ws, ws))

        for dy in range(-half_ws, half_ws + 1):
            for dx in range(-half_ws, half_ws + 1):
                px = x + dx
                py = y + dy

                y1 = max(0, py - half_h)
                y2 = min(gray.shape[0], py + half_h + 1)
                x1 = max(0, px - half_w)
                x2 = min(gray.shape[1], px + half_w + 1)

                test_patch = gray[y1:y2, x1:x2]
                if test_patch.shape[0] < patch_expert.height or test_patch.shape[1] < patch_expert.width:
                    padded = np.zeros((patch_expert.height, patch_expert.width), dtype=np.uint8)
                    padded[:test_patch.shape[0], :test_patch.shape[1]] = test_patch
                    test_patch = padded

                response_map[dy + half_ws, dx + half_ws] = patch_expert.compute_response(test_patch)

        print(f"  Response map (3x3):")
        for row in range(ws):
            row_str = "    "
            for col in range(ws):
                row_str += f"{response_map[row, col]:7.4f} "
            print(row_str)

        # Compute KDE mean-shift from response map
        sigma = 1.0
        a_kde = -0.5 / (sigma * sigma)
        center = (ws - 1) / 2.0

        total_weight = 0.0
        mx = 0.0
        my = 0.0

        for ii in range(ws):
            for jj in range(ws):
                dist_sq = (ii - center)**2 + (jj - center)**2
                kde_weight = np.exp(a_kde * dist_sq)
                weight = response_map[ii, jj] * kde_weight

                total_weight += weight
                mx += weight * jj
                my += weight * ii

        if total_weight > 1e-10:
            ms_x = (mx / total_weight) - center
            ms_y = (my / total_weight) - center
        else:
            ms_x = ms_y = 0.0

        mag = np.sqrt(ms_x**2 + ms_y**2)
        print(f"  Mean-shift: ({ms_x:.4f}, {ms_y:.4f}) mag={mag:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nKey questions to investigate:")
    print("1. Are the response values similar to C++ (need C++ debug output)?")
    print("2. Are the neurons computing the same values?")
    print("3. Is the normalized cross-correlation correct?")
    print("\nTo compare with C++, add equivalent debugging to LandmarkDetectorModel.cpp")
    print("in the NonVectorisedMeanShift_precalc_kde function for the 28-point eye model.")

if __name__ == "__main__":
    main()
