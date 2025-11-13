#!/usr/bin/env python3
"""
Quick diagnostic: Profile where time is being spent in CLNF fitting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

import cv2
import numpy as np
import time
from pyclnf import CLNF


def profile_single_iteration():
    """Profile a single iteration with detailed timing."""

    # Load test image
    image = cv2.imread("Patient Data/Normal Cohort/IMG_0434.MOV_frame_0000.jpg")
    if image is None:
        # Extract a frame first
        import subprocess
        subprocess.run([
            "ffmpeg", "-i", "Patient Data/Normal Cohort/IMG_0434.MOV",
            "-vframes", "1", "-y", "test_output/profile_frame.jpg"
        ], capture_output=True)
        image = cv2.imread("test_output/profile_frame.jpg")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Simple bbox
    h, w = gray.shape
    bbox = (w * 0.3, h * 0.2, w * 0.4, h * 0.5)

    print("🔍 PROFILING CLNF PERFORMANCE")
    print("=" * 70)

    # Test 1: Initialize CLNF
    print("\n[1/5] Testing CLNF initialization...")
    t0 = time.time()
    clnf = CLNF(max_iterations=1, detector=None)  # Just 1 iteration for profiling
    t1 = time.time()
    print(f"      ✓ Initialization: {t1-t0:.2f}s")

    # Test 2: Initialize shape
    print("\n[2/5] Testing shape initialization...")
    t0 = time.time()
    clnf.pdm.init_from_bbox(bbox, gray.shape)
    initial_params = clnf.pdm.get_params()
    t1 = time.time()
    print(f"      ✓ Shape init: {t1-t0:.2f}s")

    # Test 3: Compute response maps (THE LIKELY BOTTLENECK)
    print("\n[3/5] Testing response map computation (1 window size)...")
    from pyclnf.core.patch_expert import LNFPatchExpert

    window_size = 11
    t0 = time.time()

    # Get current landmarks
    current_shape = clnf.pdm.get_shape(initial_params)

    # Compute responses for ALL 68 landmarks
    response_maps = []
    for landmark_idx in range(68):
        lm_x, lm_y = current_shape[landmark_idx]

        # Extract patch around landmark
        half_ws = window_size // 2
        x_min = int(lm_x) - half_ws
        y_min = int(lm_y) - half_ws
        x_max = x_min + window_size
        y_max = y_min + window_size

        # Boundary check
        if x_min < 0 or y_min < 0 or x_max >= w or y_max >= h:
            response_maps.append(np.zeros((window_size, window_size)))
            continue

        patch = gray[y_min:y_max, x_min:x_max].astype(np.float32) / 255.0

        # Call patch expert
        response = clnf.patch_expert.compute_response_map(
            patch, window_size, landmark_idx
        )
        response_maps.append(response)

    t1 = time.time()
    print(f"      ✓ Response maps (68 landmarks, ws=11): {t1-t0:.2f}s")
    print(f"        → {(t1-t0)/68:.3f}s per landmark")
    print(f"        → For 20 iterations × 68 landmarks × 3 window sizes:")
    print(f"          Estimated: {(t1-t0)/68 * 20 * 68 * 3 / 60:.1f} MINUTES")

    # Test 4: Mean shift calculation
    print("\n[4/5] Testing mean shift calculation...")
    t0 = time.time()
    mean_shift = np.zeros(68 * 2)
    for i, response_map in enumerate(response_maps):
        if response_map.size == 0:
            continue
        # Find peak
        max_idx = np.argmax(response_map)
        max_y, max_x = np.unravel_index(max_idx, response_map.shape)
        center = window_size // 2
        mean_shift[i*2] = max_x - center
        mean_shift[i*2 + 1] = max_y - center
    t1 = time.time()
    print(f"      ✓ Mean shift (68 landmarks): {t1-t0:.2f}s")

    # Test 5: Single full iteration
    print("\n[5/5] Testing complete single iteration...")
    clnf_test = CLNF(max_iterations=1, detector=None)
    t0 = time.time()
    landmarks, info = clnf_test.fit(gray, bbox, return_params=True)
    t1 = time.time()
    print(f"      ✓ Full iteration (1 iter, 3 window sizes): {t1-t0:.2f}s")

    print("\n" + "=" * 70)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"\nIf response map computation is the bottleneck:")
    print(f"  Single landmark: {(t1-t0)/68:.3f}s")
    print(f"  Full 20-iteration fit: {(t1-t0)/68 * 20 * 68 * 3 / 60:.1f} minutes")
    print(f"\n⚠️  This explains the 5+ minute delays you're seeing!")


if __name__ == "__main__":
    profile_single_iteration()
