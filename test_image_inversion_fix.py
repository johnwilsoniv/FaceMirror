#!/usr/bin/env python3
"""
Test if inverting image intensities fixes the response map edge-peak issue.

Hypothesis: Patch experts may have been trained on inverted intensities,
causing the response maps to show edge peaks instead of center peaks.
"""

import cv2
import numpy as np
import sys

sys.path.insert(0, 'pyclnf')
from pyclnf import CLNF

# Test configuration
VIDEO_PATH = 'Patient Data/Normal Cohort/IMG_0433.MOV'
FRAME_NUM = 50
FACE_BBOX = (241, 555, 532, 532)


def extract_frame(video_path, frame_num):
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read frame {frame_num} from {video_path}")

    return frame


def main():
    print("="*80)
    print("TESTING IMAGE INVERSION AS FIX FOR EDGE-PEAK RESPONSE MAPS")
    print("="*80)
    print()

    # Load frame
    print(f"Loading frame {FRAME_NUM} from {VIDEO_PATH}...")
    frame = extract_frame(VIDEO_PATH, FRAME_NUM)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"Frame shape: {gray.shape}")
    print()

    # Test with NORMAL image
    print("="*80)
    print("TEST 1: NORMAL IMAGE (current broken behavior)")
    print("="*80)
    print()

    clnf_normal = CLNF(model_dir='pyclnf/models', max_iterations=20)
    landmarks_normal, info_normal = clnf_normal.fit(gray, FACE_BBOX, return_params=True)

    print(f"Converged: {info_normal['converged']}")
    print(f"Iterations: {info_normal['iterations']}")
    print(f"Final update: {info_normal['final_update']:.6f}")
    print(f"Target: 0.005")
    print(f"Ratio: {info_normal['final_update'] / 0.005:.1f}x the target")
    print()

    # Test with INVERTED image
    print("="*80)
    print("TEST 2: INVERTED IMAGE (255 - gray)")
    print("="*80)
    print()

    gray_inverted = 255 - gray
    print(f"Inverted image: min={gray_inverted.min()}, max={gray_inverted.max()}")
    print()

    clnf_inverted = CLNF(model_dir='pyclnf/models', max_iterations=20)
    landmarks_inverted, info_inverted = clnf_inverted.fit(gray_inverted, FACE_BBOX, return_params=True)

    print(f"Converged: {info_inverted['converged']}")
    print(f"Iterations: {info_inverted['iterations']}")
    print(f"Final update: {info_inverted['final_update']:.6f}")
    print(f"Target: 0.005")
    print(f"Ratio: {info_inverted['final_update'] / 0.005:.1f}x the target")
    print()

    # Compare results
    print("="*80)
    print("COMPARISON")
    print("="*80)
    print()

    print(f"Normal image:   final_update = {info_normal['final_update']:.6f}")
    print(f"Inverted image: final_update = {info_inverted['final_update']:.6f}")
    print()

    improvement = (info_normal['final_update'] - info_inverted['final_update']) / info_normal['final_update'] * 100

    if info_inverted['final_update'] < info_normal['final_update'] * 0.5:
        print(f"✓ INVERSION FIXES THE ISSUE! Improvement: {improvement:.1f}%")
        print()
        print("This confirms that patch experts expect INVERTED image intensities.")
        print("Dark pixels should be bright, bright pixels should be dark.")
        print()
        print("FIX REQUIRED in pyclnf/core/patch_expert.py:")
        print("  Change: patch_float = image_patch.astype(np.float32) / 255.0")
        print("  To:     patch_float = (255 - image_patch).astype(np.float32) / 255.0")
    elif info_inverted['final_update'] < info_normal['final_update']:
        print(f"~ Inversion helps slightly: {improvement:.1f}% improvement")
        print("May be part of the solution but not the complete fix.")
    else:
        print(f"✗ Inversion makes it WORSE: {abs(improvement):.1f}% degradation")
        print("The issue is NOT related to image inversion.")

    print()
    print("="*80)
    print("LANDMARK POSITION COMPARISON")
    print("="*80)
    print()

    # Compare landmark positions
    diff = landmarks_inverted - landmarks_normal
    diff_mag = np.linalg.norm(diff, axis=1)

    print(f"Mean landmark position difference: {diff_mag.mean():.2f} px")
    print(f"Max landmark position difference:  {diff_mag.max():.2f} px")
    print()

    if diff_mag.mean() < 5.0:
        print("Landmarks are SIMILAR between normal and inverted images.")
        print("→ Both initializations may be finding similar local minima.")
    else:
        print("Landmarks are DIFFERENT between normal and inverted images.")
        print("→ Different response map structure leads to different solutions.")

    print("="*80)


if __name__ == "__main__":
    main()
