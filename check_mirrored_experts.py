#!/usr/bin/env python3
"""
Check which landmarks use mirrored patch experts.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))

from pyclnf.core.cen_patch_expert import CENPatchExperts

def main():
    print("="*70)
    print("MIRRORED EXPERT CHECK")
    print("="*70)

    # Load CEN experts
    model_dir = Path("pyclnf/models")
    experts = CENPatchExperts(str(model_dir))

    # Get mirror indices
    mirror_inds = experts.mirror_inds
    print(f"\nMirror indices shape: {mirror_inds.shape}")

    # Check left eye landmarks (36-41)
    print("\n" + "="*70)
    print("LEFT EYE (36-41) MIRROR MAPPING")
    print("="*70)
    print(f"\n{'LM':>4} {'Mirror':>8} {'Has Expert':>12} {'Uses Mirror':>12}")
    print("-"*40)

    for lm in range(36, 42):
        mirror = mirror_inds[lm]
        # Check if this landmark has an expert at scale 0
        expert = experts.patch_experts[0][lm]
        has_expert = not expert.is_empty
        uses_mirror = not has_expert
        print(f"{lm:4d} {mirror:8d} {'Yes' if has_expert else 'No':>12} {'Yes' if uses_mirror else 'No':>12}")

    # Check right eye landmarks (42-47)
    print("\n" + "="*70)
    print("RIGHT EYE (42-47) MIRROR MAPPING")
    print("="*70)
    print(f"\n{'LM':>4} {'Mirror':>8} {'Has Expert':>12} {'Uses Mirror':>12}")
    print("-"*40)

    for lm in range(42, 48):
        mirror = mirror_inds[lm]
        expert = experts.patch_experts[0][lm]
        has_expert = not expert.is_empty
        uses_mirror = not has_expert
        print(f"{lm:4d} {mirror:8d} {'Yes' if has_expert else 'No':>12} {'Yes' if uses_mirror else 'No':>12}")

    # Count total empty/mirrored
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    empty_count = 0
    mirrored_lms = []

    for lm in range(68):
        expert = experts.patch_experts[0][lm]
        if expert.is_empty:
            empty_count += 1
            mirrored_lms.append(lm)

    print(f"\nTotal empty landmarks: {empty_count}")
    print(f"Landmarks using mirrored experts: {mirrored_lms}")

    # Check if left side or right side is mirrored
    left_side = [lm for lm in mirrored_lms if lm < 34 or (36 <= lm <= 41) or (48 <= lm <= 54)]
    right_side = [lm for lm in mirrored_lms if 34 <= lm < 36 or (42 <= lm <= 47) or (55 <= lm <= 67)]

    print(f"\nMirrored landmarks on LEFT side: {len([lm for lm in mirrored_lms if mirror_inds[lm] > lm])}")
    print(f"Mirrored landmarks on RIGHT side: {len([lm for lm in mirrored_lms if mirror_inds[lm] < lm])}")


if __name__ == "__main__":
    main()
