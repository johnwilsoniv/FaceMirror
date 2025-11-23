#!/usr/bin/env python3
"""
Check if there's a bias in the patch response computation for left eye.

Theory: If the peak of the response is systematically offset, it would
cause the mean-shift to push landmarks in a biased direction.
"""

import numpy as np
from pathlib import Path
import sys
import cv2

sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))

from pyclnf import CLNF
from pyclnf.core.cen_patch_expert import CENPatchExperts
from pymtcnn import MTCNN


def main():
    print("="*70)
    print("RESPONSE BIAS CHECK")
    print("="*70)

    # Load test frame
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read frame")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get landmarks
    detector = MTCNN()
    faces, _ = detector.detect(frame)
    bbox_np = faces[0]

    clnf = CLNF(model_dir="pyclnf/models", max_iterations=40, use_eye_refinement=False, debug_mode=False)
    landmarks, _ = clnf.fit(frame, bbox_np)

    # Load patch experts
    experts = CENPatchExperts("pyclnf/models")

    # Check response maps for left and right eye landmarks
    scale_idx = 0  # Use scale 0.25
    window_size = 11  # Window size for response

    print("\n" + "="*70)
    print("LEFT EYE RESPONSE PEAKS")
    print("="*70)

    for lm in range(36, 42):
        expert = experts.patch_experts[scale_idx][lm]
        if expert.is_empty:
            print(f"LM{lm}: Empty expert")
            continue

        # Extract patch around landmark
        x, y = int(landmarks[lm, 0]), int(landmarks[lm, 1])
        half_ws = window_size // 2

        # Make sure we're within image bounds
        x1 = max(0, x - half_ws - expert.width_support // 2)
        y1 = max(0, y - half_ws - expert.height_support // 2)
        x2 = min(gray.shape[1], x + half_ws + expert.width_support // 2 + 1)
        y2 = min(gray.shape[0], y + half_ws + expert.height_support // 2 + 1)

        patch = gray[y1:y2, x1:x2].astype(np.float32)

        if patch.shape[0] < expert.height_support or patch.shape[1] < expert.width_support:
            print(f"LM{lm}: Patch too small")
            continue

        # Get response
        response = expert.response(patch)

        # Find peak
        peak_idx = np.unravel_index(np.argmax(response), response.shape)
        center = (response.shape[0] // 2, response.shape[1] // 2)

        # Peak offset from center
        offset_y = peak_idx[0] - center[0]
        offset_x = peak_idx[1] - center[1]

        print(f"LM{lm}: Response {response.shape}, Peak at {peak_idx}, "
              f"Offset from center: ({offset_x:+d}, {offset_y:+d})")

    print("\n" + "="*70)
    print("RIGHT EYE RESPONSE PEAKS (using mirrored experts)")
    print("="*70)

    for lm in range(42, 48):
        expert = experts.patch_experts[scale_idx][lm]

        # Extract patch
        x, y = int(landmarks[lm, 0]), int(landmarks[lm, 1])
        half_ws = window_size // 2

        x1 = max(0, x - half_ws - expert.width_support // 2)
        y1 = max(0, y - half_ws - expert.height_support // 2)
        x2 = min(gray.shape[1], x + half_ws + expert.width_support // 2 + 1)
        y2 = min(gray.shape[0], y + half_ws + expert.height_support // 2 + 1)

        patch = gray[y1:y2, x1:x2].astype(np.float32)

        if patch.shape[0] < expert.height_support or patch.shape[1] < expert.width_support:
            print(f"LM{lm}: Patch too small")
            continue

        # Get response
        response = expert.response(patch)

        # Find peak
        peak_idx = np.unravel_index(np.argmax(response), response.shape)
        center = (response.shape[0] // 2, response.shape[1] // 2)

        offset_y = peak_idx[0] - center[0]
        offset_x = peak_idx[1] - center[1]

        print(f"LM{lm}: Response {response.shape}, Peak at {peak_idx}, "
              f"Offset from center: ({offset_x:+d}, {offset_y:+d})")


if __name__ == "__main__":
    main()
