#!/usr/bin/env python3
"""
Test fit_to_landmarks with initial_params vs from bbox.
"""

import numpy as np
from pathlib import Path
import sys
import subprocess
import tempfile
import cv2

sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))

from pyclnf.core.pdm import PDM
from pyclnf import CLNF
from pymtcnn import MTCNN


def test_fit_init():
    """Test fit_to_landmarks with and without initial_params."""

    print("="*70)
    print("FIT_TO_LANDMARKS INITIAL PARAMS TEST")
    print("="*70)

    # Load test frame
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read video frame")
        return

    # Initialize detector and CLNF
    detector = MTCNN()
    faces, _ = detector.detect(frame)
    bbox_np = faces[0]

    # Get base landmarks without eye refinement
    clnf = CLNF(
        model_dir="pyclnf/models",
        max_iterations=40,
        use_eye_refinement=False,
        debug_mode=False
    )
    base_landmarks, info = clnf.fit(frame, bbox_np, return_params=True)
    base_params = info['params']

    print(f"\nBase model LM36: ({base_landmarks[36, 0]:.4f}, {base_landmarks[36, 1]:.4f})")
    print(f"Base params: scale={base_params[0]:.4f}, tx={base_params[4]:.4f}, ty={base_params[5]:.4f}")

    # Simulate eye refinement moving LM36
    target_landmarks = base_landmarks.copy()
    # Move left eye landmarks as eye model would
    target_landmarks[36] = base_landmarks[36] + np.array([-0.5, 3.0])  # Simulated eye movement
    for i in range(37, 42):
        target_landmarks[i] = base_landmarks[i] + np.array([0, 2.0])

    print(f"\nSimulated eye-refined LM36: ({target_landmarks[36, 0]:.4f}, {target_landmarks[36, 1]:.4f})")
    print(f"Eye refinement delta: ({target_landmarks[36, 0] - base_landmarks[36, 0]:+.4f}, "
          f"{target_landmarks[36, 1] - base_landmarks[36, 1]:+.4f})")

    pdm = clnf.pdm

    # Test 1: fit_to_landmarks from bbox (old behavior)
    print("\n" + "="*70)
    print("Test 1: fit_to_landmarks from bbox")
    print("="*70)
    fitted_bbox = pdm.fit_to_landmarks(target_landmarks, initial_params=None)
    lm_bbox = pdm.params_to_landmarks_2d(fitted_bbox)
    print(f"Fitted LM36: ({lm_bbox[36, 0]:.4f}, {lm_bbox[36, 1]:.4f})")
    print(f"Delta from base: ({lm_bbox[36, 0] - base_landmarks[36, 0]:+.4f}, "
          f"{lm_bbox[36, 1] - base_landmarks[36, 1]:+.4f})")

    # Test 2: fit_to_landmarks from initial_params (new behavior)
    print("\n" + "="*70)
    print("Test 2: fit_to_landmarks from initial_params")
    print("="*70)
    fitted_init = pdm.fit_to_landmarks(target_landmarks, initial_params=base_params)
    lm_init = pdm.params_to_landmarks_2d(fitted_init)
    print(f"Fitted LM36: ({lm_init[36, 0]:.4f}, {lm_init[36, 1]:.4f})")
    print(f"Delta from base: ({lm_init[36, 0] - base_landmarks[36, 0]:+.4f}, "
          f"{lm_init[36, 1] - base_landmarks[36, 1]:+.4f})")

    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"\nBase LM36:             ({base_landmarks[36, 0]:.4f}, {base_landmarks[36, 1]:.4f})")
    print(f"Eye-refined target:    ({target_landmarks[36, 0]:.4f}, {target_landmarks[36, 1]:.4f})")
    print(f"Fitted from bbox:      ({lm_bbox[36, 0]:.4f}, {lm_bbox[36, 1]:.4f})")
    print(f"Fitted from init:      ({lm_init[36, 0]:.4f}, {lm_init[36, 1]:.4f})")

    print(f"\nParameter comparison:")
    print(f"  Base:        scale={base_params[0]:.6f}, tx={base_params[4]:.4f}, ty={base_params[5]:.4f}")
    print(f"  From bbox:   scale={fitted_bbox[0]:.6f}, tx={fitted_bbox[4]:.4f}, ty={fitted_bbox[5]:.4f}")
    print(f"  From init:   scale={fitted_init[0]:.6f}, tx={fitted_init[4]:.4f}, ty={fitted_init[5]:.4f}")


if __name__ == "__main__":
    test_fit_init()
