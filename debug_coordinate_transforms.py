#!/usr/bin/env python3
"""
Coordinate Transformation Diagnostic

This script traces the exact coordinate transformations in CLNF to identify
the source of the systematic ±5px peak offset error.

Key Question: Where does the ±half_window offset come from?

Transformation chain:
1. Image coords (lm_x, lm_y) → Response map window extraction
2. Response map window → Peak detection
3. Peak location → Mean-shift vector
4. Mean-shift → Parameter update
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, 'pyclnf')
from pyclnf import CLNF
from pyclnf.core.optimizer import NURLMSOptimizer

# Test configuration
VIDEO_PATH = 'Patient Data/Normal Cohort/IMG_0433.MOV'
FRAME_NUM = 50
FACE_BBOX = (241, 555, 532, 532)


class DiagnosticOptimizer(NURLMSOptimizer):
    """
    Optimizer instrumented to trace coordinate transformations in detail.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_data = []

    def _compute_mean_shift(self, landmarks_2d, patch_experts, image, pdm,
                           window_size=11, sim_img_to_ref=None,
                           sim_ref_to_img=None, sigma_components=None):
        """Trace coordinate transformations in detail."""

        print("\n" + "="*80)
        print(f"ITERATION - Window Size = {window_size}")
        print("="*80)

        n_points = landmarks_2d.shape[0]
        half_window = window_size // 2

        print(f"\nConfiguration:")
        print(f"  window_size = {window_size}")
        print(f"  half_window = {half_window}")
        print(f"  response map size = {window_size} × {window_size}")
        print(f"  response map center (float) = {(window_size - 1) / 2.0}")

        # Select a few representative landmarks to trace in detail
        trace_landmarks = list(patch_experts.keys())[:5]  # First 5

        for landmark_idx in trace_landmarks:
            if landmark_idx >= n_points:
                continue

            patch_expert = patch_experts[landmark_idx]
            lm_x, lm_y = landmarks_2d[landmark_idx]

            print(f"\n{'─'*80}")
            print(f"LANDMARK {landmark_idx}")
            print(f"{'─'*80}")

            # STEP 1: Image coordinates
            print(f"\n[STEP 1] Image Coordinates:")
            print(f"  lm_x = {lm_x:.4f}")
            print(f"  lm_y = {lm_y:.4f}")
            print(f"  int(lm_x) = {int(lm_x)}")
            print(f"  int(lm_y) = {int(lm_y)}")
            print(f"  fractional part: ({lm_x - int(lm_x):.4f}, {lm_y - int(lm_y):.4f})")

            # STEP 2: Response map window bounds
            print(f"\n[STEP 2] Response Map Window Extraction:")
            start_x = int(lm_x) - half_window
            start_y = int(lm_y) - half_window
            end_x = start_x + window_size
            end_y = start_y + window_size

            print(f"  start_x = int(lm_x) - half_window = {int(lm_x)} - {half_window} = {start_x}")
            print(f"  start_y = int(lm_y) - half_window = {int(lm_y)} - {half_window} = {start_y}")
            print(f"  end_x = {end_x}")
            print(f"  end_y = {end_y}")
            print(f"  Window spans: x ∈ [{start_x}, {end_x}), y ∈ [{start_y}, {end_y})")

            # STEP 3: Current position within response map
            resp_size = window_size
            center = (resp_size - 1) / 2.0  # Float center
            dx_frac = lm_x - int(lm_x)
            dy_frac = lm_y - int(lm_y)
            dx = dx_frac + center
            dy = dy_frac + center

            print(f"\n[STEP 3] Current Position in Response Map:")
            print(f"  center = (window_size - 1) / 2.0 = ({window_size} - 1) / 2.0 = {center}")
            print(f"  dx_frac = lm_x - int(lm_x) = {dx_frac:.4f}")
            print(f"  dy_frac = lm_y - int(lm_y) = {dy_frac:.4f}")
            print(f"  dx = dx_frac + center = {dx_frac:.4f} + {center} = {dx:.4f}")
            print(f"  dy = dy_frac + center = {dy_frac:.4f} + {center} = {dy:.4f}")

            # STEP 4: Compute actual response map
            response_map = self._compute_response_map(
                image, lm_x, lm_y, patch_expert, window_size,
                sim_img_to_ref, sim_ref_to_img, sigma_components
            )

            if response_map is None:
                print(f"\n  ⚠️  Response map is None - skipping")
                continue

            # STEP 5: Find peak in response map
            peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
            peak_y, peak_x = peak_idx  # row, col = y, x
            peak_value = response_map[peak_y, peak_x]

            print(f"\n[STEP 4] Peak Detection in Response Map:")
            print(f"  response_map.shape = {response_map.shape}")
            print(f"  peak location (row, col) = ({peak_y}, {peak_x})")
            print(f"  peak value = {peak_value:.6f}")
            print(f"  max value = {np.max(response_map):.6f}")
            print(f"  mean value = {np.mean(response_map):.6f}")

            # STEP 6: Compute offset from center to peak
            offset_x = peak_x - center
            offset_y = peak_y - center
            offset_mag = np.sqrt(offset_x**2 + offset_y**2)

            print(f"\n[STEP 5] Peak Offset from Center:")
            print(f"  offset_x = peak_x - center = {peak_x} - {center} = {offset_x:.4f}")
            print(f"  offset_y = peak_y - center = {peak_y} - {center} = {offset_y:.4f}")
            print(f"  offset magnitude = {offset_mag:.4f} px")

            # STEP 7: Compute KDE mean-shift
            a = -0.5 / (self.sigma ** 2)
            ms_x, ms_y = self._kde_mean_shift(response_map, dx, dy, a)

            print(f"\n[STEP 6] KDE Mean-Shift Computation:")
            print(f"  sigma = {self.sigma}")
            print(f"  a = -0.5 / sigma² = {a:.6f}")
            print(f"  ms_x = {ms_x:.4f}")
            print(f"  ms_y = {ms_y:.4f}")
            print(f"  mean-shift magnitude = {np.sqrt(ms_x**2 + ms_y**2):.4f} px")

            # STEP 8: Convert mean-shift to image coordinates
            # Mean-shift is computed in response map coordinates
            # It should be directly added to current landmark position
            new_lm_x = lm_x + ms_x
            new_lm_y = lm_y + ms_y

            print(f"\n[STEP 7] Convert to Image Coordinates:")
            print(f"  new_lm_x = lm_x + ms_x = {lm_x:.4f} + {ms_x:.4f} = {new_lm_x:.4f}")
            print(f"  new_lm_y = lm_y + ms_y = {lm_y:.4f} + {ms_y:.4f} = {new_lm_y:.4f}")
            print(f"  movement = ({ms_x:.4f}, {ms_y:.4f}) = {np.sqrt(ms_x**2 + ms_y**2):.4f} px")

            # ANALYSIS: Where should the peak point to?
            print(f"\n[ANALYSIS] Expected Behavior:")
            print(f"  If response map is centered correctly:")
            print(f"    - Peak at center (5, 5) means 'stay here' → offset = (0, 0)")
            print(f"    - Peak at (6, 5) means 'move right 1px' → offset = (+1, 0)")
            print(f"    - Peak at (4, 5) means 'move left 1px' → offset = (-1, 0)")
            print(f"  Current peak offset: ({offset_x:.1f}, {offset_y:.1f}) px")
            print(f"  SUSPICIOUS: offset magnitude ≈ {half_window} px!")

            # Reverse-engineer what image position the peak corresponds to
            peak_image_x_expected = start_x + peak_x
            peak_image_y_expected = start_y + peak_y

            print(f"\n[REVERSE ENGINEERING] Peak Image Position:")
            print(f"  If response_map[{peak_y}, {peak_x}] is at image position:")
            print(f"    peak_img_x = start_x + peak_x = {start_x} + {peak_x} = {peak_image_x_expected}")
            print(f"    peak_img_y = start_y + peak_y = {start_y} + {peak_y} = {peak_image_y_expected}")
            print(f"  vs current landmark at ({lm_x:.2f}, {lm_y:.2f})")
            print(f"  Direct pixel offset: ({peak_image_x_expected - lm_x:.2f}, {peak_image_y_expected - lm_y:.2f})")

        # Call original implementation
        return super()._compute_mean_shift(
            landmarks_2d, patch_experts, image, pdm, window_size,
            sim_img_to_ref, sim_ref_to_img, sigma_components
        )


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
    print("COORDINATE TRANSFORMATION DIAGNOSTIC")
    print("="*80)
    print()
    print("This script traces the exact coordinate transformations to identify")
    print("the source of the systematic ±5px peak offset error.")
    print()

    # Load frame
    print(f"Loading frame {FRAME_NUM} from {VIDEO_PATH}...")
    frame = extract_frame(VIDEO_PATH, FRAME_NUM)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"Frame shape: {frame.shape}")
    print()

    # Create instrumented CLNF
    print("Initializing diagnostic CLNF...")
    clnf = CLNF(model_dir='pyclnf/models', max_iterations=1, window_sizes=[11])

    # Replace optimizer with diagnostic version
    clnf.optimizer = DiagnosticOptimizer(
        regularization=clnf.optimizer.regularization,
        max_iterations=1,  # Just 1 iteration for detailed trace
        convergence_threshold=clnf.optimizer.convergence_threshold,
        sigma=clnf.optimizer.sigma,
        weight_multiplier=clnf.optimizer.weight_multiplier
    )

    print("\nRunning CLNF with detailed coordinate tracing...")
    print("(Only 1 iteration to keep output manageable)")
    print()

    landmarks, info = clnf.fit(gray, FACE_BBOX, return_params=True)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print("Key observations to look for:")
    print()
    print("1. INDEXING BUG:")
    print("   - Is the window extraction correct?")
    print("   - start_x = int(lm_x) - half_window → Is this the right formula?")
    print("   - Should it be: start_x = int(lm_x - half_window)?")
    print()
    print("2. PEAK OFFSET MEANING:")
    print("   - If peak is at response_map[peak_y, peak_x]...")
    print("   - ...what image position does that correspond to?")
    print("   - Is (offset_x, offset_y) being computed correctly?")
    print()
    print("3. SYSTEMATIC ERROR:")
    print("   - Do all offsets cluster around ±5px (= ±half_window)?")
    print("   - This suggests an off-by-one or wrong coordinate system")
    print()


if __name__ == "__main__":
    main()
