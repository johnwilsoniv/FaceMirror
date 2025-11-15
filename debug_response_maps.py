"""
Debug response map construction, patch extraction, and window centering.

This script investigates the 3 key areas where bugs might exist:
1. Response map grid construction (row/column ordering)
2. Patch extraction coordinates (X/Y indexing)
3. Window centering and peak offset calculation
"""

import numpy as np
import sys
import cv2
import json
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "pyclnf"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyclnf.core.pdm import PDM
from pyclnf.core.patch_expert import CCNFPatchExpert

PROJECT_ROOT = Path(__file__).parent
PDM_DIR = PROJECT_ROOT / "pyclnf" / "models" / "exported_pdm"
MODELS_DIR = PROJECT_ROOT / "pyclnf" / "models"
TEST_IMAGE = PROJECT_ROOT / "calibration_frames" / "patient1_frame1.jpg"
PYTHON_RESULT = PROJECT_ROOT / "validation_output" / "python_baseline" / "patient1_frame1_result.json"

def test_response_map_construction():
    """Test 1: Verify response map is constructed with correct row/column ordering."""

    print("="*80)
    print("TEST 1: Response Map Grid Construction")
    print("="*80)

    # Load image and initial landmarks
    img = cv2.imread(str(TEST_IMAGE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    with open(PYTHON_RESULT, 'r') as f:
        data = json.load(f)
    bbox_xyxy = data['debug_info']['face_detection']['bbox']
    bbox = (bbox_xyxy[0], bbox_xyxy[1],
            bbox_xyxy[2] - bbox_xyxy[0],
            bbox_xyxy[3] - bbox_xyxy[1])

    # Initialize PDM and get initial landmarks
    pdm = PDM(str(PDM_DIR))
    params_init = pdm.init_params(bbox)
    landmarks_init = pdm.params_to_landmarks_2d(params_init)

    # Load patch expert for landmark 36 (left eye corner)
    landmark_idx = 36
    lm_x, lm_y = landmarks_init[landmark_idx]

    patch_dir = MODELS_DIR / "exported_ccnf_0.25" / "view_00" / f"patch_{landmark_idx}"
    patch_expert = CCNFPatchExpert(str(patch_dir))

    print(f"\nLandmark {landmark_idx} position: ({lm_x:.1f}, {lm_y:.1f})")
    print(f"Patch expert: {patch_expert.width}x{patch_expert.height}")

    # Build response map with explicit coordinate tracking
    window_size = 11
    half_window = window_size // 2
    response_map = np.zeros((window_size, window_size))

    print(f"\nBuilding {window_size}x{window_size} response map centered at ({lm_x:.1f}, {lm_y:.1f})")
    print(f"Window bounds in IMAGE coordinates:")
    print(f"  X: [{int(lm_x) - half_window}, {int(lm_x) + half_window}]")
    print(f"  Y: [{int(lm_y) - half_window}, {int(lm_y) + half_window}]")

    # Track which image coordinates map to which response map positions
    coord_map = {}

    for i in range(window_size):
        for j in range(window_size):
            # What we're CURRENTLY doing:
            # i = row index (0 to 10)
            # j = col index (0 to 10)
            # patch_x = start_x + j  (moves RIGHT as j increases)
            # patch_y = start_y + i  (moves DOWN as i increases)

            patch_x = int(lm_x) - half_window + j
            patch_y = int(lm_y) - half_window + i

            # Extract patch centered at (patch_x, patch_y)
            half_w = patch_expert.width // 2
            half_h = patch_expert.height // 2
            x1 = patch_x - half_w
            y1 = patch_y - half_h
            x2 = x1 + patch_expert.width
            y2 = y1 + patch_expert.height

            if 0 <= x1 and 0 <= y1 and x2 < gray.shape[1] and y2 < gray.shape[0]:
                patch = gray[y1:y2, x1:x2]
                response_map[i, j] = patch_expert.compute_response(patch)
                coord_map[(i, j)] = (patch_x, patch_y)
            else:
                response_map[i, j] = -1e10
                coord_map[(i, j)] = None

    # Find peak
    peak_row, peak_col = np.unravel_index(response_map.argmax(), response_map.shape)
    peak_value = response_map[peak_row, peak_col]
    peak_img_coords = coord_map[(peak_row, peak_col)]

    center = window_size // 2  # = 5 for window_size=11

    print(f"\n--- RESPONSE MAP PEAK ---")
    print(f"Peak at response_map[{peak_row}, {peak_col}] = {peak_value:.3f}")
    print(f"Peak corresponds to IMAGE coordinates: {peak_img_coords}")
    print(f"Landmark is at IMAGE coordinates: ({int(lm_x)}, {int(lm_y)})")
    print(f"Center of response_map is at [{center}, {center}]")
    print(f"Center corresponds to IMAGE coordinates: {coord_map[(center, center)]}")

    # Compute offset
    offset_row = peak_row - center
    offset_col = peak_col - center
    offset_dist = np.sqrt(offset_row**2 + offset_col**2)

    print(f"\n--- PEAK OFFSET ---")
    print(f"Peak offset in response_map: row={offset_row:+d}, col={offset_col:+d}")
    print(f"Peak offset distance: {offset_dist:.1f} pixels")

    # What does this offset mean in IMAGE coordinates?
    if peak_img_coords and coord_map[(center, center)]:
        img_offset_x = peak_img_coords[0] - coord_map[(center, center)][0]
        img_offset_y = peak_img_coords[1] - coord_map[(center, center)][1]
        print(f"Image offset: dx={img_offset_x:+d}, dy={img_offset_y:+d}")

        # CRITICAL CHECK: Does response_map offset match image offset?
        if offset_col == img_offset_x and offset_row == img_offset_y:
            print("✓ Response map offset MATCHES image offset (correct)")
        else:
            print(f"✗ MISMATCH: response_map offset ({offset_col}, {offset_row}) != image offset ({img_offset_x}, {img_offset_y})")

    # Show a few example mappings
    print(f"\n--- COORDINATE MAPPING EXAMPLES ---")
    print("response_map[row, col] -> IMAGE (x, y)")
    corners = [(0, 0), (0, 10), (10, 0), (10, 10), (5, 5)]
    for (row, col) in corners:
        img_coords = coord_map[(row, col)]
        if img_coords:
            print(f"  [{row:2d}, {col:2d}] -> ({img_coords[0]:3d}, {img_coords[1]:3d})")

    return response_map, coord_map


def test_patch_extraction_order():
    """Test 2: Verify patches are extracted in the correct order."""

    print("\n" + "="*80)
    print("TEST 2: Patch Extraction X/Y Indexing")
    print("="*80)

    # Create a synthetic test image with known pattern
    # Gradient: brighter towards bottom-right
    test_img = np.zeros((100, 100), dtype=np.uint8)
    for y in range(100):
        for x in range(100):
            test_img[y, x] = (x + y) * 255 // 200

    print("\nCreated test image: 100x100 gradient (brighter towards bottom-right)")
    print(f"  Top-left corner (0,0): {test_img[0, 0]}")
    print(f"  Top-right corner (99,0): {test_img[0, 99]}")
    print(f"  Bottom-left corner (0,99): {test_img[99, 0]}")
    print(f"  Bottom-right corner (99,99): {test_img[99, 99]}")

    # Extract patches at different positions
    window_size = 5
    half_window = 2
    center_x, center_y = 50, 50

    print(f"\nExtracting {window_size}x{window_size} window around ({center_x}, {center_y})")

    # Method 1: Current implementation (i=row, j=col)
    method1 = np.zeros((window_size, window_size))
    for i in range(window_size):
        for j in range(window_size):
            img_x = center_x - half_window + j
            img_y = center_y - half_window + i
            method1[i, j] = test_img[img_y, img_x]

    print("\nMethod 1 (Current: i=row, j=col):")
    print(method1.astype(int))

    # Method 2: Transposed (i=col, j=row) - INCORRECT but let's test
    method2 = np.zeros((window_size, window_size))
    for i in range(window_size):
        for j in range(window_size):
            img_x = center_x - half_window + i  # SWAPPED
            img_y = center_y - half_window + j  # SWAPPED
            method2[i, j] = test_img[img_y, img_x]

    print("\nMethod 2 (Swapped: i=col, j=row):")
    print(method2.astype(int))

    # Expected: center should be brighter than corners if pattern is correct
    center_idx = window_size // 2
    print(f"\n--- VERIFICATION ---")
    print(f"Method 1 center value: {method1[center_idx, center_idx]:.0f}")
    print(f"Method 2 center value: {method2[center_idx, center_idx]:.0f}")
    print(f"Actual image center value: {test_img[center_y, center_x]}")

    if np.allclose(method1[center_idx, center_idx], test_img[center_y, center_x]):
        print("✓ Method 1 (current) extracts correct center value")
    else:
        print("✗ Method 1 (current) WRONG center value")

    if np.allclose(method2[center_idx, center_idx], test_img[center_y, center_x]):
        print("✓ Method 2 (swapped) extracts correct center value")
    else:
        print("✗ Method 2 (swapped) WRONG center value")


def test_window_centering():
    """Test 3: Verify window center calculation and peak offset formula."""

    print("\n" + "="*80)
    print("TEST 3: Window Centering and Peak Offset Calculation")
    print("="*80)

    for window_size in [5, 7, 9, 11]:
        print(f"\n--- Window size: {window_size} ---")

        # Two possible center calculations
        center_int = window_size // 2
        center_float = (window_size - 1) / 2.0

        print(f"Integer center: {center_int}")
        print(f"Float center: {center_float}")

        # Create response map with peak at different positions
        response_map = np.zeros((window_size, window_size))

        # Test case 1: Peak at integer center
        response_map[center_int, center_int] = 100
        peak_row, peak_col = np.unravel_index(response_map.argmax(), response_map.shape)

        offset_with_int = (peak_col - center_int, peak_row - center_int)
        offset_with_float = (peak_col - center_float, peak_row - center_float)

        print(f"\nPeak at [{peak_row}, {peak_col}] (should be center):")
        print(f"  Offset with int center: {offset_with_int}")
        print(f"  Offset with float center: {offset_with_float}")

        if offset_with_int == (0, 0):
            print("  ✓ Integer center gives (0, 0) offset")
        if np.allclose(offset_with_float, (0, 0)):
            print("  ✓ Float center gives (0, 0) offset")

        # Test case 2: Peak at corner
        response_map = np.zeros((window_size, window_size))
        response_map[0, 0] = 100
        peak_row, peak_col = np.unravel_index(response_map.argmax(), response_map.shape)

        offset_with_int = (peak_col - center_int, peak_row - center_int)
        offset_with_float = (peak_col - center_float, peak_row - center_float)
        dist_int = np.sqrt(offset_with_int[0]**2 + offset_with_int[1]**2)
        dist_float = np.sqrt(offset_with_float[0]**2 + offset_with_float[1]**2)

        print(f"\nPeak at corner [{peak_row}, {peak_col}]:")
        print(f"  Offset with int center: {offset_with_int} dist={dist_int:.2f}")
        print(f"  Offset with float center: {offset_with_float} dist={dist_float:.2f}")


def test_mean_shift_calculation():
    """Test 4: Verify mean-shift calculation from response map."""

    print("\n" + "="*80)
    print("TEST 4: Mean-Shift Calculation from Response Map")
    print("="*80)

    window_size = 11
    response_map = np.zeros((window_size, window_size))

    # Create response map with peak at offset position
    peak_offset_row, peak_offset_col = 2, 3  # 2 down, 3 right from center
    center = window_size // 2  # = 5
    response_map[center + peak_offset_row, center + peak_offset_col] = 100

    print(f"\nResponse map: {window_size}x{window_size}")
    print(f"Peak at [{center + peak_offset_row}, {center + peak_offset_col}]")
    print(f"Expected offset: ({peak_offset_col}, {peak_offset_row})")

    # Current implementation (from optimizer.py _compute_mean_shift)
    sigma = 1.5
    a = -0.5 / (sigma * sigma)

    # Sum for normalization
    sum_g = 0.0
    ms_x = 0.0
    ms_y = 0.0

    resp_size = response_map.shape[0]
    center_float = (resp_size - 1) / 2.0

    for i in range(resp_size):
        for j in range(resp_size):
            # Offset from center
            dx = j - center_float
            dy = i - center_float

            # Gaussian weight
            r_sq = dx*dx + dy*dy
            g = np.exp(a * r_sq)

            # Response at this position
            resp = response_map[i, j]

            # Weight by Gaussian and response
            weight = g * resp

            ms_x += weight * dx
            ms_y += weight * dy
            sum_g += weight

    if sum_g > 1e-8:
        ms_x /= sum_g
        ms_y /= sum_g

    print(f"\nMean-shift calculation:")
    print(f"  Sigma: {sigma}")
    print(f"  Center (float): {center_float}")
    print(f"  Computed mean-shift: ({ms_x:.2f}, {ms_y:.2f})")
    print(f"  Expected mean-shift: ({peak_offset_col}, {peak_offset_row})")

    if np.allclose([ms_x, ms_y], [peak_offset_col, peak_offset_row], atol=0.5):
        print("  ✓ Mean-shift matches expected offset")
    else:
        print("  ✗ Mean-shift DIFFERS from expected offset")


def main():
    print("CLNF Response Map Debug Investigation")
    print("="*80)
    print("\nInvestigating 3 key areas:")
    print("1. Response map grid construction")
    print("2. Patch extraction X/Y indexing")
    print("3. Window centering and offset calculation")
    print()

    # Run all tests
    response_map, coord_map = test_response_map_construction()
    test_patch_extraction_order()
    test_window_centering()
    test_mean_shift_calculation()

    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
