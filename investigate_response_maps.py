"""
Investigate response map computation to find why peaks are offset from center.

This is the most likely cause of non-convergence since initialization is working well.
"""

import numpy as np
import cv2
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "pyclnf"))
from core.pdm import PDM
from core.patch_expert import CCNFModel

PROJECT_ROOT = Path(__file__).parent
PDM_DIR = PROJECT_ROOT / "pyclnf" / "models" / "exported_pdm"
MODELS_DIR = PROJECT_ROOT / "pyclnf" / "models"
OUTPUT_DIR = PROJECT_ROOT / "validation_output" / "response_map_investigation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test case
TEST_FRAME = "patient1_frame1"
TEST_IMAGE = PROJECT_ROOT / "calibration_frames" / f"{TEST_FRAME}.jpg"
PYTHON_RESULT = PROJECT_ROOT / "validation_output" / "python_baseline" / f"{TEST_FRAME}_result.json"


def load_test_data():
    """Load test image and Python pipeline results."""
    # Load image
    img = cv2.imread(str(TEST_IMAGE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load Python results
    with open(PYTHON_RESULT, 'r') as f:
        data = json.load(f)

    # Get bbox
    bbox_xyxy = data['debug_info']['face_detection']['bbox']
    bbox = (bbox_xyxy[0], bbox_xyxy[1],
            bbox_xyxy[2] - bbox_xyxy[0],
            bbox_xyxy[3] - bbox_xyxy[1])

    # Get final landmarks (for comparison)
    landmarks_final = np.array(data['debug_info']['landmark_detection']['landmarks_68'])

    return gray, bbox, landmarks_final


def test_patch_expert_response(gray, bbox, landmark_idx=36, window_size=11):
    """
    Test patch expert response map for a single landmark.

    Args:
        gray: Grayscale image
        bbox: Face bounding box (x, y, w, h)
        landmark_idx: Which landmark to test (default: 36 = left eye corner)
        window_size: Patch window size (11, 9, or 7)
    """
    print(f"\n{'='*80}")
    print(f"Testing Patch Expert Response Map")
    print(f"Landmark {landmark_idx}, Window Size {window_size}x{window_size}")
    print(f"{'='*80}\n")

    # Load PDM
    pdm = PDM(str(PDM_DIR))

    # Initialize from bbox
    params_init = pdm.init_params(bbox)
    landmarks_init = pdm.params_to_landmarks_2d(params_init)

    print(f"Initialized landmark {landmark_idx} at: ({landmarks_init[landmark_idx, 0]:.1f}, {landmarks_init[landmark_idx, 1]:.1f})")

    # Load patch expert
    ccnf = CCNFModel(str(MODELS_DIR), scales=[0.25])
    view_idx = 0  # Frontal view

    scale_model = ccnf.scale_models.get(0.25)
    view_data = scale_model['views'].get(view_idx)

    if landmark_idx not in view_data['patches']:
        print(f"  ✗ No patch expert for landmark {landmark_idx} at view {view_idx}")
        return None

    patch_expert = view_data['patches'][landmark_idx]

    print(f"Patch expert loaded:")
    print(f"  Patch size: {patch_expert.width}x{patch_expert.height}")
    print(f"  Num neurons: {patch_expert.num_neurons}")
    print(f"  Betas shape: {patch_expert.betas.shape}")
    print(f"  Patch confidence: {patch_expert.patch_confidence:.6f}")

    # Extract current landmark position
    lm_x, lm_y = landmarks_init[landmark_idx]

    # Compute response map manually
    response_map = np.zeros((window_size, window_size))
    half_window = window_size // 2
    patch_w, patch_h = patch_expert.width, patch_expert.height
    half_patch_w, half_patch_h = patch_w // 2, patch_h // 2

    for i in range(window_size):
        for j in range(window_size):
            # Position in window relative to center
            offset_x = j - half_window
            offset_y = i - half_window

            # Patch center position in image
            patch_center_x = int(lm_x + offset_x)
            patch_center_y = int(lm_y + offset_y)

            # Extract patch bounds
            x1 = patch_center_x - half_patch_w
            y1 = patch_center_y - half_patch_h
            x2 = x1 + patch_w
            y2 = y1 + patch_h

            # Check bounds
            if x1 < 0 or y1 < 0 or x2 >= gray.shape[1] or y2 >= gray.shape[0]:
                response_map[i, j] = 0.0
                continue

            # Extract patch
            patch = gray[y1:y2, x1:x2]

            # Compute response
            response_map[i, j] = patch_expert.compute_response(patch)

    print(f"\nResponse map statistics:")
    print(f"  Shape: {response_map.shape}")
    print(f"  Min: {response_map.min():.6f}")
    print(f"  Max: {response_map.max():.6f}")
    print(f"  Mean: {response_map.mean():.6f}")
    print(f"  Std: {response_map.std():.6f}")

    # Find peak
    peak_idx = np.unravel_index(response_map.argmax(), response_map.shape)
    peak_y, peak_x = peak_idx
    center_y, center_x = response_map.shape[0] // 2, response_map.shape[1] // 2

    offset_x = peak_x - center_x
    offset_y = peak_y - center_y
    offset_dist = np.sqrt(offset_x**2 + offset_y**2)

    print(f"\nPeak location:")
    print(f"  Peak at: ({peak_x}, {peak_y})")
    print(f"  Center at: ({center_x}, {center_y})")
    print(f"  Offset: ({offset_x:+d}, {offset_y:+d}) = {offset_dist:.1f}px")
    print(f"  Peak value: {response_map[peak_y, peak_x]:.6f}")

    # Check if peak is at edge (problematic)
    edge_threshold = 2
    at_edge = (peak_x < edge_threshold or peak_x >= response_map.shape[1] - edge_threshold or
               peak_y < edge_threshold or peak_y >= response_map.shape[0] - edge_threshold)

    if at_edge:
        print(f"  ⚠️  WARNING: Peak is within {edge_threshold}px of edge!")

    if offset_dist > 3.0:
        print(f"  ⚠️  WARNING: Peak offset >3px from center!")

    # Visualize response map
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Response map heatmap
    im1 = axes[0].imshow(response_map, cmap='hot', interpolation='nearest')
    axes[0].plot(center_x, center_y, 'g+', markersize=15, markeredgewidth=2, label='Center')
    axes[0].plot(peak_x, peak_y, 'bx', markersize=15, markeredgewidth=2, label='Peak')
    axes[0].set_title(f'Response Map (Landmark {landmark_idx}, ws={window_size})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[0])

    # Response map cross-sections
    axes[1].plot(response_map[center_y, :], 'g-', label='Horizontal (through center)', linewidth=2)
    axes[1].plot(response_map[:, center_x], 'b-', label='Vertical (through center)', linewidth=2)
    axes[1].axvline(center_x, color='gray', linestyle='--', alpha=0.5, label='Center')
    axes[1].axvline(peak_x, color='red', linestyle='--', alpha=0.5, label='Peak X')
    axes[1].set_title('Response Map Cross-Sections')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Response Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"response_map_lm{landmark_idx}_ws{window_size}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_path}")
    plt.close()

    return {
        'landmark_idx': landmark_idx,
        'window_size': window_size,
        'response_map': response_map,
        'peak_location': (peak_x, peak_y),
        'center_location': (center_x, center_y),
        'offset': (offset_x, offset_y),
        'offset_distance': offset_dist,
        'peak_value': response_map[peak_y, peak_x],
        'at_edge': at_edge,
    }


def test_multiple_landmarks(gray, bbox):
    """Test response maps for multiple landmarks to find patterns."""
    print(f"\n{'='*80}")
    print(f"Testing Multiple Landmarks")
    print(f"{'='*80}\n")

    # Test key landmarks (eyes, nose, mouth corners)
    test_landmarks = [36, 39, 42, 45, 30, 48, 54]  # Eye corners, nose, mouth corners
    window_sizes = [11, 9, 7]

    results = []
    for lm_idx in test_landmarks:
        for ws in window_sizes:
            result = test_patch_expert_response(gray, bbox, lm_idx, ws)
            if result is not None:
                results.append(result)

    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")

    offsets = [r['offset_distance'] for r in results]
    at_edge_count = sum(r['at_edge'] for r in results)
    large_offset_count = sum(r['offset_distance'] > 3.0 for r in results)

    print(f"Tested {len(results)} landmark/window combinations")
    print(f"Mean peak offset: {np.mean(offsets):.2f}px")
    print(f"Max peak offset: {np.max(offsets):.2f}px")
    print(f"Peaks at edge: {at_edge_count}/{len(results)} ({100*at_edge_count/len(results):.1f}%)")
    print(f"Large offsets (>3px): {large_offset_count}/{len(results)} ({100*large_offset_count/len(results):.1f}%)")

    if large_offset_count > len(results) * 0.3:
        print(f"\n⚠️  WARNING: {100*large_offset_count/len(results):.0f}% of peaks have large offsets!")
        print(f"This will prevent convergence (updates stay >3px, threshold is 0.01px)")
        print(f"\nPossible causes:")
        print(f"  1. Patch expert weights incorrect")
        print(f"  2. Image preprocessing mismatch with training")
        print(f"  3. Response computation bug")
        print(f"  4. Incorrect patch extraction")


def main():
    print("="*80)
    print("Response Map Investigation")
    print("="*80)

    # Load test data
    gray, bbox, landmarks_final = load_test_data()
    print(f"\nLoaded test image: {TEST_IMAGE.name}")
    print(f"Image size: {gray.shape}")
    print(f"Bbox: ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})")

    # Test single landmark in detail
    print(f"\n{'='*80}")
    print(f"Detailed Test: Left Eye Corner (Landmark 36)")
    print(f"{'='*80}")
    test_patch_expert_response(gray, bbox, landmark_idx=36, window_size=11)

    # Test multiple landmarks
    test_multiple_landmarks(gray, bbox)

    print(f"\n{'='*80}")
    print(f"Investigation Complete")
    print(f"{'='*80}")
    print(f"\nVisualizations saved to: {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Compare these response maps with C++ OpenFace output")
    print(f"  2. If peaks are offset in both → initialization issue")
    print(f"  3. If peaks are centered in C++ but offset in Python → patch expert bug")
    print(f"  4. Check patch extraction code for coordinate system bugs")


if __name__ == "__main__":
    main()
