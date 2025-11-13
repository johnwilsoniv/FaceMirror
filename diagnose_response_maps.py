"""
Response Map Diagnostic Tool

Analyzes response map quality to identify why PyCLNF convergence is poor.
Compares response properties with/without sigma transformation.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add pyclnf to path
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))

from pyclnf.core.patch_expert import CCNFModel
from pyclnf.core.pdm import PDM
from pyclnf.core.optimizer import NURLMSOptimizer
from pyclnf.models.openface_loader import load_sigma_components


def analyze_response_map(response_map: np.ndarray, name: str):
    """Analyze properties of a response map."""

    min_val = response_map.min()
    max_val = response_map.max()
    mean_val = response_map.mean()
    std_val = response_map.std()

    # Find peak
    peak_idx = np.unravel_index(response_map.argmax(), response_map.shape)
    peak_val = response_map[peak_idx]

    # Measure peak sharpness: ratio of peak to mean of surrounding 8 neighbors
    i, j = peak_idx
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < response_map.shape[0] and 0 <= nj < response_map.shape[1]:
                neighbors.append(response_map[ni, nj])

    neighbor_mean = np.mean(neighbors) if neighbors else 0
    sharpness = peak_val / neighbor_mean if neighbor_mean > 1e-10 else float('inf')

    # Dynamic range (max / mean of non-peak values)
    non_peak_vals = response_map.copy()
    non_peak_vals[peak_idx] = 0
    non_peak_mean = np.mean(non_peak_vals[non_peak_vals > 0]) if np.any(non_peak_vals > 0) else 0
    dynamic_range = max_val / non_peak_mean if non_peak_mean > 1e-10 else float('inf')

    print(f"\n{name}:")
    print(f"  Shape: {response_map.shape}")
    print(f"  Min: {min_val:.6f}")
    print(f"  Max: {max_val:.6f}")
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Std: {std_val:.6f}")
    print(f"  Peak position: {peak_idx}")
    print(f"  Peak sharpness (peak/neighbor_mean): {sharpness:.3f}")
    print(f"  Dynamic range (peak/non_peak_mean): {dynamic_range:.3f}")

    # Count how many pixels are near peak (within 90% of max)
    near_peak = np.sum(response_map >= 0.9 * max_val)
    print(f"  Pixels near peak (>90% max): {near_peak} ({100*near_peak/response_map.size:.1f}%)")

    return {
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
        'std': std_val,
        'peak_pos': peak_idx,
        'sharpness': sharpness,
        'dynamic_range': dynamic_range,
        'near_peak_count': near_peak
    }


def extract_patch(image: np.ndarray, center_x: float, center_y: float,
                  width: int, height: int) -> np.ndarray:
    """Extract a patch from the image centered at (center_x, center_y)."""

    half_width = width // 2
    half_height = height // 2

    x_start = int(center_x - half_width)
    y_start = int(center_y - half_height)

    # Check bounds
    if (x_start < 0 or y_start < 0 or
        x_start + width > image.shape[1] or
        y_start + height > image.shape[0]):
        return None

    patch = image[y_start:y_start+height, x_start:x_start+width]

    return patch


def compute_response_map_manual(image: np.ndarray, center_x: float, center_y: float,
                                 patch_expert, window_size: int,
                                 apply_sigma: bool = False,
                                 sigma_components: dict = None) -> np.ndarray:
    """
    Manually compute response map to diagnose issues.
    Mimics optimizer._compute_response_map but with detailed logging.
    """

    response_map = np.zeros((window_size, window_size))
    half_window = window_size // 2

    start_x = int(center_x) - half_window
    start_y = int(center_y) - half_window

    # Sample a few positions to see individual neuron responses
    print(f"\n  Computing responses at {window_size}x{window_size} positions...")
    sample_positions = [(window_size//2, window_size//2)]  # Center position

    # Compute all responses
    neuron_responses_all = []
    patch_hashes = []  # Track if patches are actually different
    for i in range(window_size):
        for j in range(window_size):
            patch_x = start_x + j
            patch_y = start_y + i

            patch = extract_patch(image, patch_x, patch_y,
                                 patch_expert.width, patch_expert.height)

            if patch is not None:
                # Track patch hash to verify patches are different
                patch_hash = hash(patch.tobytes())
                patch_hashes.append((i, j, patch_hash))

                response = patch_expert.compute_response(patch)
                response_map[i, j] = response

                # Log neuron details for sample positions
                if (i, j) in sample_positions:
                    neuron_responses = []
                    features = patch_expert._extract_features(patch)
                    for neuron in patch_expert.neurons:
                        if abs(neuron['alpha']) >= 1e-4:
                            nr = patch_expert._compute_neuron_response(features, neuron)
                            neuron_responses.append(nr)

                    neuron_responses_all.append({
                        'pos': (i, j),
                        'total': response,
                        'neuron_responses': neuron_responses,
                        'patch': patch,
                        'features': features
                    })
            else:
                response_map[i, j] = -1e10

    # Check if patches are unique
    unique_hashes = len(set([h for _, _, h in patch_hashes]))
    print(f"  Unique patches: {unique_hashes} out of {len(patch_hashes)} total")
    if unique_hashes == 1:
        print(f"    WARNING: ALL patches are IDENTICAL!")
    elif unique_hashes < len(patch_hashes) * 0.5:
        print(f"    WARNING: Many duplicate patches detected!")

    # Log neuron-level details
    print(f"\n  Neuron-level analysis at sample positions:")
    for nr_data in neuron_responses_all:
        pos = nr_data['pos']
        total = nr_data['total']
        neuron_resps = nr_data['neuron_responses']
        print(f"    Position {pos}: total={total:.6f}")
        if neuron_resps:
            print(f"      Neuron responses: min={min(neuron_resps):.6f}, max={max(neuron_resps):.6f}, mean={np.mean(neuron_resps):.6f}")
            print(f"      Num neurons active: {len(neuron_resps)}")

    # Log RAW response statistics before any transformation
    print(f"\n  RAW response statistics (before sigma/normalization):")
    print(f"    Min: {response_map.min():.10f}")
    print(f"    Max: {response_map.max():.10f}")
    print(f"    Mean: {response_map.mean():.10f}")
    print(f"    Std: {response_map.std():.10f}")
    print(f"    Range: {response_map.max() - response_map.min():.10f}")

    # Apply sigma transformation if requested
    if apply_sigma and sigma_components is not None and window_size in sigma_components:
        print(f"\n  Applying Sigma transformation...")
        sigma_comps = sigma_components[window_size]
        Sigma = patch_expert.compute_sigma(sigma_comps, window_size=window_size)

        print(f"    Sigma matrix shape: {Sigma.shape}")
        print(f"    Sigma matrix stats: min={Sigma.min():.6f}, max={Sigma.max():.6f}, mean={Sigma.mean():.6f}")

        # Check if Sigma is diagonal-dominant or has off-diagonal structure
        diag_mean = np.mean(np.abs(np.diag(Sigma)))
        off_diag = Sigma.copy()
        np.fill_diagonal(off_diag, 0)
        off_diag_mean = np.mean(np.abs(off_diag))
        print(f"    Diagonal mean: {diag_mean:.6f}, Off-diagonal mean: {off_diag_mean:.6f}")
        print(f"    Diagonal dominance: {diag_mean / (off_diag_mean + 1e-10):.3f}")

        response_shape = response_map.shape
        response_vec = response_map.reshape(-1, 1)
        response_transformed = Sigma @ response_vec
        response_map = response_transformed.reshape(response_shape)

        print(f"\n  After Sigma transformation:")
        print(f"    Min: {response_map.min():.10f}")
        print(f"    Max: {response_map.max():.10f}")
        print(f"    Mean: {response_map.mean():.10f}")
        print(f"    Std: {response_map.std():.10f}")

    # OpenFace normalization (lines 406-413 in CCNF_patch_expert.cpp)
    min_val = response_map.min()
    if min_val < 0:
        response_map = response_map - min_val
        print(f"\n  After min-shift: min={response_map.min():.10f}, max={response_map.max():.10f}")

    max_val = response_map.max()
    if max_val > 0:
        response_map = response_map / max_val
        print(f"  After normalization: min={response_map.min():.10f}, max={response_map.max():.10f}")

    return response_map


def main():
    print("=" * 70)
    print("Response Map Diagnostic Tool")
    print("=" * 70)

    # Load test image
    test_image_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/test_output/debug_landmarks_direct.jpg"

    if not Path(test_image_path).exists():
        test_image_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/openface_bbox_test_results/IMG_0433_frame_50_comparison.jpg"

    if not Path(test_image_path).exists():
        print(f"\nError: Test image not found")
        print("Using a synthetic test image with variation...")
        # Create synthetic image with gradients
        image = np.zeros((480, 640), dtype=np.uint8)
        for i in range(480):
            for j in range(640):
                image[i, j] = (i + j) % 256
    else:
        image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        print(f"Loaded test image from: {test_image_path}")

    print(f"\nTest image shape: {image.shape}")

    # Load CCNF model
    model_dir = "/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models"
    print(f"\nLoading CCNF model from: {model_dir}")
    ccnf = CCNFModel(model_dir)

    # Load sigma components
    sigma_components = load_sigma_components(model_dir)
    if sigma_components:
        print(f"Loaded sigma components for window sizes: {list(sigma_components.keys())}")
    else:
        print("Warning: No sigma components loaded")

    # Test parameters
    scale = 0.25
    view_idx = 0
    window_size = 11

    # Test several landmarks
    test_landmarks = [30, 36, 48]  # Nose tip, left eye corner, mouth corner

    for landmark_idx in test_landmarks:
        print("\n" + "=" * 70)
        print(f"Testing Landmark {landmark_idx}")
        print("=" * 70)

        # Get patch expert
        patch_expert = ccnf.get_patch_expert(scale, view_idx, landmark_idx)
        if patch_expert is None:
            print(f"  Patch expert not found for landmark {landmark_idx}")
            continue

        patch_info = patch_expert.get_info()
        print(f"\nPatch expert info:")
        print(f"  Size: {patch_info['width']}x{patch_info['height']}")
        print(f"  Neurons: {patch_info['num_neurons']}")
        print(f"  Confidence: {patch_info['patch_confidence']:.3f}")

        # Test at image center (rough estimate of face location)
        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2

        print(f"\nComputing response maps at position ({center_x}, {center_y}):")

        # 1. Response WITHOUT sigma transformation
        print("\n" + "-" * 70)
        print("WITHOUT Sigma Transformation:")
        print("-" * 70)
        response_no_sigma = compute_response_map_manual(
            image, center_x, center_y, patch_expert, window_size,
            apply_sigma=False, sigma_components=sigma_components
        )
        stats_no_sigma = analyze_response_map(response_no_sigma, "Response (no sigma)")

        # 2. Response WITH sigma transformation
        print("\n" + "-" * 70)
        print("WITH Sigma Transformation:")
        print("-" * 70)
        response_with_sigma = compute_response_map_manual(
            image, center_x, center_y, patch_expert, window_size,
            apply_sigma=True, sigma_components=sigma_components
        )
        stats_with_sigma = analyze_response_map(response_with_sigma, "Response (with sigma)")

        # 3. Compare the two
        print("\n" + "-" * 70)
        print("COMPARISON:")
        print("-" * 70)
        print(f"  Sharpness change: {stats_no_sigma['sharpness']:.3f} -> {stats_with_sigma['sharpness']:.3f}")
        print(f"  Dynamic range change: {stats_no_sigma['dynamic_range']:.3f} -> {stats_with_sigma['dynamic_range']:.3f}")
        print(f"  Std change: {stats_no_sigma['std']:.6f} -> {stats_with_sigma['std']:.6f}")

        if stats_with_sigma['sharpness'] < stats_no_sigma['sharpness'] * 0.8:
            print(f"  WARNING: Sigma transformation REDUCES sharpness significantly!")

        if stats_with_sigma['dynamic_range'] < stats_no_sigma['dynamic_range'] * 0.8:
            print(f"  WARNING: Sigma transformation REDUCES dynamic range significantly!")

    print("\n" + "=" * 70)
    print("Diagnostic Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
