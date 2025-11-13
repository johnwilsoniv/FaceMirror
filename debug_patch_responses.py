"""
Debug patch expert responses to find why PyCLNF isn't tracking landmarks properly.

This will:
1. Extract patches at known landmark positions
2. Compute response maps around those positions
3. Visualize response maps to see if they have meaningful peaks
4. Compare response values at different offsets
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pyclnf.core.patch_expert import CCNFModel
import subprocess
import tempfile
import pandas as pd


def run_openface_get_landmarks(image_path: str, output_dir: str):
    """Get OpenFace ground truth landmarks."""
    openface_bin = Path("~/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction").expanduser()

    cmd = [str(openface_bin), "-f", image_path, "-out_dir", output_dir, "-2Dfp"]
    subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    csv_file = Path(output_dir) / (Path(image_path).stem + ".csv")
    df = pd.read_csv(csv_file)

    landmarks = np.zeros((68, 2))
    for i in range(68):
        x_col = f'x_{i}'
        y_col = f'y_{i}'
        if x_col in df.columns:
            landmarks[i, 0] = df[x_col].iloc[0]
            landmarks[i, 1] = df[y_col].iloc[0]

    return landmarks


def visualize_response_map(ccnf_model, image, landmark_idx, center_x, center_y, window_size=21):
    """
    Compute and visualize response map around a landmark position.

    Args:
        ccnf_model: Loaded CCNF model
        image: Grayscale image
        landmark_idx: Which landmark to test (e.g., 30 for nose tip)
        center_x, center_y: Ground truth position from OpenFace
        window_size: Size of window to sample
    """
    # Get patch expert for this landmark
    scale = 0.25
    view_idx = 0

    scale_model = ccnf_model.scale_models.get(scale)
    if scale_model is None:
        print(f"No scale model for {scale}")
        return None

    view_data = scale_model['views'].get(view_idx)
    if view_data is None:
        print(f"No view data for view {view_idx}")
        return None

    patch_expert = view_data['patches'].get(landmark_idx)
    if patch_expert is None:
        print(f"No patch expert for landmark {landmark_idx}")
        return None

    # Compute response map in window around ground truth position
    response_map = np.zeros((window_size, window_size))
    half_window = window_size // 2

    patch_w = patch_expert.width
    patch_h = patch_expert.height

    # Sample responses at each position in window
    for i in range(window_size):
        for j in range(window_size):
            # Position relative to center
            offset_x = j - half_window
            offset_y = i - half_window

            patch_x = int(center_x + offset_x)
            patch_y = int(center_y + offset_y)

            # Extract patch
            x1 = patch_x - patch_w // 2
            y1 = patch_y - patch_h // 2
            x2 = x1 + patch_w
            y2 = y1 + patch_h

            if x1 >= 0 and y1 >= 0 and x2 < image.shape[1] and y2 < image.shape[0]:
                patch = image[y1:y2, x1:x2]
                if patch.shape[0] == patch_h and patch.shape[1] == patch_w:
                    response_map[i, j] = patch_expert.compute_response(patch)
                else:
                    response_map[i, j] = -1e10
            else:
                response_map[i, j] = -1e10

    return response_map, patch_expert


def analyze_response_map(response_map, window_size):
    """Analyze response map characteristics."""
    center = window_size // 2

    # Peak location
    peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
    peak_value = response_map[peak_idx]

    # Response at ground truth (center)
    center_response = response_map[center, center]

    # Offset of peak from center
    peak_offset_x = peak_idx[1] - center
    peak_offset_y = peak_idx[0] - center

    # Dynamic range
    min_val = response_map[response_map > -1e9].min()
    max_val = response_map.max()
    dynamic_range = max_val - min_val

    # Gradient magnitude at center (approximate)
    if center > 0 and center < window_size - 1:
        grad_x = response_map[center, center + 1] - response_map[center, center - 1]
        grad_y = response_map[center + 1, center] - response_map[center - 1, center]
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    else:
        gradient_mag = 0

    return {
        'peak_location': peak_idx,
        'peak_value': peak_value,
        'center_response': center_response,
        'peak_offset': (peak_offset_x, peak_offset_y),
        'min_value': min_val,
        'max_value': max_val,
        'dynamic_range': dynamic_range,
        'gradient_magnitude': gradient_mag
    }


def main():
    print("=" * 80)
    print("Debugging Patch Expert Responses")
    print("=" * 80)

    # Load CCNF model
    print("\nLoading CCNF model...")
    ccnf = CCNFModel("pyclnf/models", scales=[0.25])
    print("CCNF model loaded")

    # Test on a frame
    video_path = "Patient Data/Normal Cohort/IMG_0434.MOV"
    frame_num = 50

    print(f"\nExtracting frame {frame_num} from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to extract frame")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_path = "/tmp/test_frame.jpg"
    cv2.imwrite(frame_path, frame)

    # Get ground truth landmarks from OpenFace
    print("Getting ground truth landmarks from OpenFace C++...")
    with tempfile.TemporaryDirectory() as tmpdir:
        landmarks = run_openface_get_landmarks(frame_path, tmpdir)

    print(f"Ground truth landmarks loaded: {landmarks.shape}")

    # Test key landmarks
    test_landmarks = [
        (30, "Nose tip"),
        (36, "Left eye outer corner"),
        (45, "Right eye outer corner"),
        (48, "Mouth left corner"),
        (54, "Mouth right corner"),
        (8, "Chin"),
    ]

    results = []

    for landmark_idx, landmark_name in test_landmarks:
        print(f"\n{'='*80}")
        print(f"Testing Landmark {landmark_idx}: {landmark_name}")
        print(f"{'='*80}")

        gt_x, gt_y = landmarks[landmark_idx]
        print(f"Ground truth position: ({gt_x:.1f}, {gt_y:.1f})")

        # Compute response map
        window_size = 21
        response_map, patch_expert = visualize_response_map(
            ccnf, gray, landmark_idx, gt_x, gt_y, window_size
        )

        if response_map is None:
            print(f"  No patch expert available for landmark {landmark_idx}")
            continue

        print(f"Patch expert: {patch_expert.width}x{patch_expert.height}")

        # Analyze response map
        analysis = analyze_response_map(response_map, window_size)

        print(f"\nResponse Map Analysis:")
        print(f"  Peak location: {analysis['peak_location']} (offset: {analysis['peak_offset']})")
        print(f"  Peak value: {analysis['peak_value']:.6f}")
        print(f"  Center (GT) response: {analysis['center_response']:.6f}")
        print(f"  Response range: [{analysis['min_value']:.6f}, {analysis['max_value']:.6f}]")
        print(f"  Dynamic range: {analysis['dynamic_range']:.6f}")
        print(f"  Gradient magnitude at center: {analysis['gradient_magnitude']:.6f}")

        # Check if response map has a clear peak
        if abs(analysis['peak_offset'][0]) <= 1 and abs(analysis['peak_offset'][1]) <= 1:
            status = "✓ GOOD - Peak at or near ground truth"
        else:
            status = f"✗ BAD - Peak offset by {analysis['peak_offset']}"
        print(f"  Status: {status}")

        # Visualize response map
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Response map heatmap
        im = axes[0].imshow(response_map, cmap='jet', interpolation='nearest')
        axes[0].set_title(f'Response Map - Landmark {landmark_idx}\n{landmark_name}')
        axes[0].axhline(y=window_size//2, color='w', linestyle='--', alpha=0.5)
        axes[0].axvline(x=window_size//2, color='w', linestyle='--', alpha=0.5)
        axes[0].plot(analysis['peak_location'][1], analysis['peak_location'][0], 'r*', markersize=15)
        plt.colorbar(im, ax=axes[0])

        # Cross-section through center (horizontal)
        center = window_size // 2
        axes[1].plot(response_map[center, :])
        axes[1].axvline(x=center, color='r', linestyle='--', label='Ground truth')
        axes[1].axvline(x=analysis['peak_location'][1], color='g', linestyle='--', label='Peak')
        axes[1].set_title('Horizontal Cross-Section')
        axes[1].set_xlabel('X offset')
        axes[1].set_ylabel('Response')
        axes[1].legend()
        axes[1].grid(True)

        # Cross-section through center (vertical)
        axes[2].plot(response_map[:, center])
        axes[2].axvline(x=center, color='r', linestyle='--', label='Ground truth')
        axes[2].axvline(x=analysis['peak_location'][0], color='g', linestyle='--', label='Peak')
        axes[2].set_title('Vertical Cross-Section')
        axes[2].set_xlabel('Y offset')
        axes[2].set_ylabel('Response')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        output_path = f"patch_response_debug/landmark_{landmark_idx}_{landmark_name.replace(' ', '_')}.png"
        Path("patch_response_debug").mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved visualization to {output_path}")

        results.append({
            'landmark_idx': landmark_idx,
            'name': landmark_name,
            'peak_offset': analysis['peak_offset'],
            'dynamic_range': analysis['dynamic_range'],
            'gradient_mag': analysis['gradient_magnitude'],
            'status': status
        })

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    for r in results:
        print(f"\nLandmark {r['landmark_idx']} ({r['name']}):")
        print(f"  Peak offset: {r['peak_offset']}")
        print(f"  Dynamic range: {r['dynamic_range']:.6f}")
        print(f"  Gradient magnitude: {r['gradient_mag']:.6f}")
        print(f"  {r['status']}")

    # Check if response maps are providing useful gradients
    good_count = sum(1 for r in results if '✓' in r['status'])
    total_count = len(results)

    print(f"\n{'='*80}")
    print(f"Overall: {good_count}/{total_count} landmarks have peaks near ground truth")

    if good_count < total_count / 2:
        print("\n⚠️  WARNING: Most response maps do NOT peak at ground truth!")
        print("This explains why PyCLNF landmarks aren't tracking properly.")
        print("Possible causes:")
        print("  1. Patch normalization/preprocessing differs from OpenFace")
        print("  2. SVR model weights loaded incorrectly")
        print("  3. Feature extraction has bugs")
    else:
        print("\n✓ Response maps look reasonable")
        print("The issue may be in mean-shift computation or optimization")


if __name__ == "__main__":
    main()
