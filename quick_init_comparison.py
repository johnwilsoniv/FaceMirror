#!/usr/bin/env python3
"""
Quick initialization comparison - tests PDM calc_params logic directly.
"""

import subprocess
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def run_openface_cpp(image_path, output_dir):
    """Run OpenFace C++ and get initialization."""
    binary_path = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    cmd = [str(binary_path), "-f", str(image_path), "-out_dir", str(output_dir), "-verbose"]

    print(f"Running C++ OpenFace...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Parse
    bbox_match = re.search(r'DEBUG_BBOX: ([\d.]+),([\d.]+),([\d.]+),([\d.]+)', output)
    init_params_match = re.search(r'DEBUG_INIT_PARAMS: scale=([\d.]+) wx=([\d.e+-]+) wy=([\d.e+-]+) wz=([\d.e+-]+) tx=([\d.]+) ty=([\d.]+)', output)
    init_landmarks_match = re.search(r'DEBUG_INIT_LANDMARKS: ([\d.,]+)', output)

    data = {}
    if bbox_match:
        data['bbox'] = (float(bbox_match.group(1)), float(bbox_match.group(2)),
                       float(bbox_match.group(3)), float(bbox_match.group(4)))
        print(f"BBox: {data['bbox']}")

    if init_params_match:
        data['params'] = np.array([
            float(init_params_match.group(1)),  # scale
            float(init_params_match.group(2)),  # wx
            float(init_params_match.group(3)),  # wy
            float(init_params_match.group(4)),  # wz
            float(init_params_match.group(5)),  # tx
            float(init_params_match.group(6))   # ty
        ])
        print(f"Init params: {data['params']}")

    if init_landmarks_match:
        coords = [float(x) for x in init_landmarks_match.group(1).split(',')]
        # Coords are already interleaved: x1,y1,x2,y2,... so just reshape into pairs
        data['landmarks'] = np.array(coords).reshape(-1, 2)
        print(f"Init landmarks: {data['landmarks'].shape}, mean=({data['landmarks'][:, 0].mean():.1f}, {data['landmarks'][:, 1].mean():.1f})")

    return data

def run_python_pdm(bbox):
    """Test Python PDM calc_params."""
    from pyclnf.core import PDM

    print(f"\nLoading Python PDM...")
    model_dir = Path("pyclnf/models/exported_pdm")
    pdm = PDM(str(model_dir))

    print(f"Calculating Python init params from bbox...")
    params = pdm.init_params(bbox)  # Returns full params (global + local)

    # Get landmarks
    landmarks = pdm.params_to_landmarks_2d(params)

    # Extract global params (first 6)
    params_global = params[:6]

    print(f"Python params: {params_global}")
    print(f"Python landmarks: {landmarks.shape}, mean=({landmarks[:, 0].mean():.1f}, {landmarks[:, 1].mean():.1f})")

    return params_global, landmarks

def visualize(image_path, cpp_data, py_params, py_landmarks):
    """Create visualization."""
    img = cv2.imread(str(image_path))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # C++
    ax = axes[0]
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if 'landmarks' in cpp_data:
        lm = cpp_data['landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='red', s=15, alpha=0.7)
    ax.set_title(f"C++ Init\nScale={cpp_data['params'][0]:.3f}")
    ax.axis('off')

    # Python
    ax = axes[1]
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.scatter(py_landmarks[:, 0], py_landmarks[:, 1], c='blue', s=15, alpha=0.7)
    ax.set_title(f"Python Init\nScale={py_params[0]:.3f}")
    ax.axis('off')

    # Overlay
    ax = axes[2]
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if 'landmarks' in cpp_data:
        lm = cpp_data['landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='red', s=20, alpha=0.6, label='C++', marker='o')
    ax.scatter(py_landmarks[:, 0], py_landmarks[:, 1], c='blue', s=20, alpha=0.6, label='Python', marker='x')
    ax.legend()
    ax.set_title("Overlay")
    ax.axis('off')

    plt.tight_layout()
    output_path = "test_output/quick_init_comparison.jpg"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

    # Calculate difference
    if 'landmarks' in cpp_data:
        diff = cpp_data['landmarks'] - py_landmarks
        diff_mag = np.linalg.norm(diff, axis=1)
        print(f"\nLandmark differences:")
        print(f"  Mean: {diff_mag.mean():.3f} px")
        print(f"  Max: {diff_mag.max():.3f} px")
        print(f"  Std: {diff_mag.std():.3f} px")

        print(f"\nParameter differences:")
        param_diff = cpp_data['params'] - py_params
        print(f"  Scale: {param_diff[0]:.6f}")
        print(f"  TX: {param_diff[4]:.3f}")
        print(f"  TY: {param_diff[5]:.3f}")

def main():
    image_path = Path("test_frames/frame_001.jpg")
    output_dir = Path("test_output/initialization_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("QUICK INITIALIZATION COMPARISON")
    print("="*60)

    # Run C++
    cpp_data = run_openface_cpp(image_path, output_dir)

    if 'bbox' not in cpp_data:
        print("ERROR: No bbox from C++!")
        return

    # Run Python
    py_params, py_landmarks = run_python_pdm(cpp_data['bbox'])

    # Visualize
    visualize(image_path, cpp_data, py_params, py_landmarks)

    print("\nDone!")

if __name__ == "__main__":
    main()
