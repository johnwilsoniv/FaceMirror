#!/usr/bin/env python3
"""
Test convergence after reshape bug fix.

Compares C++ OpenFace vs Python pyCLNF full pipeline:
1. Use SAME bbox (from C++ MTCNN) for both
2. Run initialization
3. Run full fitting
4. Compare final landmarks

This isolates convergence testing from detector variance.
"""

import subprocess
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def run_cpp_full_pipeline(image_path, output_dir):
    """Run C++ OpenFace full pipeline (detection + fitting)."""
    binary_path = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    cmd = [
        str(binary_path),
        "-f", str(image_path),
        "-out_dir", str(output_dir),
        "-verbose"
    ]

    print("="*70)
    print("C++ OpenFace - Full Pipeline")
    print("="*70)

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Parse debug output
    bbox_match = re.search(r'DEBUG_BBOX: ([\d.]+),([\d.]+),([\d.]+),([\d.]+)', output)
    init_params_match = re.search(r'DEBUG_INIT_PARAMS: scale=([\d.]+) wx=([\d.e+-]+) wy=([\d.e+-]+) wz=([\d.e+-]+) tx=([\d.]+) ty=([\d.]+)', output)
    init_landmarks_match = re.search(r'DEBUG_INIT_LANDMARKS: ([\d.,]+)', output)
    final_params_match = re.search(r'DEBUG_PARAMS: scale=([\d.]+) wx=([\d.e+-]+) wy=([\d.e+-]+) wz=([\d.e+-]+) tx=([\d.]+) ty=([\d.]+)', output)

    # Also read the CSV output for final landmarks
    csv_path = output_dir / f"{image_path.stem}.csv"

    data = {}

    if bbox_match:
        data['bbox'] = (
            float(bbox_match.group(1)),
            float(bbox_match.group(2)),
            float(bbox_match.group(3)),
            float(bbox_match.group(4))
        )
        print(f"BBox: {data['bbox']}")

    if init_params_match:
        data['init_params'] = np.array([
            float(init_params_match.group(1)),
            float(init_params_match.group(2)),
            float(init_params_match.group(3)),
            float(init_params_match.group(4)),
            float(init_params_match.group(5)),
            float(init_params_match.group(6))
        ])
        print(f"Init params: scale={data['init_params'][0]:.4f}")

    if init_landmarks_match:
        coords = [float(x) for x in init_landmarks_match.group(1).split(',')]
        data['init_landmarks'] = np.array(coords).reshape(-1, 2)
        print(f"Init landmarks: mean=({data['init_landmarks'][:, 0].mean():.2f}, {data['init_landmarks'][:, 1].mean():.2f})")

    if final_params_match:
        data['final_params'] = np.array([
            float(final_params_match.group(1)),
            float(final_params_match.group(2)),
            float(final_params_match.group(3)),
            float(final_params_match.group(4)),
            float(final_params_match.group(5)),
            float(final_params_match.group(6))
        ])
        print(f"Final params: scale={data['final_params'][0]:.4f}")

    # Read final landmarks from CSV
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)

        # Extract x and y coordinates (columns are like x_0, y_0, x_1, y_1, ...)
        x_cols = [col for col in df.columns if col.startswith('x_')]
        y_cols = [col for col in df.columns if col.startswith('y_')]

        if x_cols and y_cols:
            x_coords = df[x_cols].values[0]
            y_coords = df[y_cols].values[0]
            final_landmarks = np.column_stack([x_coords, y_coords])
            data['final_landmarks'] = final_landmarks
            print(f"Final landmarks: mean=({final_landmarks[:, 0].mean():.2f}, {final_landmarks[:, 1].mean():.2f})")

    return data

def run_python_full_pipeline(image_path, bbox):
    """Run Python pyCLNF full pipeline using same bbox."""
    from pyclnf import CLNF

    print("\n" + "="*70)
    print("Python pyCLNF - Full Pipeline")
    print("="*70)

    print(f"Using C++ bbox: {bbox}")

    # Load image
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize CLNF
    model_dir = Path("pyclnf/models")
    clnf = CLNF(model_dir=str(model_dir))

    # Get initialization params manually (for comparison)
    init_params = clnf.pdm.init_params(bbox)
    init_landmarks = clnf.pdm.params_to_landmarks_2d(init_params)

    init_params_global = init_params[:6]
    print(f"Init params: scale={init_params_global[0]:.4f}")
    print(f"Init landmarks: mean=({init_landmarks[:, 0].mean():.2f}, {init_landmarks[:, 1].mean():.2f})")

    # Run fitting with the bbox (fit handles initialization internally)
    print("Running fitting...")
    final_landmarks, info = clnf.fit(gray, bbox, return_params=True)

    # Get final params from info
    final_params = info.get('params', None)
    if final_params is not None:
        final_params_global = final_params[:6]
    else:
        # Fallback: try to get current params from PDM
        final_params = clnf.pdm.params_to_landmarks_2d.im_self.params if hasattr(clnf.pdm, 'params') else init_params
        final_params_global = final_params[:6]

    print(f"Fitting completed")
    print(f"Final params: scale={final_params_global[0]:.4f}")
    print(f"Final landmarks: mean=({final_landmarks[:, 0].mean():.2f}, {final_landmarks[:, 1].mean():.2f})")

    data = {
        'bbox': bbox,
        'init_params': init_params_global,
        'init_landmarks': init_landmarks,
        'final_params': final_params_global,
        'final_landmarks': final_landmarks
    }

    return data

def visualize_convergence(image_path, cpp_data, py_data, output_path):
    """Visualize convergence comparison."""
    img = cv2.imread(str(image_path))

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Row 1: Initial landmarks
    # C++ Init
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if 'init_landmarks' in cpp_data:
        lm = cpp_data['init_landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='red', s=15, alpha=0.7)
    ax.set_title(f"C++ Initialization\nScale: {cpp_data['init_params'][0]:.4f}", fontsize=12, fontweight='bold')
    ax.axis('off')

    # Python Init
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if 'init_landmarks' in py_data:
        lm = py_data['init_landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='blue', s=15, alpha=0.7)
    ax.set_title(f"Python Initialization\nScale: {py_data['init_params'][0]:.4f}", fontsize=12, fontweight='bold')
    ax.axis('off')

    # Init Overlay
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if 'init_landmarks' in cpp_data:
        lm = cpp_data['init_landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='red', s=20, alpha=0.6, label='C++', marker='o')
    if 'init_landmarks' in py_data:
        lm = py_data['init_landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='blue', s=20, alpha=0.6, label='Python', marker='x')
    init_diff = np.linalg.norm(cpp_data['init_landmarks'] - py_data['init_landmarks'], axis=1).mean()
    ax.set_title(f"Init Overlay\nMean diff: {init_diff:.3f}px", fontsize=12, fontweight='bold')
    ax.legend()
    ax.axis('off')

    # Row 2: Final landmarks
    # C++ Final
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if 'final_landmarks' in cpp_data:
        lm = cpp_data['final_landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='red', s=15, alpha=0.7)
    ax.set_title(f"C++ Final Result\nScale: {cpp_data['final_params'][0]:.4f}", fontsize=12, fontweight='bold')
    ax.axis('off')

    # Python Final
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if 'final_landmarks' in py_data:
        lm = py_data['final_landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='blue', s=15, alpha=0.7)
    ax.set_title(f"Python Final Result\nScale: {py_data['final_params'][0]:.4f}", fontsize=12, fontweight='bold')
    ax.axis('off')

    # Final Overlay
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if 'final_landmarks' in cpp_data:
        lm = cpp_data['final_landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='red', s=20, alpha=0.6, label='C++', marker='o')
    if 'final_landmarks' in py_data:
        lm = py_data['final_landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='blue', s=20, alpha=0.6, label='Python', marker='x')
    final_diff = np.linalg.norm(cpp_data['final_landmarks'] - py_data['final_landmarks'], axis=1).mean()
    ax.set_title(f"Final Overlay\nMean diff: {final_diff:.3f}px", fontsize=12, fontweight='bold')
    ax.legend()
    ax.axis('off')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")
    plt.close()

    return init_diff, final_diff

def main():
    """Main convergence test."""
    image_path = Path("test_frames/frame_001.jpg")
    output_dir = Path("test_output/convergence_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CONVERGENCE TEST WITH RESHAPE BUG FIX")
    print("="*70)
    print(f"Testing: {image_path}")
    print()

    # Run C++ full pipeline
    cpp_data = run_cpp_full_pipeline(image_path, output_dir)

    if 'bbox' not in cpp_data:
        print("ERROR: C++ failed to detect face!")
        return

    # Run Python using SAME bbox
    py_data = run_python_full_pipeline(image_path, cpp_data['bbox'])

    # Visualize
    output_path = output_dir / f"{image_path.stem}_convergence_comparison.jpg"
    init_diff, final_diff = visualize_convergence(image_path, cpp_data, py_data, output_path)

    # Summary
    print("\n" + "="*70)
    print("CONVERGENCE TEST RESULTS")
    print("="*70)

    print("\nInitialization:")
    print(f"  Mean landmark diff: {init_diff:.3f} px")
    if init_diff < 1.0:
        print(f"  ✓ EXCELLENT: Initialization matches perfectly!")
    elif init_diff < 5.0:
        print(f"  ✓ GOOD: Small initialization difference")
    else:
        print(f"  ✗ POOR: Large initialization difference")

    print("\nAfter Fitting:")
    print(f"  Mean landmark diff: {final_diff:.3f} px")
    if final_diff < 2.0:
        print(f"  ✓ EXCELLENT: Convergence matches C++ perfectly!")
    elif final_diff < 5.0:
        print(f"  ✓ GOOD: Convergence is close")
    elif final_diff < 10.0:
        print(f"  ⚠ MODERATE: Some convergence difference")
    else:
        print(f"  ✗ POOR: Significant convergence difference")

    print("\nConclusion:")
    if init_diff < 1.0 and final_diff < 5.0:
        print("  🎉 SUCCESS! The reshape bug fix solved the convergence problem!")
        print("  Python pyCLNF now matches C++ OpenFace performance.")
    elif init_diff < 1.0 and final_diff < 10.0:
        print("  ✓ The reshape fix worked! Minor differences may be due to:")
        print("    - Floating-point precision")
        print("    - Patch expert implementation differences")
    else:
        print("  ⚠ There may be other issues beyond the reshape bug.")

    print(f"\nVisualization: {output_path}")

if __name__ == "__main__":
    main()
