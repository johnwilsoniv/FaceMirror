#!/usr/bin/env python3
"""
Compare C++ OpenFace initialization (using C++ MTCNN) vs
Python pyCLNF initialization (using pyMTCNN).

This tests real-world scenario where each uses its own face detector.
"""

import subprocess
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def run_cpp_openface(image_path, output_dir):
    """Run C++ OpenFace with its built-in MTCNN."""
    binary_path = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    cmd = [
        str(binary_path),
        "-f", str(image_path),
        "-out_dir", str(output_dir),
        "-verbose"
    ]

    print("="*70)
    print("C++ OpenFace (with C++ MTCNN)")
    print("="*70)

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Parse debug output
    bbox_match = re.search(r'DEBUG_BBOX: ([\d.]+),([\d.]+),([\d.]+),([\d.]+)', output)
    init_params_match = re.search(r'DEBUG_INIT_PARAMS: scale=([\d.]+) wx=([\d.e+-]+) wy=([\d.e+-]+) wz=([\d.e+-]+) tx=([\d.]+) ty=([\d.]+)', output)
    init_landmarks_match = re.search(r'DEBUG_INIT_LANDMARKS: ([\d.,]+)', output)

    data = {}

    if bbox_match:
        data['bbox'] = {
            'x': float(bbox_match.group(1)),
            'y': float(bbox_match.group(2)),
            'width': float(bbox_match.group(3)),
            'height': float(bbox_match.group(4))
        }
        print(f"BBox (C++ MTCNN): {data['bbox']}")

    if init_params_match:
        data['params'] = np.array([
            float(init_params_match.group(1)),
            float(init_params_match.group(2)),
            float(init_params_match.group(3)),
            float(init_params_match.group(4)),
            float(init_params_match.group(5)),
            float(init_params_match.group(6))
        ])
        print(f"Init params: scale={data['params'][0]:.6f}, tx={data['params'][4]:.2f}, ty={data['params'][5]:.2f}")

    if init_landmarks_match:
        coords = [float(x) for x in init_landmarks_match.group(1).split(',')]
        data['landmarks'] = np.array(coords).reshape(-1, 2)
        print(f"Init landmarks: {data['landmarks'].shape}, mean=({data['landmarks'][:, 0].mean():.2f}, {data['landmarks'][:, 1].mean():.2f})")

    return data

def run_python_pymtcnn(image_path):
    """Run Python with pyMTCNN face detector."""
    import sys
    import importlib.util
    from pyclnf.core import PDM

    # Import OpenFaceMTCNN directly without loading pyfaceau package
    spec = importlib.util.spec_from_file_location(
        "openface_mtcnn",
        "pyfaceau/pyfaceau/detectors/openface_mtcnn.py"
    )
    openface_mtcnn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(openface_mtcnn)
    OpenFaceMTCNN = openface_mtcnn.OpenFaceMTCNN

    print("\n" + "="*70)
    print("Python pyCLNF (with pyMTCNN)")
    print("="*70)

    # Load image
    img = cv2.imread(str(image_path))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect face with pyMTCNN
    detector = OpenFaceMTCNN()
    bboxes, landmarks = detector.detect(rgb_img)

    if len(bboxes) == 0:
        print("ERROR: No face detected by pyMTCNN!")
        return None

    # Get first detection
    # OpenFaceMTCNN returns bboxes as (x1, y1, x2, y2)
    x1, y1, x2, y2 = bboxes[0]
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1

    bbox = {
        'x': float(x),
        'y': float(y),
        'width': float(w),
        'height': float(h)
    }

    print(f"BBox (pyMTCNN): {bbox}")

    # Load PDM and initialize
    model_dir = Path("pyclnf/models/exported_pdm")
    pdm = PDM(str(model_dir))

    bbox_tuple = (bbox['x'], bbox['y'], bbox['width'], bbox['height'])
    params = pdm.init_params(bbox_tuple)

    # Get landmarks
    landmarks = pdm.params_to_landmarks_2d(params)

    params_global = params[:6]
    print(f"Init params: scale={params_global[0]:.6f}, tx={params_global[4]:.2f}, ty={params_global[5]:.2f}")
    print(f"Init landmarks: {landmarks.shape}, mean=({landmarks[:, 0].mean():.2f}, {landmarks[:, 1].mean():.2f})")

    data = {
        'bbox': bbox,
        'params': params_global,
        'landmarks': landmarks
    }

    return data

def visualize_comparison(image_path, cpp_data, py_data, output_path):
    """Visualize the comparison."""
    img = cv2.imread(str(image_path))

    fig, axes = plt.subplots(2, 2, figsize=(18, 18))

    # 1. C++ OpenFace
    ax = axes[0, 0]
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if 'landmarks' in cpp_data:
        lm = cpp_data['landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='red', s=25, alpha=0.7, label='C++ Init')
    if 'bbox' in cpp_data:
        bbox = cpp_data['bbox']
        rect = plt.Rectangle((bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                             fill=False, color='red', linewidth=2, linestyle='--')
        ax.add_patch(rect)
    ax.set_title(f"C++ OpenFace (C++ MTCNN)\nScale: {cpp_data['params'][0]:.4f}", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.axis('off')

    # 2. Python pyCLNF
    ax = axes[0, 1]
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if 'landmarks' in py_data:
        lm = py_data['landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='blue', s=25, alpha=0.7, label='Python Init')
    if 'bbox' in py_data:
        bbox = py_data['bbox']
        rect = plt.Rectangle((bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                             fill=False, color='blue', linewidth=2, linestyle='--')
        ax.add_patch(rect)
    ax.set_title(f"Python pyCLNF (pyMTCNN)\nScale: {py_data['params'][0]:.4f}", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.axis('off')

    # 3. Overlay
    ax = axes[1, 0]
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if 'landmarks' in cpp_data:
        lm = cpp_data['landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='red', s=35, alpha=0.6, label='C++ Init', marker='o')
    if 'landmarks' in py_data:
        lm = py_data['landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='blue', s=35, alpha=0.6, label='Python Init', marker='x')

    # Draw bboxes
    if 'bbox' in cpp_data:
        bbox = cpp_data['bbox']
        rect = plt.Rectangle((bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                             fill=False, color='red', linewidth=2, linestyle='--', label='C++ BBox')
        ax.add_patch(rect)
    if 'bbox' in py_data:
        bbox = py_data['bbox']
        rect = plt.Rectangle((bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                             fill=False, color='blue', linewidth=2, linestyle='--', label='Python BBox')
        ax.add_patch(rect)

    ax.set_title("Overlay: Red=C++, Blue=Python", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.axis('off')

    # 4. Detailed comparison
    ax = axes[1, 1]
    ax.axis('off')

    # Calculate differences
    cpp_bbox = cpp_data['bbox']
    py_bbox = py_data['bbox']
    cpp_lm = cpp_data['landmarks']
    py_lm = py_data['landmarks']
    cpp_params = cpp_data['params']
    py_params = py_data['params']

    diff_lm = cpp_lm - py_lm
    diff_magnitude = np.linalg.norm(diff_lm, axis=1)

    text = "COMPARISON ANALYSIS\n"
    text += "="*50 + "\n\n"

    text += "Bounding Boxes:\n"
    text += f"  C++ MTCNN:\n"
    text += f"    x={cpp_bbox['x']:.2f}, y={cpp_bbox['y']:.2f}\n"
    text += f"    w={cpp_bbox['width']:.2f}, h={cpp_bbox['height']:.2f}\n"
    text += f"  pyMTCNN:\n"
    text += f"    x={py_bbox['x']:.2f}, y={py_bbox['y']:.2f}\n"
    text += f"    w={py_bbox['width']:.2f}, h={py_bbox['height']:.2f}\n"
    text += f"  Differences:\n"
    text += f"    Δx={abs(cpp_bbox['x'] - py_bbox['x']):.2f}, "
    text += f"Δy={abs(cpp_bbox['y'] - py_bbox['y']):.2f}\n"
    text += f"    Δw={abs(cpp_bbox['width'] - py_bbox['width']):.2f}, "
    text += f"Δh={abs(cpp_bbox['height'] - py_bbox['height']):.2f}\n\n"

    text += "Initialization Parameters:\n"
    text += f"  Scale:  C++={cpp_params[0]:.6f}, Py={py_params[0]:.6f}, "
    text += f"Δ={abs(cpp_params[0] - py_params[0]):.6f}\n"
    text += f"  TX:     C++={cpp_params[4]:.2f}, Py={py_params[4]:.2f}, "
    text += f"Δ={abs(cpp_params[4] - py_params[4]):.2f}px\n"
    text += f"  TY:     C++={cpp_params[5]:.2f}, Py={py_params[5]:.2f}, "
    text += f"Δ={abs(cpp_params[5] - py_params[5]):.2f}px\n\n"

    text += "Initialization Landmarks:\n"
    text += f"  Mean diff:   {diff_magnitude.mean():.3f} px\n"
    text += f"  Max diff:    {diff_magnitude.max():.3f} px\n"
    text += f"  Min diff:    {diff_magnitude.min():.3f} px\n"
    text += f"  Std diff:    {diff_magnitude.std():.3f} px\n\n"

    text += "Top 5 Largest Differences:\n"
    worst_indices = np.argsort(diff_magnitude)[-5:][::-1]
    for idx in worst_indices:
        text += f"  Landmark {idx}: {diff_magnitude[idx]:.2f}px "
        text += f"({diff_lm[idx, 0]:+.1f}, {diff_lm[idx, 1]:+.1f})\n"

    # Determine if initialization is good
    text += "\n" + "="*50 + "\n"
    if diff_magnitude.mean() < 5.0:
        text += "✓ EXCELLENT: Initializations match well!\n"
        text += "  Both detectors produce compatible results.\n"
    elif diff_magnitude.mean() < 15.0:
        text += "⚠ GOOD: Small differences due to detector variance.\n"
        text += "  Should still converge well.\n"
    else:
        text += "✗ WARNING: Large initialization differences!\n"
        text += "  May affect convergence.\n"

    ax.text(0.05, 0.95, text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"Visualization saved: {output_path}")
    plt.close()

def main():
    """Main comparison function."""
    image_path = Path("test_frames/frame_001.jpg")
    output_dir = Path("test_output/mtcnn_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("MTCNN INITIALIZATION COMPARISON")
    print("="*70)
    print(f"Image: {image_path}")
    print()

    # Run C++ OpenFace (uses C++ MTCNN)
    cpp_data = run_cpp_openface(image_path, output_dir)

    if not cpp_data or 'bbox' not in cpp_data:
        print("ERROR: C++ OpenFace failed!")
        return

    # Run Python pyCLNF (uses pyMTCNN)
    py_data = run_python_pymtcnn(image_path)

    if not py_data or 'bbox' not in py_data:
        print("ERROR: Python pyCLNF failed!")
        return

    # Create visualization
    output_path = output_dir / f"{image_path.stem}_mtcnn_comparison.jpg"
    visualize_comparison(image_path, cpp_data, py_data, output_path)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    bbox_diff_x = abs(cpp_data['bbox']['x'] - py_data['bbox']['x'])
    bbox_diff_y = abs(cpp_data['bbox']['y'] - py_data['bbox']['y'])
    bbox_diff_w = abs(cpp_data['bbox']['width'] - py_data['bbox']['width'])
    bbox_diff_h = abs(cpp_data['bbox']['height'] - py_data['bbox']['height'])

    print(f"BBox differences: Δx={bbox_diff_x:.2f}, Δy={bbox_diff_y:.2f}, Δw={bbox_diff_w:.2f}, Δh={bbox_diff_h:.2f}")

    scale_diff = abs(cpp_data['params'][0] - py_data['params'][0])
    tx_diff = abs(cpp_data['params'][4] - py_data['params'][4])
    ty_diff = abs(cpp_data['params'][5] - py_data['params'][5])

    print(f"Param differences: Δscale={scale_diff:.6f}, Δtx={tx_diff:.2f}px, Δty={ty_diff:.2f}px")

    lm_diff = np.linalg.norm(cpp_data['landmarks'] - py_data['landmarks'], axis=1)
    print(f"Landmark differences: mean={lm_diff.mean():.3f}px, max={lm_diff.max():.3f}px")

    print()
    if lm_diff.mean() < 5.0:
        print("✓ Result: Excellent agreement between C++ and Python initialization!")
    elif lm_diff.mean() < 15.0:
        print("⚠ Result: Good agreement - small detector variance is normal")
    else:
        print("✗ Result: Significant differences detected")

if __name__ == "__main__":
    main()
