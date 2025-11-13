#!/usr/bin/env python3
"""
Compare C++ OpenFace and Python pyCLNF initialization.

This script:
1. Runs OpenFace C++ binary on test images
2. Runs pyCLNF on the same images
3. Extracts initialization data from both
4. Visualizes the differences
"""

import subprocess
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add pyclnf to path
sys.path.insert(0, str(Path(__file__).parent))

def run_openface_cpp(image_path, output_dir):
    """Run OpenFace C++ binary and capture debug output."""
    binary_path = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    cmd = [
        str(binary_path),
        "-f", str(image_path),
        "-out_dir", str(output_dir),
        "-verbose"
    ]

    print(f"Running OpenFace C++: {image_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    output = result.stdout + result.stderr

    # Parse debug output
    bbox_match = re.search(r'DEBUG_BBOX: ([\d.]+),([\d.]+),([\d.]+),([\d.]+)', output)
    init_params_match = re.search(r'DEBUG_INIT_PARAMS: scale=([\d.]+) wx=([\d.e+-]+) wy=([\d.e+-]+) wz=([\d.e+-]+) tx=([\d.]+) ty=([\d.]+)', output)
    init_landmarks_match = re.search(r'DEBUG_INIT_LANDMARKS: ([\d.,]+)', output)
    final_params_match = re.search(r'DEBUG_PARAMS: scale=([\d.]+) wx=([\d.e+-]+) wy=([\d.e+-]+) wz=([\d.e+-]+) tx=([\d.]+) ty=([\d.]+)', output)

    data = {}

    if bbox_match:
        data['bbox'] = {
            'x': float(bbox_match.group(1)),
            'y': float(bbox_match.group(2)),
            'width': float(bbox_match.group(3)),
            'height': float(bbox_match.group(4))
        }
        print(f"  BBox: {data['bbox']}")

    if init_params_match:
        data['init_params'] = {
            'scale': float(init_params_match.group(1)),
            'wx': float(init_params_match.group(2)),
            'wy': float(init_params_match.group(3)),
            'wz': float(init_params_match.group(4)),
            'tx': float(init_params_match.group(5)),
            'ty': float(init_params_match.group(6))
        }
        print(f"  Init params: scale={data['init_params']['scale']:.3f}, tx={data['init_params']['tx']:.1f}, ty={data['init_params']['ty']:.1f}")

    if init_landmarks_match:
        coords = [float(x) for x in init_landmarks_match.group(1).split(',')]
        n_landmarks = len(coords) // 2
        landmarks = np.array(coords[:n_landmarks] + coords[n_landmarks:]).reshape(2, -1).T
        data['init_landmarks'] = landmarks
        print(f"  Init landmarks: {n_landmarks} points, mean=({landmarks[:, 0].mean():.1f}, {landmarks[:, 1].mean():.1f})")

    if final_params_match:
        data['final_params'] = {
            'scale': float(final_params_match.group(1)),
            'wx': float(final_params_match.group(2)),
            'wy': float(final_params_match.group(3)),
            'wz': float(final_params_match.group(4)),
            'tx': float(final_params_match.group(5)),
            'ty': float(final_params_match.group(6))
        }
        print(f"  Final params: scale={data['final_params']['scale']:.3f}, tx={data['final_params']['tx']:.1f}, ty={data['final_params']['ty']:.1f}")

    return data

def run_pyclnf(image_path, bbox=None):
    """Run pyCLNF and capture initialization."""
    from pyclnf import CLNF

    print(f"Running pyCLNF: {image_path.name}")

    # Load model
    model_dir = Path(__file__).parent / "pyclnf/models"
    clnf = CLNF(model_dir=str(model_dir))

    # Load image
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use provided bbox or detect
    if bbox is None:
        # Simple face detection for testing (you could use MTCNN here)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            print("  No face detected!")
            return None
        x, y, w, h = faces[0]
        bbox = {'x': float(x), 'y': float(y), 'width': float(w), 'height': float(h)}

    print(f"  BBox: {bbox}")

    # Initialize from bbox
    bbox_rect = (bbox['x'], bbox['y'], bbox['width'], bbox['height'])
    clnf.params_local[:] = 0
    clnf.pdm.calc_params(clnf.params_global, bbox_rect, clnf.params_local)

    # Get initial landmarks
    init_landmarks = clnf.pdm.calc_shape_2d(clnf.params_local, clnf.params_global)

    print(f"  Init params: scale={clnf.params_global[0]:.3f}, tx={clnf.params_global[4]:.1f}, ty={clnf.params_global[5]:.1f}")
    print(f"  Init landmarks: {len(init_landmarks)} points, mean=({init_landmarks[:, 0].mean():.1f}, {init_landmarks[:, 1].mean():.1f})")

    # Fit landmarks (optional - comment out to compare initialization only)
    # success = clnf.fit(gray, num_iterations=5, window_sizes=[11, 9, 7])

    data = {
        'bbox': bbox,
        'init_params': {
            'scale': float(clnf.params_global[0]),
            'wx': float(clnf.params_global[1]),
            'wy': float(clnf.params_global[2]),
            'wz': float(clnf.params_global[3]),
            'tx': float(clnf.params_global[4]),
            'ty': float(clnf.params_global[5])
        },
        'init_landmarks': init_landmarks
    }

    return data

def visualize_comparison(image_path, cpp_data, python_data, output_path):
    """Visualize the initialization comparison."""
    img = cv2.imread(str(image_path))

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # 1. C++ initialization
    ax = axes[0, 0]
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if 'init_landmarks' in cpp_data:
        landmarks = cpp_data['init_landmarks']
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=20, alpha=0.7, label='C++ Init')
    if 'bbox' in cpp_data:
        bbox = cpp_data['bbox']
        rect = plt.Rectangle((bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                             fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
    ax.set_title(f"C++ OpenFace Initialization\nScale: {cpp_data['init_params']['scale']:.3f}", fontsize=12)
    ax.legend()
    ax.axis('off')

    # 2. Python initialization
    ax = axes[0, 1]
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if 'init_landmarks' in python_data:
        landmarks = python_data['init_landmarks']
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='blue', s=20, alpha=0.7, label='Python Init')
    if 'bbox' in python_data:
        bbox = python_data['bbox']
        rect = plt.Rectangle((bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                             fill=False, color='blue', linewidth=2)
        ax.add_patch(rect)
    ax.set_title(f"pyCLNF Initialization\nScale: {python_data['init_params']['scale']:.3f}", fontsize=12)
    ax.legend()
    ax.axis('off')

    # 3. Overlay comparison
    ax = axes[1, 0]
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if 'init_landmarks' in cpp_data:
        landmarks = cpp_data['init_landmarks']
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=30, alpha=0.6, label='C++ Init', marker='o')
    if 'init_landmarks' in python_data:
        landmarks = python_data['init_landmarks']
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='blue', s=30, alpha=0.6, label='Python Init', marker='x')
    ax.set_title("Overlay: Red=C++, Blue=Python", fontsize=12)
    ax.legend()
    ax.axis('off')

    # 4. Difference analysis
    ax = axes[1, 1]
    ax.axis('off')

    # Calculate differences
    if 'init_landmarks' in cpp_data and 'init_landmarks' in python_data:
        cpp_lm = cpp_data['init_landmarks']
        py_lm = python_data['init_landmarks']

        diff = cpp_lm - py_lm
        diff_magnitude = np.linalg.norm(diff, axis=1)

        text = "Initialization Differences:\n\n"
        text += f"Parameters:\n"
        text += f"  Scale: C++={cpp_data['init_params']['scale']:.4f}, "
        text += f"Py={python_data['init_params']['scale']:.4f}, "
        text += f"Diff={cpp_data['init_params']['scale'] - python_data['init_params']['scale']:.4f}\n"
        text += f"  TX: C++={cpp_data['init_params']['tx']:.2f}, "
        text += f"Py={python_data['init_params']['tx']:.2f}, "
        text += f"Diff={cpp_data['init_params']['tx'] - python_data['init_params']['tx']:.2f}\n"
        text += f"  TY: C++={cpp_data['init_params']['ty']:.2f}, "
        text += f"Py={python_data['init_params']['ty']:.2f}, "
        text += f"Diff={cpp_data['init_params']['ty'] - python_data['init_params']['ty']:.2f}\n\n"

        text += f"Landmarks:\n"
        text += f"  Mean diff: {diff_magnitude.mean():.3f} pixels\n"
        text += f"  Max diff: {diff_magnitude.max():.3f} pixels\n"
        text += f"  Min diff: {diff_magnitude.min():.3f} pixels\n"
        text += f"  Std diff: {diff_magnitude.std():.3f} pixels\n\n"

        # Find worst landmarks
        worst_indices = np.argsort(diff_magnitude)[-5:][::-1]
        text += f"Top 5 worst landmarks:\n"
        for i, idx in enumerate(worst_indices):
            text += f"  {idx}: {diff_magnitude[idx]:.2f}px ({diff[idx, 0]:.1f}, {diff[idx, 1]:.1f})\n"

        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_path}")
    plt.close()

def main():
    """Main comparison function."""
    # Setup
    test_frames_dir = Path("test_frames")
    output_dir = Path("test_output/initialization_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find test frames (or create them)
    test_images = sorted(test_frames_dir.glob("*.jpg"))[:3]  # Use up to 3 frames

    if not test_images:
        print("No test frames found! Creating one...")
        test_frames_dir.mkdir(exist_ok=True)
        # Use existing test image
        src = Path("test_output/debug_landmarks_direct.jpg")
        if src.exists():
            import shutil
            dst = test_frames_dir / "frame_001.jpg"
            shutil.copy(src, dst)
            test_images = [dst]
        else:
            print("ERROR: No test images available!")
            return

    print(f"Processing {len(test_images)} test frames...\n")

    # Process each frame
    results = []
    for i, image_path in enumerate(test_images):
        print(f"\n{'='*60}")
        print(f"Frame {i+1}/{len(test_images)}: {image_path.name}")
        print('='*60)

        # Run C++ OpenFace
        cpp_data = run_openface_cpp(image_path, output_dir)

        print()

        # Run Python pyCLNF (use same bbox if available)
        bbox = cpp_data.get('bbox', None) if cpp_data else None
        python_data = run_pyclnf(image_path, bbox=bbox)

        if cpp_data and python_data:
            # Visualize comparison
            output_path = output_dir / f"{image_path.stem}_init_comparison.jpg"
            visualize_comparison(image_path, cpp_data, python_data, output_path)

            results.append({
                'image': image_path.name,
                'cpp': cpp_data,
                'python': python_data
            })

        print()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Processed {len(results)} frames")
    print(f"Results saved in: {output_dir}")

    if results:
        print("\nOverall initialization differences:")
        for result in results:
            cpp = result['cpp']
            py = result['python']

            if 'init_landmarks' in cpp and 'init_landmarks' in py:
                diff = cpp['init_landmarks'] - py['init_landmarks']
                diff_magnitude = np.linalg.norm(diff, axis=1)

                print(f"  {result['image']}:")
                print(f"    Mean landmark diff: {diff_magnitude.mean():.3f} px")
                print(f"    Scale diff: {abs(cpp['init_params']['scale'] - py['init_params']['scale']):.5f}")

if __name__ == "__main__":
    main()
