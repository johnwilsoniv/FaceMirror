#!/usr/bin/env python3
"""
Create detailed accuracy analysis comparing C++ OpenFace vs Python pyCLNF.
Shows all landmarks with numbering and final bboxes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyclnf import CLNF
import subprocess
import csv


def run_cpp_openface(image_path, output_dir):
    """Run C++ OpenFace and extract all results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpp_binary = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    cmd = [
        str(cpp_binary),
        "-f", str(image_path),
        "-out_dir", str(output_dir),
        "-2Dfp", "-3Dfp", "-pdmparams", "-tracked"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse CSV output
    csv_files = list(output_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV output from C++ OpenFace")

    csv_path = csv_files[0]

    # Read landmarks and parameters
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data_row = next(reader)

    # Extract 2D landmarks
    x_indices = [i for i, h in enumerate(header) if h.strip().startswith('x_') and h.strip()[2:].isdigit()]
    y_indices = [i for i, h in enumerate(header) if h.strip().startswith('y_') and h.strip()[2:].isdigit()]

    x_indices.sort(key=lambda i: int(header[i].strip().split('_')[1]))
    y_indices.sort(key=lambda i: int(header[i].strip().split('_')[1]))

    landmarks = []
    for x_idx, y_idx in zip(x_indices, y_indices):
        x = float(data_row[x_idx])
        y = float(data_row[y_idx])
        landmarks.append([x, y])

    landmarks = np.array(landmarks)

    # Extract bbox from tracked output (if available)
    # Estimate bbox from landmarks if not available
    x_min, y_min = landmarks.min(axis=0)
    x_max, y_max = landmarks.max(axis=0)

    # Add margin (OpenFace typically uses ~20% margin)
    margin = 0.1
    width = x_max - x_min
    height = y_max - y_min
    x_min -= margin * width
    y_min -= margin * height
    width *= (1 + 2*margin)
    height *= (1 + 2*margin)

    bbox = (x_min, y_min, width, height)

    # Extract convergence info from stdout if available
    iterations = 0
    converged = False
    if "iterations" in result.stdout.lower():
        # Parse from output
        pass

    return {
        'landmarks': landmarks,
        'bbox': bbox,
        'num_points': len(landmarks),
        'converged': converged,
        'iterations': iterations
    }


def run_python_pyclnf(image_path):
    """Run Python pyCLNF with corrected RetinaFace."""
    image = cv2.imread(str(image_path))

    # Use pyCLNF with integrated detector
    clnf = CLNF()
    landmarks, info = clnf.detect_and_fit(image, return_params=True)

    # Estimate final bbox from landmarks (similar to C++)
    x_min, y_min = landmarks.min(axis=0)
    x_max, y_max = landmarks.max(axis=0)

    margin = 0.1
    width = x_max - x_min
    height = y_max - y_min
    x_min -= margin * width
    y_min -= margin * height
    width *= (1 + 2*margin)
    height *= (1 + 2*margin)

    final_bbox = (x_min, y_min, width, height)

    return {
        'landmarks': landmarks,
        'bbox': final_bbox,
        'detection_bbox': info['bbox'],
        'num_points': len(landmarks),
        'converged': info['converged'],
        'iterations': info['iterations'],
        'params': info.get('params'),
        'info': info
    }


def create_detailed_comparison(image_path, cpp_data, py_data, output_path):
    """Create side-by-side comparison with all landmarks numbered."""

    # Load image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create figure with 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), dpi=150)

    # LEFT: C++ OpenFace
    ax1.imshow(image_rgb)

    # Draw C++ bbox
    x, y, w, h = cpp_data['bbox']
    rect = patches.Rectangle((x, y), w, h, linewidth=4,
                             edgecolor='lime', facecolor='none',
                             linestyle='-', label='Final BBox')
    ax1.add_patch(rect)

    # Draw C++ landmarks
    cpp_lm = cpp_data['landmarks']
    ax1.scatter(cpp_lm[:, 0], cpp_lm[:, 1], c='lime', s=80,
               alpha=0.7, edgecolors='white', linewidths=2, zorder=5)

    # Number key landmarks
    key_landmarks = [0, 8, 16, 17, 21, 22, 26, 27, 30, 33, 36, 39, 42, 45, 48, 54, 57, 60, 64, 67]
    for i in key_landmarks:
        ax1.text(cpp_lm[i, 0] + 5, cpp_lm[i, 1] - 8, str(i),
                fontsize=10, color='yellow', fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='green', alpha=0.8, pad=0.2))

    ax1.set_title(f"C++ OpenFace (MTCNN)\n{cpp_data['num_points']} landmarks",
                  fontsize=18, fontweight='bold', color='green', pad=15)
    ax1.axis('off')

    # RIGHT: Python pyCLNF
    ax2.imshow(image_rgb)

    # Draw detection bbox (dashed)
    dx, dy, dw, dh = py_data['detection_bbox']
    rect_detect = patches.Rectangle((dx, dy), dw, dh, linewidth=2,
                                    edgecolor='orange', facecolor='none',
                                    linestyle='--', alpha=0.5, label='Detection BBox')
    ax2.add_patch(rect_detect)

    # Draw final bbox (solid)
    x, y, w, h = py_data['bbox']
    rect_final = patches.Rectangle((x, y), w, h, linewidth=4,
                                   edgecolor='red', facecolor='none',
                                   linestyle='-', label='Final BBox')
    ax2.add_patch(rect_final)

    # Draw Python landmarks
    py_lm = py_data['landmarks']
    ax2.scatter(py_lm[:, 0], py_lm[:, 1], c='red', s=80,
               alpha=0.7, edgecolors='white', linewidths=2, zorder=5)

    # Number key landmarks
    for i in key_landmarks:
        ax2.text(py_lm[i, 0] + 5, py_lm[i, 1] - 8, str(i),
                fontsize=10, color='yellow', fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='red', alpha=0.8, pad=0.2))

    ax2.set_title(f"Python pyCLNF (RetinaFace Corrected)\n{py_data['num_points']} landmarks, "
                 f"{py_data['iterations']} iterations",
                  fontsize=18, fontweight='bold', color='red', pad=15)
    ax2.legend(loc='upper right', fontsize=12)
    ax2.axis('off')

    # Calculate accuracy metrics
    distances = np.sqrt(np.sum((py_lm - cpp_lm)**2, axis=1))
    mean_error = np.mean(distances)
    max_error = np.max(distances)
    std_error = np.std(distances)

    # Find worst landmarks
    worst_indices = np.argsort(distances)[-5:][::-1]

    # Add title and statistics
    title = f"ACCURACY ANALYSIS: C++ OpenFace vs Python pyCLNF\nMean Error: {mean_error:.2f}px | Max: {max_error:.2f}px | Std: {std_error:.2f}px"
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)

    # Add statistics panel
    stats_text = f"""
LANDMARK ERROR STATISTICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Mean error:        {mean_error:.3f}px
  Max error:         {max_error:.3f}px (landmark {worst_indices[0]})
  Std deviation:     {std_error:.3f}px
  Median error:      {np.median(distances):.3f}px

  Worst 5 landmarks:
    #{worst_indices[0]:2d}: {distances[worst_indices[0]]:6.2f}px
    #{worst_indices[1]:2d}: {distances[worst_indices[1]]:6.2f}px
    #{worst_indices[2]:2d}: {distances[worst_indices[2]]:6.2f}px
    #{worst_indices[3]:2d}: {distances[worst_indices[3]]:6.2f}px
    #{worst_indices[4]:2d}: {distances[worst_indices[4]]:6.2f}px

  Best 5 landmarks:
    #{np.argsort(distances)[0]:2d}: {distances[np.argsort(distances)[0]]:6.2f}px
    #{np.argsort(distances)[1]:2d}: {distances[np.argsort(distances)[1]]:6.2f}px
    #{np.argsort(distances)[2]:2d}: {distances[np.argsort(distances)[2]]:6.2f}px
    #{np.argsort(distances)[3]:2d}: {distances[np.argsort(distances)[3]]:6.2f}px
    #{np.argsort(distances)[4]:2d}: {distances[np.argsort(distances)[4]]:6.2f}px
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    fig.text(0.5, 0.02, stats_text, fontsize=12, ha='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=15))

    plt.tight_layout(rect=[0, 0.15, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Detailed comparison saved to: {output_path}")

    return distances, {
        'mean': mean_error,
        'max': max_error,
        'std': std_error,
        'median': np.median(distances),
        'worst_landmarks': worst_indices,
        'best_landmarks': np.argsort(distances)[:5]
    }


def main():
    print("=" * 70)
    print("ACCURACY ANALYSIS: C++ OpenFace vs Python pyCLNF")
    print("=" * 70)

    # Test image
    image_path = Path("Patient Data/Normal Cohort/IMG_0434.MOV_frame_0000.jpg")

    # Run C++ OpenFace
    print("\n1. Running C++ OpenFace...")
    cpp_output_dir = Path("test_output/accuracy_analysis/cpp_output")
    cpp_data = run_cpp_openface(image_path, cpp_output_dir)
    print(f"   ✅ C++ landmarks: {cpp_data['num_points']} points")

    # Run Python pyCLNF
    print("\n2. Running Python pyCLNF...")
    py_data = run_python_pyclnf(image_path)
    print(f"   ✅ Python landmarks: {py_data['num_points']} points")

    # Create detailed comparison
    print("\n3. Creating detailed comparison visualization...")
    output_path = Path("test_output/accuracy_analysis/cpp_vs_pyclnf_detailed.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    distances, stats = create_detailed_comparison(image_path, cpp_data, py_data, output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Mean landmark error:     {stats['mean']:.3f}px")
    print(f"Max landmark error:      {stats['max']:.3f}px (landmark #{stats['worst_landmarks'][0]})")
    print(f"Std deviation:           {stats['std']:.3f}px")
    print(f"Median error:            {stats['median']:.3f}px")
    print("\nWorst 5 landmarks:")
    for idx in stats['worst_landmarks']:
        print(f"  Landmark #{idx:2d}: {distances[idx]:6.2f}px")
    print("=" * 70)


if __name__ == "__main__":
    main()
