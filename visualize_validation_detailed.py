#!/usr/bin/env python3
"""
Create detailed visualizations comparing C++ vs Python landmarks

Visualizations:
1. Individual frame comparisons (C++ green, Python blue)
2. Grid view showing all frames
3. Per-landmark error heatmap
4. Statistical summary
"""

import numpy as np
import pandas as pd
import cv2
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
CALIBRATION_FRAMES = PROJECT_ROOT / "calibration_frames"
CPP_OUTPUT = PROJECT_ROOT / "validation_output" / "cpp_baseline"
PYTHON_OUTPUT = PROJECT_ROOT / "validation_output" / "python_baseline"
VIS_OUTPUT = PROJECT_ROOT / "validation_output" / "visualizations"

VIS_OUTPUT.mkdir(parents=True, exist_ok=True)


def parse_cpp_landmarks(csv_path):
    """Parse C++ OpenFace CSV to extract landmarks"""
    df = pd.read_csv(csv_path)
    landmarks = np.zeros((68, 2))
    for i in range(68):
        landmarks[i, 0] = df[f'x_{i}'].iloc[0]
        landmarks[i, 1] = df[f'y_{i}'].iloc[0]
    return landmarks


def load_cpp_bbox(frame_name):
    """Load C++ MTCNN bbox from debug CSV"""
    mtcnn_csv = CPP_OUTPUT / f"{frame_name}_mtcnn_debug.csv"
    if not mtcnn_csv.exists():
        return None

    df = pd.read_csv(mtcnn_csv)
    # Get final bbox (using correct column names)
    x = df['bbox_x'].iloc[-1]
    y = df['bbox_y'].iloc[-1]
    w = df['bbox_w'].iloc[-1]
    h = df['bbox_h'].iloc[-1]

    return (x, y, w, h)


def load_python_landmarks(json_path):
    """Load Python landmarks from JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    landmark_info = data['debug_info']['landmark_detection']
    landmarks = np.array(landmark_info['landmarks_68'])
    converged = landmark_info['clnf_converged']
    iterations = landmark_info['clnf_iterations']

    # Extract bbox from face detection info
    face_info = data['debug_info']['face_detection']
    bbox_xyxy = face_info['bbox']  # [x1, y1, x2, y2]

    # Convert from [x1, y1, x2, y2] to (x, y, w, h)
    x = bbox_xyxy[0]
    y = bbox_xyxy[1]
    w = bbox_xyxy[2] - bbox_xyxy[0]
    h = bbox_xyxy[3] - bbox_xyxy[1]
    bbox = (x, y, w, h)

    return landmarks, converged, iterations, bbox


def compute_errors(lm_python, lm_cpp):
    """Compute per-landmark errors"""
    errors = np.sqrt(np.sum((lm_python - lm_cpp) ** 2, axis=1))
    return errors


def draw_landmarks_comparison(img, lm_cpp, lm_python, errors, frame_name, bbox_cpp=None, bbox_python=None):
    """Draw both C++ (green) and Python (blue) landmarks with error info and bboxes"""
    vis = img.copy()

    # Draw C++ bbox in GREEN (if available)
    if bbox_cpp is not None:
        x, y, w, h = bbox_cpp
        cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)

    # Draw Python bbox in BLUE (if available)
    if bbox_python is not None:
        x, y, w, h = bbox_python
        cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)

    # Draw C++ landmarks in GREEN
    for i, (x, y) in enumerate(lm_cpp):
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)  # Green

    # Draw Python landmarks in BLUE
    for i, (x, y) in enumerate(lm_python):
        cv2.circle(vis, (int(x), int(y)), 3, (255, 0, 0), -1)  # Blue

    # Draw error lines between corresponding landmarks (yellow)
    for i in range(len(lm_cpp)):
        if errors[i] > 5.0:  # Only show errors > 5px
            pt_cpp = (int(lm_cpp[i, 0]), int(lm_cpp[i, 1]))
            pt_python = (int(lm_python[i, 0]), int(lm_python[i, 1]))
            cv2.line(vis, pt_cpp, pt_python, (0, 255, 255), 1)  # Yellow

    # Add legend
    legend_y = 30
    cv2.putText(vis, "Green: C++ OpenFace (bbox + landmarks)", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, "Blue: Python pyclnf (bbox + landmarks)", (10, legend_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(vis, "Yellow: Error > 5px", (10, legend_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Add error stats
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    cv2.putText(vis, f"Mean error: {mean_err:.2f}px", (10, vis.shape[0] - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Max error: {max_err:.2f}px", (10, vis.shape[0] - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Add frame name
    cv2.putText(vis, frame_name, (10, vis.shape[0] - 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis


def get_frames():
    """Get list of test frames"""
    csv_files = sorted([f for f in CPP_OUTPUT.glob("patient*.csv")
                       if "_mtcnn_debug" not in f.stem])
    return [f.stem for f in csv_files]


def create_individual_visualizations():
    """Create individual frame visualizations"""
    print("Creating individual frame visualizations...")

    frames = get_frames()
    results = []

    for frame in frames:
        print(f"  Processing {frame}...")

        # Load image
        img_path = CALIBRATION_FRAMES / f"{frame}.jpg"
        if not img_path.exists():
            print(f"    Warning: Image not found: {img_path}")
            continue

        img = cv2.imread(str(img_path))

        # Load C++ landmarks
        cpp_csv = CPP_OUTPUT / f"{frame}.csv"
        lm_cpp = parse_cpp_landmarks(cpp_csv)

        # Load Python landmarks and bbox
        python_json = PYTHON_OUTPUT / f"{frame}_result.json"
        if not python_json.exists():
            print(f"    Warning: Python result not found")
            continue

        lm_python, converged, iterations, bbox_python = load_python_landmarks(python_json)

        # Load C++ bbox
        bbox_cpp = load_cpp_bbox(frame)

        # Compute errors
        errors = compute_errors(lm_python, lm_cpp)

        # Create visualization
        vis = draw_landmarks_comparison(img, lm_cpp, lm_python, errors, frame, bbox_cpp, bbox_python)

        # Save
        output_path = VIS_OUTPUT / f"{frame}_comparison.jpg"
        cv2.imwrite(str(output_path), vis)

        results.append({
            'frame': frame,
            'mean_error': np.mean(errors),
            'max_error': np.max(errors),
            'converged': converged,
            'iterations': iterations
        })

    return results


def create_grid_visualization(results):
    """Create grid showing all frames together"""
    print("\nCreating grid visualization...")

    frames = get_frames()
    n_frames = len(frames)

    # Create 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    axes = axes.flatten()

    for idx, frame in enumerate(frames):
        if idx >= len(axes):
            break

        # Load image
        img_path = CALIBRATION_FRAMES / f"{frame}.jpg"
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load landmarks
        cpp_csv = CPP_OUTPUT / f"{frame}.csv"
        lm_cpp = parse_cpp_landmarks(cpp_csv)

        python_json = PYTHON_OUTPUT / f"{frame}_result.json"
        lm_python, converged, iterations, bbox_python = load_python_landmarks(python_json)
        bbox_cpp = load_cpp_bbox(frame)

        # Compute errors
        errors = compute_errors(lm_python, lm_cpp)
        mean_err = np.mean(errors)
        max_err = np.max(errors)

        # Plot
        axes[idx].imshow(img_rgb)

        # Draw C++ bbox (green)
        if bbox_cpp is not None:
            from matplotlib.patches import Rectangle
            x, y, w, h = bbox_cpp
            rect_cpp = Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
            axes[idx].add_patch(rect_cpp)

        # Draw Python bbox (blue)
        if bbox_python is not None:
            from matplotlib.patches import Rectangle
            x, y, w, h = bbox_python
            rect_python = Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
            axes[idx].add_patch(rect_python)

        # Plot C++ landmarks (green)
        axes[idx].scatter(lm_cpp[:, 0], lm_cpp[:, 1], c='green', s=30, alpha=0.8, label='C++')

        # Plot Python landmarks (blue)
        axes[idx].scatter(lm_python[:, 0], lm_python[:, 1], c='blue', s=30, alpha=0.8, label='Python')

        # Draw error lines for large errors
        for i in range(len(lm_cpp)):
            if errors[i] > 10.0:
                axes[idx].plot([lm_cpp[i, 0], lm_python[i, 0]],
                              [lm_cpp[i, 1], lm_python[i, 1]],
                              'yellow', linewidth=1, alpha=0.5)

        axes[idx].set_title(f"{frame}\nMean: {mean_err:.2f}px, Max: {max_err:.2f}px\n"
                           f"Converged: {converged}, Iters: {iterations}",
                           fontsize=10)
        axes[idx].axis('off')
        axes[idx].legend(loc='upper right', fontsize=8)

    # Hide unused subplots
    for idx in range(n_frames, len(axes)):
        axes[idx].axis('off')

    plt.suptitle("Landmark Comparison: C++ OpenFace (Green) vs Python pyclnf (Blue)",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    grid_path = VIS_OUTPUT / "all_frames_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"  Saved grid: {grid_path}")
    plt.close()


def create_error_heatmap():
    """Create heatmap showing per-landmark errors across all frames"""
    print("\nCreating error heatmap...")

    frames = get_frames()
    all_errors = []

    for frame in frames:
        cpp_csv = CPP_OUTPUT / f"{frame}.csv"
        lm_cpp = parse_cpp_landmarks(cpp_csv)

        python_json = PYTHON_OUTPUT / f"{frame}_result.json"
        lm_python, _, _, _ = load_python_landmarks(python_json)

        errors = compute_errors(lm_python, lm_cpp)
        all_errors.append(errors)

    # Create heatmap
    errors_matrix = np.array(all_errors).T  # (68 landmarks, n_frames)

    fig, ax = plt.subplots(figsize=(12, 16))

    sns.heatmap(errors_matrix, cmap='YlOrRd', cbar_kws={'label': 'Error (pixels)'},
                xticklabels=[f.replace('patient', 'P').replace('_frame', 'F') for f in frames],
                yticklabels=range(68), ax=ax)

    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Landmark Index', fontsize=12)
    ax.set_title('Per-Landmark Error Heatmap\n(Python pyclnf vs C++ OpenFace)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    heatmap_path = VIS_OUTPUT / "error_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"  Saved heatmap: {heatmap_path}")
    plt.close()


def create_summary_statistics(results):
    """Create statistical summary visualization"""
    print("\nCreating summary statistics...")

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Error distribution
    axes[0, 0].hist(df['mean_error'], bins=10, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['mean_error'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["mean_error"].mean():.2f}px')
    axes[0, 0].set_xlabel('Mean Error (pixels)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Distribution of Mean Errors', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Error by frame
    axes[0, 1].bar(range(len(df)), df['mean_error'], color='steelblue', alpha=0.7)
    axes[0, 1].axhline(df['mean_error'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["mean_error"].mean():.2f}px')
    axes[0, 1].set_xlabel('Frame Index', fontsize=11)
    axes[0, 1].set_ylabel('Mean Error (pixels)', fontsize=11)
    axes[0, 1].set_title('Mean Error per Frame', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Max error by frame
    axes[1, 0].bar(range(len(df)), df['max_error'], color='coral', alpha=0.7)
    axes[1, 0].axhline(df['max_error'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["max_error"].mean():.2f}px')
    axes[1, 0].set_xlabel('Frame Index', fontsize=11)
    axes[1, 0].set_ylabel('Max Error (pixels)', fontsize=11)
    axes[1, 0].set_title('Maximum Error per Frame', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 4. Convergence stats
    convergence_rate = df['converged'].sum() / len(df) * 100
    avg_iterations = df['iterations'].mean()

    axes[1, 1].text(0.5, 0.7, f"Convergence Rate: {convergence_rate:.1f}%",
                    ha='center', va='center', fontsize=14, fontweight='bold')
    axes[1, 1].text(0.5, 0.5, f"Average Iterations: {avg_iterations:.1f}",
                    ha='center', va='center', fontsize=14)
    axes[1, 1].text(0.5, 0.3, f"Overall Mean Error: {df['mean_error'].mean():.3f} ± {df['mean_error'].std():.3f} px",
                    ha='center', va='center', fontsize=14)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Summary Statistics', fontsize=12, fontweight='bold')

    plt.suptitle('PyCLNF Integration Validation Statistics', fontsize=16, fontweight='bold')
    plt.tight_layout()

    stats_path = VIS_OUTPUT / "summary_statistics.png"
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    print(f"  Saved stats: {stats_path}")
    plt.close()


def main():
    print("=" * 80)
    print("Creating Detailed Validation Visualizations")
    print("=" * 80)

    # 1. Individual frame comparisons
    results = create_individual_visualizations()

    # 2. Grid view
    create_grid_visualization(results)

    # 3. Error heatmap
    create_error_heatmap()

    # 4. Summary statistics
    create_summary_statistics(results)

    print("\n" + "=" * 80)
    print("Visualizations Complete!")
    print("=" * 80)
    print(f"\nOutput directory: {VIS_OUTPUT}")
    print(f"\nGenerated:")
    print(f"  - {len(results)} individual frame comparisons")
    print(f"  - 1 grid view (all frames)")
    print(f"  - 1 error heatmap")
    print(f"  - 1 summary statistics")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
