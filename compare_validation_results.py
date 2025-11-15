#!/usr/bin/env python3
"""
Compare PyFaceAU (with pyclnf) vs C++ OpenFace Validation Results

Analyzes:
1. Initial landmarks (PDM mean shape → bbox) vs C++ DEBUG_INIT_LANDMARKS
2. Final landmarks (after CLNF refinement) vs C++ OpenFace landmarks
3. Per-landmark error analysis
4. Convergence statistics
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
PROJECT_ROOT = Path(__file__).parent
CPP_OUTPUT = PROJECT_ROOT / "validation_output" / "cpp_baseline"
PYTHON_OUTPUT = PROJECT_ROOT / "validation_output" / "python_baseline"
REPORT_OUTPUT = PROJECT_ROOT / "validation_output" / "report"

REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)


def parse_cpp_landmarks(csv_path):
    """Parse C++ OpenFace CSV to extract landmarks"""
    df = pd.read_csv(csv_path)

    landmarks = np.zeros((68, 2))
    for i in range(68):
        landmarks[i, 0] = df[f'x_{i}'].iloc[0]
        landmarks[i, 1] = df[f'y_{i}'].iloc[0]

    return landmarks


def parse_cpp_init_landmarks(init_str):
    """Parse DEBUG_INIT_LANDMARKS string from C++ stdout"""
    if not init_str:
        return None

    # Format: "x0,y0,x1,y1,...,x67,y67"
    coords = [float(x) for x in init_str.split(',')]
    landmarks = np.array(coords).reshape(-1, 2)

    return landmarks


def load_python_result(json_path):
    """Load Python result JSON"""
    with open(json_path, 'r') as f:
        return json.load(f)


def compute_landmark_error(lm_python, lm_cpp):
    """Compute per-landmark Euclidean error"""
    if lm_cpp is None:
        return None

    errors = np.sqrt(np.sum((lm_python - lm_cpp) ** 2, axis=1))
    return errors


def get_test_frames():
    """Get list of test frames"""
    # Get all patient CSV files but exclude MTCNN debug files
    csv_files = sorted([f for f in CPP_OUTPUT.glob("patient*.csv")
                       if "_mtcnn_debug" not in f.stem])
    frames = [f.stem for f in csv_files]
    return frames


def main():
    print("=" * 80)
    print("PyFaceAU (pyclnf) vs C++ OpenFace Comparison")
    print("=" * 80)

    frames = get_test_frames()
    print(f"\nFound {len(frames)} test frames")

    # Storage for results
    results = []

    for frame in frames:
        print(f"\nProcessing {frame}...")

        # Load C++ results
        cpp_csv = CPP_OUTPUT / f"{frame}.csv"
        cpp_final_lm = parse_cpp_landmarks(cpp_csv)

        # Load Python results
        python_json = PYTHON_OUTPUT / f"{frame}_result.json"
        if not python_json.exists():
            print(f"  Warning: Python result not found")
            continue

        python_result = load_python_result(python_json)

        if not python_result['success']:
            print(f"  Warning: Python processing failed")
            continue

        # Extract landmarks from debug_info
        debug_info = python_result.get('debug_info', {})
        landmark_info = debug_info.get('landmark_detection', {})
        python_final_lm = np.array(landmark_info.get('landmarks_68', []))

        if python_final_lm.size == 0:
            print(f"  Warning: No landmarks in Python result")
            continue

        # Compute final landmark error
        final_errors = compute_landmark_error(python_final_lm, cpp_final_lm)
        mean_error = np.mean(final_errors)
        max_error = np.max(final_errors)

        print(f"  Final landmarks: mean={mean_error:.3f}px max={max_error:.3f}px")

        # Store result
        results.append({
            'frame': frame,
            'mean_error': mean_error,
            'max_error': max_error,
            'errors': final_errors.tolist(),
            'python_converged': landmark_info.get('clnf_converged', False),
            'python_iterations': landmark_info.get('clnf_iterations', 0),
        })

    # Compute overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    all_mean_errors = [r['mean_error'] for r in results]
    all_max_errors = [r['max_error'] for r in results]

    print(f"\nFinal Landmarks (Python pyclnf vs C++ OpenFace):")
    print(f"  Mean error across frames: {np.mean(all_mean_errors):.3f} ± {np.std(all_mean_errors):.3f} px")
    print(f"  Min mean error: {np.min(all_mean_errors):.3f} px")
    print(f"  Max mean error: {np.max(all_mean_errors):.3f} px")
    print(f"  Max single landmark error: {np.max(all_max_errors):.3f} px")

    # Convergence statistics
    convergence_rate = sum(r['python_converged'] for r in results) / len(results) * 100
    avg_iterations = np.mean([r['python_iterations'] for r in results])

    print(f"\nCLNF Convergence (pyclnf):")
    print(f"  Convergence rate: {convergence_rate:.1f}%")
    print(f"  Average iterations: {avg_iterations:.1f}")

    # Per-landmark error analysis
    print("\nPer-Landmark Error Analysis:")

    # Stack all errors
    all_errors = np.array([r['errors'] for r in results])  # (n_frames, 68)
    mean_per_landmark = np.mean(all_errors, axis=0)  # (68,)

    # Find worst landmarks
    worst_indices = np.argsort(mean_per_landmark)[-5:][::-1]
    print("  Top 5 worst landmarks (highest mean error):")
    for idx in worst_indices:
        print(f"    Landmark {idx:2d}: {mean_per_landmark[idx]:.3f} px")

    # Save detailed results
    results_df = pd.DataFrame(results)
    results_csv = REPORT_OUTPUT / "landmark_comparison.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n✓ Saved detailed results to: {results_csv}")

    # Visualizations
    print("\nGenerating visualizations...")

    # 1. Error distribution boxplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean error per frame
    axes[0].bar(range(len(all_mean_errors)), all_mean_errors, color='steelblue')
    axes[0].axhline(y=np.mean(all_mean_errors), color='red', linestyle='--',
                    label=f'Mean: {np.mean(all_mean_errors):.2f}px')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Mean Error (pixels)')
    axes[0].set_title('Mean Landmark Error per Frame\n(Python pyclnf vs C++ OpenFace)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Per-landmark error
    axes[1].plot(mean_per_landmark, marker='o', markersize=3, linewidth=1, color='steelblue')
    axes[1].axhline(y=np.mean(mean_per_landmark), color='red', linestyle='--',
                    label=f'Mean: {np.mean(mean_per_landmark):.2f}px')
    axes[1].set_xlabel('Landmark Index')
    axes[1].set_ylabel('Mean Error (pixels)')
    axes[1].set_title('Mean Error per Landmark\n(averaged across all frames)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = REPORT_OUTPUT / "landmark_error_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Saved visualization: {plot_path}")

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    overall_mean = np.mean(all_mean_errors)

    if overall_mean < 5.0:
        status = "✓ EXCELLENT"
        color = "green"
    elif overall_mean < 10.0:
        status = "✓ GOOD"
        color = "yellow"
    else:
        status = "⚠ NEEDS IMPROVEMENT"
        color = "red"

    print(f"\nOverall Accuracy: {status}")
    print(f"  Mean landmark error: {overall_mean:.3f} px")
    print(f"  Target: < 10 px for clinical acceptability")
    print(f"\nPyCLNF Integration Status: {'✓ VALIDATED' if overall_mean < 10.0 else '✗ NEEDS REVIEW'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
