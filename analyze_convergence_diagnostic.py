#!/usr/bin/env python3
"""
Late-Stage Convergence Diagnostic Analysis

Compares Python vs C++ optimization behavior to understand why Python
doesn't converge as well in the final 5-10 iterations.

Analyzes:
1. Mean-shift decay rates
2. Per-landmark error distribution
3. Regularization dominance
4. Hessian conditioning
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import re

# Output directory
OUTPUT_DIR = Path("convergence_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_cpp_trace(trace_path: str) -> dict:
    """Parse C++ iteration trace file."""
    iterations = []

    with open(trace_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            try:
                iter_data = {
                    'iteration': int(parts[0]),
                    'window_size': int(parts[1]),
                    'phase': parts[2],
                    'scale': float(parts[3]),
                    'rot_x': float(parts[4]),
                    'rot_y': float(parts[5]),
                    'rot_z': float(parts[6]),
                    'trans_x': float(parts[7]),
                    'trans_y': float(parts[8]),
                }

                # Extract additional metrics if available
                if len(parts) > 9:
                    iter_data['mean_shift_norm'] = float(parts[9]) if parts[9] != 'nan' else None
                if len(parts) > 10:
                    iter_data['jwtm_norm'] = float(parts[10]) if parts[10] != 'nan' else None
                if len(parts) > 11:
                    iter_data['update_magnitude'] = float(parts[11]) if parts[11] != 'nan' else None

                iterations.append(iter_data)
            except (ValueError, IndexError):
                continue

    return {'iterations': iterations}


def parse_python_iterations(iterations: list) -> dict:
    """Process Python iteration data."""
    return {'iterations': iterations}


def plot_mean_shift_decay(py_data: dict, cpp_data: dict, output_path: str):
    """Plot mean-shift norm decay for Python vs C++."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Mean-Shift Decay Analysis', fontsize=14, fontweight='bold')

    # Panel 1: Overall mean-shift decay
    ax1 = axes[0, 0]

    py_iters = py_data['iterations']
    cpp_iters = cpp_data['iterations']

    py_ms = [it.get('mean_shift_norm', 0) for it in py_iters]
    cpp_ms = [it.get('mean_shift_norm', 0) for it in cpp_iters if it.get('mean_shift_norm') is not None]

    ax1.plot(range(len(py_ms)), py_ms, 'r-', label='Python', linewidth=2)
    if cpp_ms:
        ax1.plot(range(len(cpp_ms)), cpp_ms, 'b-', label='C++', linewidth=2)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean-Shift Norm')
    ax1.set_title('Mean-Shift Magnitude Over Iterations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Panel 2: Mean-shift decay rate
    ax2 = axes[0, 1]

    if len(py_ms) > 1:
        py_decay = [py_ms[i] / (py_ms[i-1] + 1e-10) for i in range(1, len(py_ms))]
        ax2.plot(range(1, len(py_ms)), py_decay, 'r-', label='Python', linewidth=2)

    if len(cpp_ms) > 1:
        cpp_decay = [cpp_ms[i] / (cpp_ms[i-1] + 1e-10) for i in range(1, len(cpp_ms))]
        ax2.plot(range(1, len(cpp_ms)), cpp_decay, 'b-', label='C++', linewidth=2)

    ax2.axhline(y=1.0, color='gray', linestyle='--', label='No Change')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Decay Ratio (iter/iter-1)')
    ax2.set_title('Mean-Shift Decay Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 2)

    # Panel 3: Update magnitude
    ax3 = axes[1, 0]

    py_update = [it.get('update_magnitude', 0) for it in py_iters]
    cpp_update = [it.get('update_magnitude', 0) for it in cpp_iters if it.get('update_magnitude') is not None]

    ax3.plot(range(len(py_update)), py_update, 'r-', label='Python', linewidth=2)
    if cpp_update:
        ax3.plot(range(len(cpp_update)), cpp_update, 'b-', label='C++', linewidth=2)

    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Update Magnitude')
    ax3.set_title('Parameter Update Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Panel 4: Late-stage comparison (last 15 iterations)
    ax4 = axes[1, 1]

    late_start = max(0, len(py_ms) - 15)
    late_py_ms = py_ms[late_start:]

    ax4.plot(range(late_start, len(py_ms)), late_py_ms, 'r-o', label='Python', linewidth=2, markersize=4)

    if cpp_ms:
        late_start_cpp = max(0, len(cpp_ms) - 15)
        late_cpp_ms = cpp_ms[late_start_cpp:]
        ax4.plot(range(late_start_cpp, len(cpp_ms)), late_cpp_ms, 'b-o', label='C++', linewidth=2, markersize=4)

    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Mean-Shift Norm')
    ax4.set_title('Late-Stage Mean-Shift (Last 15 Iterations)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {'py_final_ms': py_ms[-1] if py_ms else None,
            'cpp_final_ms': cpp_ms[-1] if cpp_ms else None}


def plot_per_landmark_error(py_data: dict, output_path: str):
    """Plot per-landmark mean-shift as heatmap."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Landmark Mean-Shift Analysis', fontsize=14, fontweight='bold')

    py_iters = py_data['iterations']

    # Get iterations with per-landmark data
    iters_with_data = [it for it in py_iters if 'per_landmark_ms' in it and it['per_landmark_ms']]

    if not iters_with_data:
        plt.text(0.5, 0.5, 'No per-landmark data available', ha='center', va='center')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return {}

    # Build heatmap matrix
    n_iters = len(iters_with_data)
    n_landmarks = len(iters_with_data[0]['per_landmark_ms'])

    heatmap = np.zeros((n_landmarks, n_iters))
    for i, it in enumerate(iters_with_data):
        heatmap[:, i] = it['per_landmark_ms']

    # Panel 1: Full heatmap
    ax1 = axes[0, 0]
    im = ax1.imshow(heatmap, aspect='auto', cmap='hot', interpolation='nearest')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Landmark Index')
    ax1.set_title('Per-Landmark Mean-Shift Heatmap')
    plt.colorbar(im, ax=ax1, label='Mean-Shift Magnitude')

    # Panel 2: Final iteration per-landmark
    ax2 = axes[0, 1]
    final_ms = iters_with_data[-1]['per_landmark_ms']

    # Color by face region
    face_regions = {
        'Jaw (0-16)': range(0, 17),
        'Brow R (17-21)': range(17, 22),
        'Brow L (22-26)': range(22, 27),
        'Nose (27-35)': range(27, 36),
        'Eye R (36-41)': range(36, 42),
        'Eye L (42-47)': range(42, 48),
        'Mouth (48-67)': range(48, 68),
    }

    colors = ['gray', 'orange', 'orange', 'green', 'blue', 'blue', 'red']

    for (region, indices), color in zip(face_regions.items(), colors):
        valid_indices = [i for i in indices if i < n_landmarks]
        ax2.bar(valid_indices, [final_ms[i] for i in valid_indices], color=color, alpha=0.7, label=region)

    ax2.set_xlabel('Landmark Index')
    ax2.set_ylabel('Mean-Shift Magnitude')
    ax2.set_title('Final Iteration Per-Landmark Mean-Shift')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Region average over iterations
    ax3 = axes[1, 0]

    for (region, indices), color in zip(face_regions.items(), colors):
        valid_indices = [i for i in indices if i < n_landmarks]
        region_avg = [np.mean([heatmap[i, j] for i in valid_indices]) for j in range(n_iters)]
        ax3.plot(range(n_iters), region_avg, color=color, label=region, linewidth=1.5)

    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Mean Regional Mean-Shift')
    ax3.set_title('Regional Mean-Shift Over Time')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Panel 4: Top problematic landmarks
    ax4 = axes[1, 1]

    # Find landmarks with highest final error
    sorted_indices = np.argsort(final_ms)[::-1][:10]

    for idx in sorted_indices:
        ax4.plot(range(n_iters), heatmap[idx, :], label=f'LM {idx}', linewidth=1.5)

    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Mean-Shift Magnitude')
    ax4.set_title('Top 10 Problematic Landmarks')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {'top_problematic': sorted_indices.tolist(),
            'final_ms_by_landmark': final_ms}


def plot_regularization_analysis(py_data: dict, output_path: str):
    """Plot regularization dominance analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Regularization Analysis', fontsize=14, fontweight='bold')

    py_iters = py_data['iterations']

    # Get nonrigid iterations with diagnostic data
    nonrigid_iters = [it for it in py_iters if it['phase'] == 'nonrigid' and it.get('hessian_cond') is not None]

    if not nonrigid_iters:
        plt.text(0.5, 0.5, 'No regularization diagnostic data available', ha='center', va='center')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return {}

    # Panel 1: Hessian condition number
    ax1 = axes[0, 0]

    hessian_cond = [it['hessian_cond'] for it in nonrigid_iters]
    ax1.plot(range(len(hessian_cond)), hessian_cond, 'r-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Nonrigid Iteration')
    ax1.set_ylabel('Condition Number')
    ax1.set_title('Hessian Matrix Conditioning')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Panel 2: Regularization ratio
    ax2 = axes[0, 1]

    reg_ratio = [it['reg_ratio'] for it in nonrigid_iters if it['reg_ratio'] is not None]
    ax2.plot(range(len(reg_ratio)), reg_ratio, 'b-o', linewidth=2, markersize=4)
    ax2.axhline(y=1.0, color='red', linestyle='--', label='Reg = Data Term')
    ax2.set_xlabel('Nonrigid Iteration')
    ax2.set_ylabel('Reg Term / Data Term')
    ax2.set_title('Regularization Dominance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Data term vs Reg term magnitude
    ax3 = axes[1, 0]

    jtw_norm = [it['jtw_norm'] for it in nonrigid_iters if it['jtw_norm'] is not None]
    reg_norm = [it['reg_term_norm'] for it in nonrigid_iters if it['reg_term_norm'] is not None]

    ax3.plot(range(len(jtw_norm)), jtw_norm, 'g-', label='Data Term (JtWv)', linewidth=2)
    ax3.plot(range(len(reg_norm)), reg_norm, 'purple', label='Reg Term', linewidth=2)
    ax3.set_xlabel('Nonrigid Iteration')
    ax3.set_ylabel('Term Magnitude')
    ax3.set_title('Data vs Regularization Term')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Panel 4: Update magnitude vs regularization ratio
    ax4 = axes[1, 1]

    update_mag = [it['update_magnitude'] for it in nonrigid_iters]

    ax4.scatter(reg_ratio[:len(update_mag)], update_mag, c=range(len(update_mag)), cmap='coolwarm', s=50)
    ax4.set_xlabel('Regularization Ratio')
    ax4.set_ylabel('Update Magnitude')
    ax4.set_title('Update Size vs Regularization')
    ax4.grid(True, alpha=0.3)

    # Add colorbar to show iteration
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, len(update_mag)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax4, label='Iteration')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {'final_hessian_cond': hessian_cond[-1] if hessian_cond else None,
            'final_reg_ratio': reg_ratio[-1] if reg_ratio else None}


def generate_diagnostic_summary(py_data: dict, cpp_data: dict) -> str:
    """Generate text summary of diagnostic findings."""
    summary = []
    summary.append("=" * 80)
    summary.append("LATE-STAGE CONVERGENCE DIAGNOSTIC SUMMARY")
    summary.append("=" * 80)

    py_iters = py_data['iterations']
    cpp_iters = cpp_data['iterations']

    # Mean-shift analysis
    py_ms = [it.get('mean_shift_norm', 0) for it in py_iters]
    cpp_ms = [it.get('mean_shift_norm', 0) for it in cpp_iters if it.get('mean_shift_norm') is not None]

    summary.append("\n1. MEAN-SHIFT ANALYSIS:")
    summary.append(f"   Python final mean-shift: {py_ms[-1]:.4f}" if py_ms else "   Python: No data")
    summary.append(f"   C++ final mean-shift:    {cpp_ms[-1]:.4f}" if cpp_ms else "   C++: No data")

    if py_ms and cpp_ms:
        ratio = py_ms[-1] / (cpp_ms[-1] + 1e-10)
        if ratio > 1.5:
            summary.append(f"   ⚠️  Python mean-shift {ratio:.1f}x larger - may indicate convergence stall")

    # Regularization analysis
    nonrigid_iters = [it for it in py_iters if it['phase'] == 'nonrigid' and it.get('reg_ratio') is not None]
    if nonrigid_iters:
        final_reg_ratio = nonrigid_iters[-1]['reg_ratio']
        summary.append("\n2. REGULARIZATION ANALYSIS:")
        summary.append(f"   Final reg/data ratio: {final_reg_ratio:.4f}")
        if final_reg_ratio > 1.0:
            summary.append("   ⚠️  Regularization dominates - shape prior may be limiting convergence")

        final_cond = nonrigid_iters[-1]['hessian_cond']
        summary.append(f"   Final Hessian condition: {final_cond:.2e}")
        if final_cond > 1e6:
            summary.append("   ⚠️  Poorly conditioned - numerical instability possible")

    # Per-landmark analysis
    if nonrigid_iters and 'per_landmark_ms' in nonrigid_iters[-1]:
        final_ms = nonrigid_iters[-1]['per_landmark_ms']
        sorted_idx = np.argsort(final_ms)[::-1][:5]
        summary.append("\n3. TOP PROBLEMATIC LANDMARKS:")
        for idx in sorted_idx:
            summary.append(f"   Landmark {idx}: {final_ms[idx]:.4f}")

    return "\n".join(summary)


def main():
    """Run diagnostic analysis on latest convergence data."""
    print("=" * 80)
    print("LATE-STAGE CONVERGENCE DIAGNOSTIC ANALYSIS")
    print("=" * 80)

    # Check for trace files
    cpp_trace_path = Path("/tmp/clnf_iteration_traces/cpp_face_trace.txt")
    py_trace_path = Path("/tmp/clnf_iteration_traces/python_face_trace.txt")

    # First, run the convergence analysis to generate fresh data
    print("\nRunning convergence analysis to generate iteration data...")
    import subprocess
    result = subprocess.run(
        ["env", "PYTHONPATH=pyclnf:pymtcnn:.",
         "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3",
         "analyze_convergence.py"],
        capture_output=True,
        text=True,
        cwd="/Users/johnwilsoniv/Documents/SplitFace Open3"
    )

    if result.returncode != 0:
        print(f"Warning: analyze_convergence.py returned non-zero exit code")
        print(result.stderr[:500] if result.stderr else "")

    # Load iteration data from the convergence analysis output
    # For now, we'll parse what we can from the trace files

    py_data = {'iterations': []}
    cpp_data = {'iterations': []}

    if cpp_trace_path.exists():
        print(f"\nParsing C++ trace: {cpp_trace_path}")
        cpp_data = parse_cpp_trace(str(cpp_trace_path))
        print(f"  Found {len(cpp_data['iterations'])} C++ iterations")
    else:
        print(f"\n⚠️  C++ trace not found: {cpp_trace_path}")

    # For Python data, we need to import from the actual analysis
    # This would require modifying analyze_convergence.py to save iteration data
    print("\nNote: Full Python diagnostic data requires running analyze_convergence.py")
    print("      with the enhanced optimizer logging enabled.")

    # Generate plots with available data
    if cpp_data['iterations']:
        print("\nGenerating diagnostic plots...")

        # Mean-shift decay
        ms_results = plot_mean_shift_decay(
            {'iterations': cpp_data['iterations']},  # Use cpp as proxy if no py data
            cpp_data,
            str(OUTPUT_DIR / "diagnostic_mean_shift_decay.png")
        )
        print(f"  Saved: diagnostic_mean_shift_decay.png")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nTo get full Python diagnostics, run analyze_convergence.py and")
    print("inspect the iteration_info returned by clnf.fit().")


if __name__ == "__main__":
    main()
