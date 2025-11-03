#!/usr/bin/env python3
"""
Analyze where Python CLNF diverges from expected behavior.

This script loads diagnostic JSON files and performs deeper analysis:
- Identifies which iteration divergence begins
- Analyzes response map quality
- Checks for numerical instability
- Suggests specific bug locations
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def load_diagnostic(json_path):
    """Load diagnostic JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_convergence_trajectory(diagnostic):
    """
    Analyze convergence trajectory to identify issues.

    Args:
        diagnostic: Diagnostic dict from JSON

    Returns:
        analysis: Dict with findings
    """
    iterations = diagnostic['iterations']
    analysis = {
        'total_iterations': len(iterations) - 2,  # Exclude init and final
        'converged': diagnostic['errors']['converged'],
        'issues': []
    }

    # Check if stuck (not converging)
    movements = [it['avg_movement'] for it in iterations if it['iteration'] >= 0]
    if len(movements) > 3:
        # Check if movement is decreasing
        if movements[-1] > movements[0] * 0.9:
            analysis['issues'].append({
                'type': 'convergence_stalled',
                'description': 'Average movement not decreasing significantly',
                'initial_movement': movements[0],
                'final_movement': movements[-1],
            })

    # Check for divergence (increasing error)
    if len(movements) > 1:
        for i in range(1, len(movements)):
            if movements[i] > movements[i-1] * 1.5:
                analysis['issues'].append({
                    'type': 'divergence',
                    'description': f'Movement increased at iteration {i}',
                    'iteration': i,
                    'prev_movement': movements[i-1],
                    'curr_movement': movements[i],
                })

    # Check response map quality
    for it in iterations:
        if it['iteration'] >= 0 and 'response_mean_avg' in it:
            if it['response_mean_avg'] < 0.01:
                analysis['issues'].append({
                    'type': 'weak_responses',
                    'description': f'Very weak response maps at iteration {it["iteration"]}',
                    'iteration': it['iteration'],
                    'response_mean': it['response_mean_avg'],
                })

    # Check jacobian condition number
    for it in iterations:
        if it['iteration'] >= 0 and 'jacobian_condition' in it:
            if it['jacobian_condition'] > 1e10:
                analysis['issues'].append({
                    'type': 'ill_conditioned_jacobian',
                    'description': f'Jacobian is ill-conditioned at iteration {it["iteration"]}',
                    'iteration': it['iteration'],
                    'condition_number': it['jacobian_condition'],
                })

    # Check parameter update magnitude
    for it in iterations:
        if it['iteration'] >= 0 and 'delta_params_norm' in it:
            if it['delta_params_norm'] < 1e-6:
                analysis['issues'].append({
                    'type': 'tiny_parameter_update',
                    'description': f'Very small parameter update at iteration {it["iteration"]}',
                    'iteration': it['iteration'],
                    'delta_params_norm': it['delta_params_norm'],
                })

    return analysis


def identify_bug_location(analysis):
    """
    Identify likely bug location based on analysis.

    Args:
        analysis: Analysis dict from analyze_convergence_trajectory

    Returns:
        bug_location: Dict with suspected bug location and fix
    """
    bug_locations = []

    for issue in analysis['issues']:
        if issue['type'] == 'weak_responses':
            bug_locations.append({
                'file': 'pyfaceau/clnf/cen_patch_experts.py',
                'function': 'CENPatchExperts.response() or CENPatchExpert.response()',
                'issue': 'Response maps are too weak',
                'possible_causes': [
                    'Patch extraction bounds incorrect (coordinate system issue)',
                    'Contrast normalization failing',
                    'im2col_bias creating wrong patch layout',
                    'Neural network weights not loaded correctly',
                ],
                'debug_steps': [
                    'Print extracted patch before/after contrast normalization',
                    'Compare patch extraction coordinates with OpenFace C++',
                    'Verify im2col output matches OpenFace format',
                    'Check neural network forward pass outputs',
                ]
            })

        elif issue['type'] == 'convergence_stalled':
            bug_locations.append({
                'file': 'pyfaceau/clnf/nu_rlms.py',
                'function': 'NURLMSOptimizer._mean_shift_targets()',
                'issue': 'Mean shift not finding correct peak positions',
                'possible_causes': [
                    'Coordinate transformation from response map to image is wrong',
                    'extraction_bounds not accounting for image boundary clamping',
                    'Weighted mean calculation incorrect',
                    'Gaussian smoothing destroying peaks',
                ],
                'debug_steps': [
                    'Visualize response maps and check if peaks are visible',
                    'Print target_landmarks and compare with current landmarks',
                    'Verify extraction_bounds match actual extracted patch',
                    'Check if mean_x, mean_y to target_x, target_y transformation is correct',
                ]
            })

        elif issue['type'] == 'ill_conditioned_jacobian':
            bug_locations.append({
                'file': 'pyfaceau/clnf/nu_rlms.py',
                'function': 'NURLMSOptimizer._compute_jacobian()',
                'issue': 'Jacobian is ill-conditioned',
                'possible_causes': [
                    'PDM eigenvectors not scaled correctly',
                    'Scale factor too small or too large',
                    'Numerical precision issues',
                ],
                'debug_steps': [
                    'Print scale, translation values',
                    'Check PDM eigenvector magnitudes',
                    'Compare jacobian with OpenFace C++ implementation',
                ]
            })

        elif issue['type'] == 'divergence':
            bug_locations.append({
                'file': 'pyfaceau/clnf/nu_rlms.py',
                'function': 'NURLMSOptimizer._solve_regularized_ls()',
                'issue': 'Parameter update causing divergence',
                'possible_causes': [
                    'Regularization too weak',
                    'Residual sign flipped',
                    'Jacobian transposed incorrectly',
                    'Parameter clamping not working',
                ],
                'debug_steps': [
                    'Print delta_params and check magnitude/direction',
                    'Verify JtJ matrix is positive definite',
                    'Check if params are being clamped properly',
                    'Compare regularized_matrix condition number',
                ]
            })

    return bug_locations


def plot_convergence(diagnostic, save_path):
    """
    Plot convergence trajectory.

    Args:
        diagnostic: Diagnostic dict
        save_path: Where to save plot
    """
    iterations = diagnostic['iterations']

    # Extract data
    iter_nums = [it['iteration'] for it in iterations if it['iteration'] >= 0]
    movements = [it['avg_movement'] for it in iterations if it['iteration'] >= 0]
    response_means = [it.get('response_mean_avg', 0) for it in iterations if it['iteration'] >= 0]
    jacobian_conds = [it.get('jacobian_condition', 0) for it in iterations if it['iteration'] >= 0]

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Plot 1: Average movement
    axes[0].plot(iter_nums, movements, 'o-', linewidth=2, markersize=8)
    axes[0].axhline(y=0.01, color='r', linestyle='--', label='Convergence threshold')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Average Movement (pixels)')
    axes[0].set_title('Convergence Trajectory')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Response map strength
    axes[1].plot(iter_nums, response_means, 's-', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Average Response Map Value')
    axes[1].set_title('Response Map Strength')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Jacobian condition number
    if any(jacobian_conds):
        axes[2].semilogy(iter_nums, jacobian_conds, '^-', linewidth=2, markersize=8, color='orange')
        axes[2].axhline(y=1e10, color='r', linestyle='--', label='Ill-conditioned threshold')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Jacobian Condition Number')
        axes[2].set_title('Jacobian Conditioning')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved convergence plot to: {save_path}")
    plt.close()


def print_analysis_report(diagnostic, analysis, bug_locations):
    """
    Print comprehensive analysis report.

    Args:
        diagnostic: Diagnostic dict
        analysis: Analysis dict
        bug_locations: List of suspected bug locations
    """
    print("\n" + "="*80)
    print("CLNF DIVERGENCE ANALYSIS REPORT")
    print("="*80)

    print(f"\nTest: {diagnostic['test_info']['name']} frame {diagnostic['test_info']['frame']}")
    print(f"\nErrors:")
    print(f"  Initialization error: {diagnostic['errors']['init_avg_error']:.2f} pixels")
    print(f"  CLNF error: {diagnostic['errors']['clnf_avg_error']:.2f} pixels")
    print(f"  Improvement: {diagnostic['errors']['improvement']:.2f} pixels")
    print(f"  Converged: {diagnostic['errors']['converged']}")
    print(f"  Iterations: {diagnostic['errors']['num_iterations']}")

    print(f"\n{'='*80}")
    print(f"IDENTIFIED ISSUES ({len(analysis['issues'])} found)")
    print(f"{'='*80}")

    if not analysis['issues']:
        print("✅ No obvious issues detected!")
    else:
        for i, issue in enumerate(analysis['issues'], 1):
            print(f"\nIssue #{i}: {issue['type'].upper()}")
            print(f"  Description: {issue['description']}")
            for key, value in issue.items():
                if key not in ['type', 'description']:
                    print(f"  {key}: {value}")

    print(f"\n{'='*80}")
    print(f"SUSPECTED BUG LOCATIONS ({len(bug_locations)} locations)")
    print(f"{'='*80}")

    for i, loc in enumerate(bug_locations, 1):
        print(f"\nLocation #{i}:")
        print(f"  File: {loc['file']}")
        print(f"  Function: {loc['function']}")
        print(f"  Issue: {loc['issue']}")
        print(f"\n  Possible causes:")
        for cause in loc['possible_causes']:
            print(f"    - {cause}")
        print(f"\n  Debug steps:")
        for step in loc['debug_steps']:
            print(f"    {step}")

    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    if any(issue['type'] == 'weak_responses' for issue in analysis['issues']):
        print("\n⚠️  HIGH PRIORITY: Response maps are weak")
        print("   This is likely the root cause. Focus on:")
        print("   1. Verify patch extraction coordinates match OpenFace")
        print("   2. Check contrast_norm implementation")
        print("   3. Verify im2col_bias produces correct layout")
        print("   4. Compare response map values with OpenFace debug output")

    elif any(issue['type'] == 'convergence_stalled' for issue in analysis['issues']):
        print("\n⚠️  HIGH PRIORITY: Convergence stalled")
        print("   Focus on:")
        print("   1. Mean shift coordinate transformation")
        print("   2. extraction_bounds vs actual patch coordinates")
        print("   3. Weighted mean calculation in _mean_shift_targets")

    else:
        print("\n✅ No critical issues found, but CLNF may still be less accurate than OpenFace")
        print("   Consider:")
        print("   1. Multi-scale optimization (coarse to fine)")
        print("   2. Temporal tracking (use previous frame as initialization)")
        print("   3. Tuning regularization parameter")


def main():
    """Analyze diagnostic results."""

    print("="*80)
    print("CLNF DIVERGENCE ANALYSIS")
    print("="*80)

    results_dir = Path('/tmp/clnf_diagnostic_results')

    # Find all diagnostic JSON files
    json_files = list(results_dir.glob('*_diagnostic.json'))

    if not json_files:
        print(f"\nNo diagnostic files found in {results_dir}")
        print("Run compare_clnf_cpp_vs_python.py first to generate diagnostic data.")
        return

    for json_path in json_files:
        print(f"\nAnalyzing: {json_path.name}")

        # Load diagnostic
        diagnostic = load_diagnostic(json_path)

        # Analyze convergence
        analysis = analyze_convergence_trajectory(diagnostic)

        # Identify bug locations
        bug_locations = identify_bug_location(analysis)

        # Print report
        print_analysis_report(diagnostic, analysis, bug_locations)

        # Plot convergence
        plot_path = json_path.with_suffix('.png')
        plot_convergence(diagnostic, plot_path)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
