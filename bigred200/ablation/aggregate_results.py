"""
Aggregate Ablation Study Results

Combines results from all experiments into summary tables
and identifies optimal configurations.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import argparse


def load_experiment_results(results_dir: str) -> List[Dict]:
    """
    Load all experiment result JSON files from directory.

    Args:
        results_dir: Directory containing experiment_*.json files

    Returns:
        List of result dictionaries
    """
    results_dir = Path(results_dir)
    results = []

    for json_file in sorted(results_dir.glob("experiment_*.json")):
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    print(f"Loaded {len(results)} experiment results")
    return results


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """
    Convert results list to flat DataFrame.

    Args:
        results: List of experiment result dicts

    Returns:
        DataFrame with one row per experiment
    """
    rows = []

    for r in results:
        if r['metrics']['overall']['mean_error'] is None:
            continue

        row = {
            'experiment_id': r['experiment_id'],
        }

        # Add config parameters
        for key, value in r['config'].items():
            row[f'param_{key}'] = value

        # Add overall metrics
        row['mean_error'] = r['metrics']['overall']['mean_error']
        row['std_error'] = r['metrics']['overall']['std_error']
        row['max_error'] = r['metrics']['overall']['max_error']

        # Add region metrics
        for region, metrics in r['metrics']['regions'].items():
            if metrics['mean_error'] is not None:
                row[f'{region}_error'] = metrics['mean_error']

        # Add pose metrics
        for pose_key, value in r['metrics']['pose'].items():
            if value is not None:
                row[f'pose_{pose_key}'] = value

        # Add timing
        row['frames_processed'] = r['timing']['frames_processed']
        row['fps'] = r['timing']['fps']

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def compute_parameter_effects(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Compute main effects for each parameter.

    Args:
        df: Results DataFrame

    Returns:
        Dict mapping parameter name to effect statistics
    """
    param_cols = [c for c in df.columns if c.startswith('param_')]

    effects = {}

    for param_col in param_cols:
        param_name = param_col.replace('param_', '')

        # Group by parameter value
        grouped = df.groupby(param_col)['mean_error'].agg(['mean', 'std', 'count'])

        # Compute effect size (range of means)
        effect_size = grouped['mean'].max() - grouped['mean'].min()

        # Find best value
        best_idx = grouped['mean'].idxmin()
        best_mean = grouped['mean'].min()

        effects[param_name] = {
            'effect_size': float(effect_size),
            'best_value': best_idx,
            'best_mean_error': float(best_mean),
            'values': grouped['mean'].to_dict(),
        }

    # Sort by effect size
    effects = dict(sorted(effects.items(), key=lambda x: -x[1]['effect_size']))

    return effects


def compute_region_effects(df: pd.DataFrame, region: str = 'jaw') -> Dict[str, Dict]:
    """
    Compute parameter effects for a specific region.

    Args:
        df: Results DataFrame
        region: Region name (e.g., 'jaw', 'left_eye')

    Returns:
        Dict mapping parameter name to effect statistics for this region
    """
    param_cols = [c for c in df.columns if c.startswith('param_')]
    region_col = f'{region}_error'

    if region_col not in df.columns:
        return {}

    effects = {}

    for param_col in param_cols:
        param_name = param_col.replace('param_', '')

        grouped = df.groupby(param_col)[region_col].agg(['mean', 'std', 'count'])
        effect_size = grouped['mean'].max() - grouped['mean'].min()

        best_idx = grouped['mean'].idxmin()
        best_mean = grouped['mean'].min()

        effects[param_name] = {
            'effect_size': float(effect_size),
            'best_value': best_idx,
            'best_mean_error': float(best_mean),
            'values': grouped['mean'].to_dict(),
        }

    effects = dict(sorted(effects.items(), key=lambda x: -x[1]['effect_size']))

    return effects


def find_optimal_config(df: pd.DataFrame,
                         target_metric: str = 'mean_error',
                         n_top: int = 10) -> pd.DataFrame:
    """
    Find optimal configurations.

    Args:
        df: Results DataFrame
        target_metric: Metric to optimize
        n_top: Number of top configurations to return

    Returns:
        DataFrame of top configurations
    """
    sorted_df = df.sort_values(target_metric)
    return sorted_df.head(n_top)


def generate_report(df: pd.DataFrame,
                    effects: Dict[str, Dict],
                    jaw_effects: Dict[str, Dict],
                    output_dir: Path):
    """Generate analysis report."""

    report_lines = [
        "=" * 70,
        "ABLATION STUDY RESULTS",
        "=" * 70,
        "",
        f"Total experiments: {len(df)}",
        "",
        "--- OVERALL METRICS ---",
        f"Mean error range: {df['mean_error'].min():.3f} - {df['mean_error'].max():.3f} px",
        f"Best overall: {df['mean_error'].min():.3f} px",
        "",
        "--- PARAMETER IMPORTANCE (Overall) ---",
    ]

    for param, stats in effects.items():
        report_lines.append(f"  {param}: effect={stats['effect_size']:.3f}px, best={stats['best_value']}")

    report_lines.extend([
        "",
        "--- PARAMETER IMPORTANCE (Jaw Region) ---",
    ])

    for param, stats in jaw_effects.items():
        report_lines.append(f"  {param}: effect={stats['effect_size']:.3f}px, best={stats['best_value']}")

    # Top configs
    report_lines.extend([
        "",
        "--- TOP 5 CONFIGURATIONS ---",
    ])

    top5 = find_optimal_config(df, 'mean_error', 5)
    for _, row in top5.iterrows():
        params = {c.replace('param_', ''): row[c] for c in df.columns if c.startswith('param_')}
        report_lines.append(f"  Error={row['mean_error']:.3f}px: {params}")

    report_lines.extend([
        "",
        "--- TOP 5 FOR JAW REGION ---",
    ])

    if 'jaw_error' in df.columns:
        top5_jaw = find_optimal_config(df, 'jaw_error', 5)
        for _, row in top5_jaw.iterrows():
            params = {c.replace('param_', ''): row[c] for c in df.columns if c.startswith('param_')}
            report_lines.append(f"  Jaw={row['jaw_error']:.3f}px: {params}")

    report = "\n".join(report_lines)
    print(report)

    # Save report
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write(report)


def main():
    """Main entry point for aggregation."""
    parser = argparse.ArgumentParser(description='Aggregate ablation study results')
    parser.add_argument('--input-dir', default='bigred200/ablation/results/raw',
                        help='Directory with experiment JSON files')
    parser.add_argument('--output-dir', default='bigred200/ablation/results/aggregated',
                        help='Output directory for aggregated results')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_experiment_results(args.input_dir)

    if not results:
        print("No results found!")
        return

    # Convert to DataFrame
    df = results_to_dataframe(results)
    print(f"Created DataFrame with {len(df)} valid experiments")

    # Save full results
    df.to_csv(output_dir / 'all_results.csv', index=False)

    # Compute effects
    effects = compute_parameter_effects(df)
    jaw_effects = compute_region_effects(df, 'jaw')

    # Save effects
    with open(output_dir / 'parameter_effects.json', 'w') as f:
        json.dump({'overall': effects, 'jaw': jaw_effects}, f, indent=2)

    # Generate report
    generate_report(df, effects, jaw_effects, output_dir)

    # Find optimal config
    optimal = find_optimal_config(df, 'mean_error', 1)
    if len(optimal) > 0:
        optimal_config = {c.replace('param_', ''): optimal.iloc[0][c]
                          for c in df.columns if c.startswith('param_')}
        with open(output_dir / 'optimal_config.json', 'w') as f:
            json.dump(optimal_config, f, indent=2)
        print(f"\nOptimal config saved to: {output_dir / 'optimal_config.json'}")


if __name__ == '__main__':
    main()
