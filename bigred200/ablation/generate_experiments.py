"""
Generate Parameter Grid for Ablation Study

Creates experiment configurations using Latin Hypercube Sampling
for efficient exploration of the parameter space.

Output: CSV manifest mapping SLURM_ARRAY_TASK_ID to parameter configs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json
import argparse
from scipy.stats import qmc


# Default parameter ranges for ablation study
DEFAULT_PARAMETER_RANGES = {
    # KDE sigma - affects mean-shift kernel bandwidth
    'sigma': [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],

    # Regularization weight - affects shape prior strength
    'regularization': [10, 15, 20, 25, 30, 35],

    # Maximum iterations per optimization phase
    'max_iterations': [5, 10, 15, 20, 25],

    # Convergence threshold - early stopping
    'convergence_threshold': [0.01, 0.05, 0.1, 0.2],

    # Weight multiplier - patch confidence weighting
    'weight_multiplier': [0.0, 1.0, 3.0, 5.0, 7.0],
}

# Focused parameter ranges for jaw region optimization
JAW_FOCUSED_RANGES = {
    'sigma': [1.5, 2.0, 2.5, 3.0, 3.5],  # Wider range for jaw
    'regularization': [10, 12, 15, 18, 20],  # Lower values
    'max_iterations': [15, 20, 25, 30],  # More iterations
    'convergence_threshold': [0.01, 0.02, 0.05],  # Stricter
    'weight_multiplier': [3.0, 5.0, 7.0],
}


def generate_full_grid(param_ranges: Dict[str, List]) -> pd.DataFrame:
    """
    Generate full factorial grid of all parameter combinations.

    Args:
        param_ranges: Dict mapping parameter names to lists of values

    Returns:
        DataFrame with one row per configuration
    """
    import itertools

    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())

    # Generate all combinations
    combinations = list(itertools.product(*param_values))

    # Create DataFrame
    df = pd.DataFrame(combinations, columns=param_names)
    df['experiment_id'] = range(len(df))

    # Reorder columns
    cols = ['experiment_id'] + param_names
    df = df[cols]

    return df


def generate_latin_hypercube(param_ranges: Dict[str, List],
                              n_samples: int,
                              seed: int = 42) -> pd.DataFrame:
    """
    Generate Latin Hypercube Sampling of parameter space.

    More efficient than full grid for high-dimensional spaces.

    Args:
        param_ranges: Dict mapping parameter names to lists of values
        n_samples: Number of configurations to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with one row per configuration
    """
    param_names = list(param_ranges.keys())
    n_params = len(param_names)

    # Create Latin Hypercube sampler
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    samples = sampler.random(n=n_samples)

    # Map samples to parameter values
    configs = []
    for sample in samples:
        config = {}
        for i, (name, values) in enumerate(param_ranges.items()):
            # Map [0, 1] to index in values list
            idx = int(sample[i] * len(values))
            idx = min(idx, len(values) - 1)  # Clamp to valid range
            config[name] = values[idx]
        configs.append(config)

    df = pd.DataFrame(configs)
    df['experiment_id'] = range(len(df))

    # Reorder columns
    cols = ['experiment_id'] + param_names
    df = df[cols]

    return df


def generate_random_sample(param_ranges: Dict[str, List],
                           n_samples: int,
                           seed: int = 42) -> pd.DataFrame:
    """
    Generate random sample of parameter configurations.

    Args:
        param_ranges: Dict mapping parameter names to lists of values
        n_samples: Number of configurations to generate
        seed: Random seed

    Returns:
        DataFrame with one row per configuration
    """
    np.random.seed(seed)

    param_names = list(param_ranges.keys())

    configs = []
    for _ in range(n_samples):
        config = {}
        for name, values in param_ranges.items():
            config[name] = np.random.choice(values)
        configs.append(config)

    df = pd.DataFrame(configs)
    df['experiment_id'] = range(len(df))

    cols = ['experiment_id'] + param_names
    df = df[cols]

    return df


def create_experiment_manifest(mode: str = 'latin_hypercube',
                                n_samples: int = 500,
                                param_ranges: Optional[Dict] = None,
                                output_path: str = 'bigred200/config/experiment_manifest.csv',
                                seed: int = 42) -> pd.DataFrame:
    """
    Create experiment manifest for SLURM array jobs.

    Args:
        mode: 'full', 'latin_hypercube', 'random', or 'jaw_focused'
        n_samples: Number of samples for LHS/random modes
        param_ranges: Custom parameter ranges (None = use defaults)
        output_path: Path to save manifest CSV
        seed: Random seed

    Returns:
        DataFrame with experiment configurations
    """
    # Select parameter ranges
    if param_ranges is not None:
        ranges = param_ranges
    elif mode == 'jaw_focused':
        ranges = JAW_FOCUSED_RANGES
    else:
        ranges = DEFAULT_PARAMETER_RANGES

    # Generate configurations
    if mode == 'full':
        df = generate_full_grid(ranges)
        print(f"Generated full grid: {len(df)} configurations")
    elif mode == 'latin_hypercube':
        df = generate_latin_hypercube(ranges, n_samples, seed)
        print(f"Generated Latin Hypercube: {len(df)} configurations")
    elif mode == 'random':
        df = generate_random_sample(ranges, n_samples, seed)
        print(f"Generated random sample: {len(df)} configurations")
    elif mode == 'jaw_focused':
        df = generate_latin_hypercube(ranges, n_samples, seed)
        print(f"Generated jaw-focused LHS: {len(df)} configurations")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Save manifest
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved manifest to: {output_path}")

    # Also save parameter ranges as JSON
    ranges_path = output_path.parent / 'parameter_ranges.json'
    with open(ranges_path, 'w') as f:
        json.dump({k: [float(v) if isinstance(v, (int, float)) else v for v in vals]
                   for k, vals in ranges.items()}, f, indent=2)
    print(f"Saved parameter ranges to: {ranges_path}")

    # Print summary
    print("\nParameter ranges:")
    for name, values in ranges.items():
        print(f"  {name}: {values}")

    return df


def main():
    """Command-line interface for experiment generation."""
    parser = argparse.ArgumentParser(description='Generate ablation study experiments')
    parser.add_argument('--mode', choices=['full', 'latin_hypercube', 'random', 'jaw_focused'],
                        default='latin_hypercube', help='Generation mode')
    parser.add_argument('--n-samples', type=int, default=500,
                        help='Number of samples for LHS/random modes')
    parser.add_argument('--output', default='bigred200/config/experiment_manifest.csv',
                        help='Output CSV path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    create_experiment_manifest(
        mode=args.mode,
        n_samples=args.n_samples,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
