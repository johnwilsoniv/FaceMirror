"""
HPC Ablation Study Tools for pyCLNF Parameter Optimization

This package provides tools for systematic parameter sweeps to find
optimal CLNF configuration, designed for Big Red 200 SLURM array jobs.

Key components:
- generate_experiments: Create Latin Hypercube parameter grid
- run_single_experiment: Worker script for one configuration
- aggregate_results: Combine results across experiments
- analyze_significance: Statistical analysis (ANOVA, Tukey HSD)
"""

__all__ = [
    'generate_experiments',
    'run_single_experiment',
    'aggregate_results',
]
