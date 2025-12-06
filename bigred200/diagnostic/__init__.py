"""
HPC Diagnostic Tools for pyCLNF Analysis

This package provides tools for detailed diagnostic analysis of the pyCLNF
landmark detection pipeline, designed to run on Big Red 200 HPC cluster.

Key components:
- InstrumentedOptimizer: Captures per-iteration convergence data
- HPCDiagnosticRunner: Parallel frame processing with multiprocessing
- DiagnosticAnalyzer: Root cause analysis (jaw vs eye comparison)
"""

from .data_structures import IterationDiagnostic, FrameDiagnostic, LandmarkDiagnostic
from .instrumented_optimizer import InstrumentedNURLMSOptimizer

__all__ = [
    'IterationDiagnostic',
    'FrameDiagnostic',
    'LandmarkDiagnostic',
    'InstrumentedNURLMSOptimizer',
]
