"""
pyAUface - Pure Python OpenFace 2.2 AU Extraction

A complete Python implementation of OpenFace 2.2's AU extraction pipeline
with high-performance parallel processing support.
"""

__version__ = "0.2.0"

from .pipeline import FullPythonAUPipeline
from .parallel_pipeline import ParallelAUPipeline

__all__ = ['FullPythonAUPipeline', 'ParallelAUPipeline']
