#!/usr/bin/env python3
"""
Setup script for Cython optimization modules

Builds:
1. cython_rotation_update - For 99.9% CalcParams accuracy
2. cython_histogram_median - For 10-20x running median speedup

Usage:
    python setup_cython_modules.py build_ext --inplace

This compiles .pyx files to .so/.pyd shared libraries that can be
imported directly as Python modules.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Compiler optimization flags
extra_compile_args = [
    '-O3',              # Maximum optimization
    '-march=native',     # Optimize for this CPU
    '-ffast-math',      # Fast floating-point math (matches C++)
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'
]

extra_link_args = [
    '-O3'
]

# Extension 1: Rotation update for CalcParams accuracy
rotation_ext = Extension(
    name="cython_rotation_update",
    sources=["cython_rotation_update.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c"
)

# Extension 2: Histogram median for running median performance
histogram_ext = Extension(
    name="cython_histogram_median",
    sources=["cython_histogram_median.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c"
)

# Build configuration
setup(
    name="openface_cython_optimizations",
    version="1.0.0",
    description="Cython optimization modules for OpenFace 2.2 Python replication",
    ext_modules=cythonize(
        [rotation_ext, histogram_ext],
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,    # Disable bounds checking (C-like speed)
            'wraparound': False,     # Disable negative indexing
            'cdivision': True,       # C-style division (no Python exceptions)
            'initializedcheck': False,  # Assume variables are initialized
            'nonecheck': False,      # Don't check for None
            'embedsignature': True,  # Embed function signatures for help()
        },
        annotate=True  # Generate HTML annotation files for profiling
    ),
    zip_safe=False,
)

print("\n" + "=" * 80)
print("Cython Modules Build Complete!")
print("=" * 80)
print("\nGenerated files:")
print("  • cython_rotation_update.so/.pyd - CalcParams rotation update")
print("  • cython_histogram_median.so/.pyd - Running median tracker")
print("  • *.html - Cython annotation files (performance analysis)")
print("\nTo use:")
print("  from cython_rotation_update import update_rotation_cython")
print("  from cython_histogram_median import DualHistogramMedianTrackerCython")
print("\n" + "=" * 80)
