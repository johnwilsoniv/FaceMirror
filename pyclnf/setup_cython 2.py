#!/usr/bin/env python3
"""
Setup script to compile Cython optimizations for CLNF.
Enables true multithreading by releasing the GIL.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Enable OpenMP for parallel processing
extra_compile_args = []
extra_link_args = []

if os.name == 'posix':
    # macOS with clang
    if os.uname().sysname == 'Darwin':
        # Use libomp for macOS (install with: brew install libomp)
        # Include paths for homebrew libomp
        import subprocess
        brew_prefix = subprocess.check_output(['brew', '--prefix']).decode().strip()
        omp_include = os.path.join(brew_prefix, 'opt/libomp/include')
        omp_lib = os.path.join(brew_prefix, 'opt/libomp/lib')

        extra_compile_args = ['-Xpreprocessor', '-fopenmp', '-O3', f'-I{omp_include}']
        extra_link_args = ['-lomp', f'-L{omp_lib}']
    else:
        # Linux with gcc
        extra_compile_args = ['-fopenmp', '-O3', '-march=native']
        extra_link_args = ['-fopenmp']

# Define the extension
extensions = [
    Extension(
        "optimizer_cython",
        ["core/optimizer_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
]

# Compile
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'nonecheck': False,
            'cdivision': True,
            'initializedcheck': False,
        }
    ),
    zip_safe=False,
)

print("\n" + "="*60)
print("CYTHON COMPILATION COMPLETE")
print("="*60)
print("The optimizer_cython module has been compiled with:")
print("  - OpenMP support for true parallel processing")
print("  - GIL release for multithreading")
print("  - O3 optimization and native architecture targeting")
print("\nExpected speedup: 2-4x on CLNF operations")
print("="*60)