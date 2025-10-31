#!/usr/bin/env python3
"""Test just importing the pipeline module"""

import multiprocessing
import sys

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork', force=True)
        print("✓ Fork set")
    except RuntimeError:
        print("⚠ Fork already set")

print("About to import full_python_au_pipeline...")
sys.path.insert(0, '../pyfhog/src')

from full_python_au_pipeline import FullPythonAUPipeline

print("✓ Import successful!")
print("This means the crash is NOT happening during import")
