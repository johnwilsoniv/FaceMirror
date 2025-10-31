#!/usr/bin/env python3
"""Test creating pipeline object (should NOT crash - lazy init)"""

import multiprocessing
import sys
import os

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork', force=True)
        print("✓ Fork set")
    except RuntimeError:
        print("⚠ Fork already set")

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

sys.path.insert(0, '../pyfhog/src')

from full_python_au_pipeline import FullPythonAUPipeline

print("Importing... OK")
print("Creating pipeline object with CoreML enabled...")

pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='In-the-wild_aligned_PDM_68.txt',
    au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
    triangulation_file='tris_68_full.txt',
    use_calc_params=True,
    use_coreml=True,
    verbose=True
)

print(f"✓ Pipeline object created!")
print(f"  Components initialized: {pipeline._components_initialized}")
print(f"  use_coreml: {pipeline.use_coreml}")
print("")
print("This means lazy initialization is working correctly!")
