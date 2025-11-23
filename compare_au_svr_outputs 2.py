#!/usr/bin/env python3
"""
Compare AU SVR outputs between Python and C++ OpenFace for specific AUs.
Focus on AU10, AU12, AU14 which are underperforming (all static models).
"""

import numpy as np
import pandas as pd
import subprocess
import tempfile
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))

from pyfaceau.pipeline import FullPythonAUPipeline
from pyfaceau.prediction.model_parser import OF22ModelParser

# Paths
VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
OPENFACE_BINARY = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
MODELS_DIR = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors"

def get_cpp_aus(video_path, num_frames=30):
    """Get C++ OpenFace AU predictions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            OPENFACE_BINARY,
            "-f", video_path,
            "-out_dir", temp_dir,
            "-q"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"OpenFace error: {result.stderr}")

        csv_file = Path(temp_dir) / f"{Path(video_path).stem}.csv"
        if not csv_file.exists():
            # List files in temp dir
            files = list(Path(temp_dir).glob("*"))
            print(f"Files in temp dir: {files}")
            raise FileNotFoundError(f"CSV not created: {csv_file}")

        df = pd.read_csv(csv_file)
        return df.head(num_frames)

def main():
    print("=" * 80)
    print("AU SVR Output Comparison: Python vs C++")
    print("Focus on Static Models: AU10, AU12, AU14")
    print("=" * 80)

    # Get C++ results
    print("\n[1] Getting C++ OpenFace results...")
    cpp_df = get_cpp_aus(VIDEO_PATH, num_frames=30)

    # Get Python results
    print("\n[2] Getting Python pipeline results...")

    # OpenFace model paths
    openface_base = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace"

    pipeline = FullPythonAUPipeline(
        pdm_file=f"{openface_base}/lib/local/FaceAnalyser/AU_predictors/In-the-wild_aligned_PDM_68.txt",
        au_models_dir=f"{openface_base}/lib/local/FaceAnalyser/AU_predictors",
        triangulation_file=f"{openface_base}/lib/local/FaceAnalyser/AU_predictors/tris_68_full.txt",
        patch_expert_file=f"{openface_base}/build/bin/model/patch_experts/cen_patches_1.00_of.dat",
        verbose=False
    )
    python_df = pipeline.process_video(VIDEO_PATH, max_frames=30)

    # Compare specific AUs
    problem_aus = ['AU10_r', 'AU12_r', 'AU14_r']
    good_aus = ['AU01_r', 'AU02_r', 'AU06_r', 'AU04_r']  # Mix of dynamic and static

    print("\n" + "=" * 80)
    print("Per-AU Analysis (Static vs Dynamic)")
    print("=" * 80)

    # Load model parser to check model types
    parser = OF22ModelParser(MODELS_DIR)
    models = parser.load_all_models(use_recommended=True, use_combined=True)

    all_aus = problem_aus + good_aus

    for au_name in all_aus:
        if au_name not in cpp_df.columns or au_name not in python_df.columns:
            continue

        cpp_vals = cpp_df[au_name].values
        py_vals = python_df[au_name].values

        # Align lengths
        n = min(len(cpp_vals), len(py_vals))
        cpp_vals = cpp_vals[:n]
        py_vals = py_vals[:n]

        # Calculate errors
        errors = np.abs(cpp_vals - py_vals)
        mean_error = np.mean(errors)
        max_error = np.max(errors)

        # Get model info
        model = models.get(au_name, {})
        model_type = model.get('model_type', 'unknown')
        cutoff = model.get('cutoff', 0)
        bias = model.get('bias', 0)

        # Check for systematic bias
        mean_cpp = np.mean(cpp_vals)
        mean_py = np.mean(py_vals)
        bias_diff = mean_py - mean_cpp

        status = "⚠️ PROBLEM" if au_name in problem_aus else "✅ OK"

        print(f"\n{au_name} [{model_type.upper()}] {status}")
        print(f"  Mean error: {mean_error:.4f}")
        print(f"  Max error:  {max_error:.4f}")
        print(f"  C++ mean:   {mean_cpp:.4f}")
        print(f"  Python mean: {mean_py:.4f}")
        print(f"  Bias diff:  {bias_diff:+.4f} (Python - C++)")
        print(f"  Model bias: {bias:.4f}, cutoff: {cutoff:.4f}")

        # Show frame-by-frame for problem AUs
        if au_name in problem_aus:
            print(f"  Frame-by-frame (first 10):")
            for i in range(min(10, n)):
                print(f"    Frame {i}: C++={cpp_vals[i]:.3f}, Py={py_vals[i]:.3f}, err={errors[i]:.3f}")

    # Check for pattern in static vs dynamic
    print("\n" + "=" * 80)
    print("Static vs Dynamic Model Analysis")
    print("=" * 80)

    static_errors = []
    dynamic_errors = []

    for au_name, model in models.items():
        if au_name not in cpp_df.columns or au_name not in python_df.columns:
            continue

        cpp_vals = cpp_df[au_name].values
        py_vals = python_df[au_name].values
        n = min(len(cpp_vals), len(py_vals))

        mean_error = np.mean(np.abs(cpp_vals[:n] - py_vals[:n]))

        if model['model_type'] == 'static':
            static_errors.append((au_name, mean_error))
        else:
            dynamic_errors.append((au_name, mean_error))

    print("\nStatic models:")
    for au, err in sorted(static_errors, key=lambda x: -x[1]):
        print(f"  {au}: {err:.4f}")

    print("\nDynamic models:")
    for au, err in sorted(dynamic_errors, key=lambda x: -x[1]):
        print(f"  {au}: {err:.4f}")

    print(f"\nStatic avg error:  {np.mean([e for _, e in static_errors]):.4f}")
    print(f"Dynamic avg error: {np.mean([e for _, e in dynamic_errors]):.4f}")

if __name__ == "__main__":
    main()
