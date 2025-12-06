#!/usr/bin/env python3
"""
Compare Python vs C++ OpenFace pipelines on 100 frames.
- Python uses PyMTCNN for face detection (no C++ bbox)
- C++ uses native OpenFace MTCNN
Both pipelines operate independently.
"""

import sys
import time
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import subprocess
import os

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))


def run_cpp_openface(video_path: str, max_frames: int = 100) -> tuple:
    """
    Run C++ OpenFace on video and return results + timing.

    Returns:
        (dataframe, total_time_seconds, fps)
    """
    out_dir = '/tmp/openface_cpp_compare'
    os.makedirs(out_dir, exist_ok=True)

    # Clear previous output
    video_name = Path(video_path).stem
    csv_path = os.path.join(out_dir, f'{video_name}.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)

    openface_path = '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction'

    cmd = [
        openface_path,
        '-f', video_path,
        '-out_dir', out_dir,
        '-aus',  # Action Units
        '-2Dfp',  # 2D landmarks
        '-pose',  # Head pose
        '-verbose'
    ]

    print(f"Running C++ OpenFace...")
    print(f"  Command: {' '.join(cmd)}")

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return None, elapsed, 0

    # Load results
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Limit to max_frames
        if len(df) > max_frames:
            df = df.head(max_frames)
        fps = len(df) / elapsed
        return df, elapsed, fps
    else:
        print(f"  Error: Output not found at {csv_path}")
        return None, elapsed, 0


def run_python_pipeline(video_path: str, max_frames: int = 100) -> tuple:
    """
    Run Python pipeline on video and return results + timing.
    Uses PyMTCNN for face detection (no C++ bbox).

    Returns:
        (dataframe, total_time_seconds, fps, component_times)
    """
    from pyfaceau import FullPythonAUPipeline

    print(f"Initializing Python pipeline...")
    print(f"  Backend: auto (will select best: CoreML > CUDA > CPU)")

    # Initialize pipeline - this uses PyMTCNN internally (no C++ bbox!)
    pipeline = FullPythonAUPipeline(
        pdm_file="pyfaceau/weights/In-the-wild_aligned_PDM_68.txt",
        au_models_dir="pyfaceau/weights/AU_predictors",
        triangulation_file="pyfaceau/weights/tris_68_full.txt",
        patch_expert_file="pyfaceau/weights/svr_patches_0.25_general.txt",
        mtcnn_backend='auto',  # Uses PyMTCNN (CoreML on Apple Silicon)
        use_calc_params=True,
        track_faces=False,
        verbose=False  # Suppress per-frame output
    )

    print(f"  Pipeline initialized with PyMTCNN face detection")

    # Process video
    print(f"Processing {max_frames} frames with Python pipeline...")
    start = time.perf_counter()
    df = pipeline.process_video(
        video_path=video_path,
        max_frames=max_frames
    )
    elapsed = time.perf_counter() - start

    success_count = df['success'].sum() if 'success' in df.columns else len(df)
    fps = success_count / elapsed

    return df, elapsed, fps


def compare_landmarks(python_row, cpp_row) -> float:
    """Compare landmarks between Python and C++ results."""
    errors = []

    for i in range(68):
        # Get Python landmarks (from debug_info if available, else from row)
        py_x_key = f'x_{i}'
        py_y_key = f'y_{i}'

        # Check C++ columns (may have leading space)
        cpp_x_key = f'x_{i}' if f'x_{i}' in cpp_row else f' x_{i}'
        cpp_y_key = f'y_{i}' if f'y_{i}' in cpp_row else f' y_{i}'

        if cpp_x_key in cpp_row and cpp_y_key in cpp_row:
            if py_x_key in python_row and py_y_key in python_row:
                px = python_row[py_x_key]
                py = python_row[py_y_key]
                cx = cpp_row[cpp_x_key]
                cy = cpp_row[cpp_y_key]
                error = np.sqrt((px - cx)**2 + (py - cy)**2)
                errors.append(error)

    return np.mean(errors) if errors else float('inf')


def compare_aus(python_row, cpp_row) -> dict:
    """Compare AU predictions between Python and C++ results."""
    au_list = ['01', '02', '04', '05', '06', '07', '09', '10',
               '12', '14', '15', '17', '20', '23', '25', '26', '45']

    comparison = {}

    for au_num in au_list:
        au_name = f'AU{au_num}_r'

        # Get C++ value (may have leading space)
        cpp_key = au_name if au_name in cpp_row else f' {au_name}'

        if cpp_key in cpp_row and au_name in python_row:
            py_val = python_row[au_name]
            cpp_val = cpp_row[cpp_key]

            if not np.isnan(py_val) and not np.isnan(cpp_val):
                comparison[au_name] = {
                    'python': py_val,
                    'cpp': cpp_val,
                    'error': abs(py_val - cpp_val)
                }

    return comparison


def main():
    print("=" * 80)
    print("PYTHON vs C++ PIPELINE COMPARISON (100 FRAMES)")
    print("=" * 80)
    print()
    print("IMPORTANT: Python uses PyMTCNN for face detection (no C++ bbox)")
    print("Both pipelines operate completely independently")
    print()

    # Video path
    video_path = "S Data/Normal Cohort/IMG_0942.MOV"
    max_frames = 100

    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return

    print(f"Video: {video_path}")
    print(f"Max frames: {max_frames}")
    print()

    # 1. Run C++ OpenFace
    print("-" * 80)
    print("1. RUNNING C++ OPENFACE")
    print("-" * 80)
    cpp_df, cpp_time, cpp_fps = run_cpp_openface(video_path, max_frames)

    if cpp_df is not None:
        print(f"  Processed {len(cpp_df)} frames in {cpp_time:.2f}s")
        print(f"  FPS: {cpp_fps:.2f}")
    else:
        print("  C++ OpenFace failed!")
    print()

    # 2. Run Python pipeline
    print("-" * 80)
    print("2. RUNNING PYTHON PIPELINE (PyMTCNN bbox)")
    print("-" * 80)
    python_df, python_time, python_fps = run_python_pipeline(video_path, max_frames)

    success_count = python_df['success'].sum() if 'success' in python_df.columns else len(python_df)
    print(f"  Processed {len(python_df)} frames ({success_count} successful) in {python_time:.2f}s")
    print(f"  FPS: {python_fps:.2f}")
    print()

    # 3. Compare results
    print("=" * 80)
    print("3. COMPARISON RESULTS")
    print("=" * 80)
    print()

    # FPS comparison
    print("--- SPEED COMPARISON ---")
    print(f"{'Pipeline':<20} {'Total Time':<15} {'FPS':<10} {'vs C++':<10}")
    print("-" * 55)
    print(f"{'C++ OpenFace':<20} {cpp_time:<15.2f}s {cpp_fps:<10.2f} {'(baseline)':<10}")
    print(f"{'Python (PyMTCNN)':<20} {python_time:<15.2f}s {python_fps:<10.2f} {python_fps/cpp_fps:<10.2f}x")
    print()

    # AU accuracy comparison (if both succeeded)
    if cpp_df is not None and python_df is not None:
        print("--- AU ACCURACY COMPARISON ---")

        # Find common frames
        min_frames = min(len(cpp_df), len(python_df))

        au_errors = {}
        au_correlations = {}

        au_list = ['01', '02', '04', '05', '06', '07', '09', '10',
                   '12', '14', '15', '17', '20', '23', '25', '26', '45']

        for au_num in au_list:
            au_name = f'AU{au_num}_r'

            # Get columns
            cpp_col = au_name if au_name in cpp_df.columns else f' {au_name}'

            if cpp_col in cpp_df.columns and au_name in python_df.columns:
                cpp_vals = cpp_df[cpp_col].values[:min_frames]
                py_vals = python_df[au_name].values[:min_frames]

                # Filter valid values
                valid_mask = ~(np.isnan(cpp_vals) | np.isnan(py_vals))
                if valid_mask.sum() > 0:
                    cpp_valid = cpp_vals[valid_mask]
                    py_valid = py_vals[valid_mask]

                    # Mean absolute error
                    mae = np.mean(np.abs(cpp_valid - py_valid))
                    au_errors[au_name] = mae

                    # Correlation
                    if np.std(cpp_valid) > 0 and np.std(py_valid) > 0:
                        corr = np.corrcoef(cpp_valid, py_valid)[0, 1]
                        au_correlations[au_name] = corr

        print(f"{'AU':<10} {'MAE':<12} {'Correlation':<12}")
        print("-" * 35)

        for au_name in sorted(au_errors.keys()):
            mae = au_errors[au_name]
            corr = au_correlations.get(au_name, float('nan'))
            print(f"{au_name:<10} {mae:<12.4f} {corr:<12.4f}")

        if au_errors:
            overall_mae = np.mean(list(au_errors.values()))
            overall_corr = np.mean([c for c in au_correlations.values() if not np.isnan(c)])
            print("-" * 35)
            print(f"{'OVERALL':<10} {overall_mae:<12.4f} {overall_corr:<12.4f}")

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"C++ OpenFace:        {cpp_fps:.2f} FPS")
    print(f"Python (PyMTCNN):    {python_fps:.2f} FPS")
    print(f"Speed ratio:         {python_fps/cpp_fps:.2f}x (Python/C++)")
    print()
    if au_errors:
        print(f"AU Mean Absolute Error: {overall_mae:.4f}")
        print(f"AU Correlation:         {overall_corr:.4f}")
    print()
    print("NOTE: Python pipeline uses PyMTCNN (CoreML on Apple Silicon)")
    print("      for face detection - completely independent of C++ bbox.")
    print()

    # Save results
    if python_df is not None:
        python_df.to_csv('python_pipeline_results.csv', index=False)
        print("Python results saved to: python_pipeline_results.csv")
    if cpp_df is not None:
        cpp_df.to_csv('cpp_openface_results.csv', index=False)
        print("C++ results saved to: cpp_openface_results.csv")


if __name__ == '__main__':
    main()
