#!/usr/bin/env python3
"""
Comprehensive benchmark script comparing Python AU pipeline vs C++ OpenFace.

This script measures:
1. Accuracy - AU prediction accuracy compared to ground truth
2. Speed - FPS and processing time per frame
3. Memory usage
4. Component-wise performance breakdown
"""

import time
import numpy as np
import cv2
import subprocess
import json
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import psutil
import tracemalloc

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))


class PythonAUPipeline:
    """Python AU pipeline implementation."""

    def __init__(self):
        """Initialize all components."""
        print("Initializing Python pipeline...")

        # Import components
        from pymtcnn import MTCNN
        from pyclnf import CLNF
        from pyfaceau import FullPythonAUPipeline

        # Initialize detectors
        self.detector = MTCNN()  # Auto-selects backend
        self.clnf = CLNF(
            model_dir="pyclnf/models"
        )
        # Initialize full pipeline for AU prediction
        self.au_pipeline = FullPythonAUPipeline(
            pdm_file="pyfaceau/weights/In-the-wild_aligned_PDM_68.txt",
            au_models_dir="pyfaceau/weights/AU_predictors",
            triangulation_file="pyfaceau/weights/tris_68_full.txt",
            patch_expert_file="pyfaceau/weights/svr_patches_0.25_general.txt"
        )

        # Timing storage
        self.times = {
            'face_detection': [],
            'landmark_detection': [],
            'feature_extraction': [],
            'au_prediction': [],
            'total': []
        }

    def process_frame(self, image: np.ndarray) -> Dict:
        """Process single frame through pipeline."""
        start_total = time.perf_counter()
        result = {'success': False}

        # 1. Face Detection
        start = time.perf_counter()
        faces = self.detector.detect(image)
        self.times['face_detection'].append(time.perf_counter() - start)

        if not faces:
            return result

        # Get largest face
        face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
        bbox = face['box']

        # 2. Landmark Detection
        start = time.perf_counter()
        self.clnf.initialize_from_bbox(image, bbox)
        success = self.clnf.fit_image(image)
        self.times['landmark_detection'].append(time.perf_counter() - start)

        if not success:
            return result

        landmarks = self.clnf.get_landmarks()

        # 3. Feature Extraction & AU Prediction
        start = time.perf_counter()
        # Process frame through the full pipeline
        pipeline_result = self.au_pipeline.process_frame(image)
        if pipeline_result and 'aus' in pipeline_result:
            aus = pipeline_result['aus']
        else:
            aus = {}
        self.times['feature_extraction'].append(0)  # Combined with AU prediction
        self.times['au_prediction'].append(time.perf_counter() - start)

        # Total time
        self.times['total'].append(time.perf_counter() - start_total)

        result.update({
            'success': True,
            'bbox': bbox,
            'landmarks': landmarks,
            'aus': aus,
            'confidence': face.get('confidence', 1.0)
        })

        return result

    def get_statistics(self) -> Dict:
        """Get timing statistics."""
        stats = {}
        for component, times in self.times.items():
            if times:
                times_ms = np.array(times) * 1000
                stats[component] = {
                    'mean': np.mean(times_ms),
                    'std': np.std(times_ms),
                    'min': np.min(times_ms),
                    'max': np.max(times_ms),
                    'median': np.median(times_ms)
                }
        return stats


class OpenFaceWrapper:
    """Wrapper for C++ OpenFace executable."""

    def __init__(self, openface_path: str = None):
        """Initialize OpenFace wrapper."""
        # Find OpenFace executable
        if openface_path is None:
            potential_paths = [
                "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction",
                "/usr/local/bin/FeatureExtraction",
                "./OpenFace/build/bin/FeatureExtraction"
            ]

            for path in potential_paths:
                if Path(path).exists():
                    self.openface_path = path
                    break
            else:
                raise FileNotFoundError("OpenFace FeatureExtraction not found. Please specify path.")
        else:
            self.openface_path = openface_path

        print(f"Using OpenFace at: {self.openface_path}")

        # Timing storage
        self.times = []

    def process_video(self, video_path: str, output_dir: str = "openface_output") -> pd.DataFrame:
        """Process video with OpenFace."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Run OpenFace
        cmd = [
            self.openface_path,
            "-f", str(video_path),
            "-out_dir", str(output_path),
            "-aus",  # Action Units
            "-2Dfp",  # 2D landmarks
            "-3Dfp",  # 3D landmarks
            "-pose",  # Head pose
            "-gaze",  # Gaze
            "-verbose"
        ]

        print(f"Running: {' '.join(cmd)}")
        start = time.perf_counter()

        result = subprocess.run(cmd, capture_output=True, text=True)

        elapsed = time.perf_counter() - start
        self.times.append(elapsed)

        # Parse CSV output
        video_name = Path(video_path).stem
        csv_path = output_path / f"{video_name}.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return df
        else:
            print(f"Error: OpenFace output not found at {csv_path}")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            return None

    def process_image(self, image_path: str) -> Dict:
        """Process single image with OpenFace."""
        # Save as temporary video (single frame)
        temp_video = "temp_single_frame.mp4"
        image = cv2.imread(str(image_path))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, 1.0, (image.shape[1], image.shape[0]))
        out.write(image)
        out.release()

        # Process with OpenFace
        df = self.process_video(temp_video)

        # Clean up
        Path(temp_video).unlink()

        if df is not None and len(df) > 0:
            return df.iloc[0].to_dict()
        return None


def compare_au_predictions(python_aus: Dict, openface_aus: pd.Series) -> Dict:
    """Compare AU predictions between Python and OpenFace."""
    # Common AU names
    au_names = [
        'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07',
        'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17',
        'AU20', 'AU23', 'AU25', 'AU26', 'AU45'
    ]

    comparison = {}

    for au in au_names:
        # Get intensity values (AU##_r columns in OpenFace)
        of_intensity_key = f"{au}_r"
        of_class_key = f"{au}_c"

        if of_intensity_key in openface_aus:
            of_value = openface_aus[of_intensity_key]
            of_binary = openface_aus[of_class_key] if of_class_key in openface_aus else (of_value > 0.5)

            # Map AU name to Python output
            au_num = int(au[2:])
            py_value = python_aus.get(f'au{au_num}', 0.0) if python_aus else 0.0

            comparison[au] = {
                'python': py_value,
                'openface': of_value,
                'difference': abs(py_value - of_value),
                'python_binary': py_value > 0.5,
                'openface_binary': of_binary
            }

    return comparison


def benchmark_accuracy(test_data_path: str, ground_truth_path: str = None):
    """Benchmark accuracy against ground truth or between pipelines."""
    print("\n" + "=" * 80)
    print("ACCURACY BENCHMARK")
    print("=" * 80)

    # Initialize pipelines
    python_pipeline = PythonAUPipeline()
    openface = OpenFaceWrapper()

    # Load test data
    test_path = Path(test_data_path)

    if test_path.is_file():
        # Single image
        if test_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            image = cv2.imread(str(test_path))

            # Process with Python
            print("\nProcessing with Python pipeline...")
            py_result = python_pipeline.process_frame(image)

            # Process with OpenFace
            print("Processing with OpenFace...")
            of_result = openface.process_image(str(test_path))

            if py_result['success'] and of_result:
                comparison = compare_au_predictions(py_result.get('aus', {}), pd.Series(of_result))

                print("\nAU Comparison:")
                print("-" * 60)
                print(f"{'AU':<6} {'Python':<10} {'OpenFace':<10} {'Difference':<10} {'Match':<10}")
                print("-" * 60)

                matches = 0
                total = 0

                for au, values in comparison.items():
                    match = values['python_binary'] == values['openface_binary']
                    matches += match
                    total += 1

                    print(f"{au:<6} {values['python']:>9.3f} {values['openface']:>9.3f} "
                          f"{values['difference']:>9.3f} {'✓' if match else '✗':^10}")

                print("-" * 60)
                print(f"Binary accuracy: {matches}/{total} = {matches/total*100:.1f}%")

    elif test_path.is_dir():
        # Process directory of images
        images = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))

        if not images:
            print(f"No images found in {test_path}")
            return

        print(f"Processing {len(images)} images...")

        all_comparisons = []

        for img_path in images[:10]:  # Limit to 10 for testing
            print(f"\nProcessing: {img_path.name}")

            image = cv2.imread(str(img_path))
            py_result = python_pipeline.process_frame(image)
            of_result = openface.process_image(str(img_path))

            if py_result['success'] and of_result:
                comparison = compare_au_predictions(py_result.get('aus', {}), pd.Series(of_result))
                all_comparisons.append(comparison)

        # Aggregate results
        if all_comparisons:
            print("\n" + "=" * 60)
            print("AGGREGATE RESULTS")
            print("=" * 60)

            # Calculate mean absolute error per AU
            au_errors = {}
            for comparison in all_comparisons:
                for au, values in comparison.items():
                    if au not in au_errors:
                        au_errors[au] = []
                    au_errors[au].append(values['difference'])

            print(f"{'AU':<6} {'MAE':<10} {'Std':<10}")
            print("-" * 30)

            for au, errors in sorted(au_errors.items()):
                mae = np.mean(errors)
                std = np.std(errors)
                print(f"{au:<6} {mae:>9.3f} {std:>9.3f}")


def benchmark_speed(test_data_path: str, n_frames: int = 100):
    """Benchmark processing speed."""
    print("\n" + "=" * 80)
    print("SPEED BENCHMARK")
    print("=" * 80)

    # Initialize pipelines
    print("\nInitializing pipelines...")
    python_pipeline = PythonAUPipeline()
    openface = OpenFaceWrapper()

    test_path = Path(test_data_path)

    # Prepare test frames
    frames = []

    if test_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        # Video file
        cap = cv2.VideoCapture(str(test_path))
        for _ in range(min(n_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    else:
        # Image file - duplicate for testing
        image = cv2.imread(str(test_path))
        frames = [image] * min(n_frames, 10)

    print(f"Testing with {len(frames)} frames...")

    # Benchmark Python pipeline
    print("\n1. Python Pipeline Performance:")
    print("-" * 40)

    # Warmup
    for frame in frames[:5]:
        _ = python_pipeline.process_frame(frame)

    # Reset timers
    python_pipeline.times = {k: [] for k in python_pipeline.times}

    # Measure
    start = time.perf_counter()
    successful = 0

    for frame in frames:
        result = python_pipeline.process_frame(frame)
        if result['success']:
            successful += 1

    py_total_time = time.perf_counter() - start

    # Get statistics
    stats = python_pipeline.get_statistics()

    print(f"   Total time: {py_total_time:.2f}s")
    print(f"   Frames processed: {successful}/{len(frames)}")
    print(f"   Average FPS: {successful/py_total_time:.1f}")
    print(f"   Average time per frame: {py_total_time/successful*1000:.1f}ms")

    print("\n   Component breakdown:")
    for component, stat in stats.items():
        if stat['mean'] > 0:
            print(f"     {component:20s}: {stat['mean']:>7.1f}ms ± {stat['std']:>5.1f}ms")

    # Benchmark OpenFace
    print("\n2. OpenFace (C++) Performance:")
    print("-" * 40)

    if test_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        # Process video directly
        start = time.perf_counter()
        df = openface.process_video(str(test_path))
        of_total_time = time.perf_counter() - start

        if df is not None:
            n_frames_processed = len(df)
            print(f"   Total time: {of_total_time:.2f}s")
            print(f"   Frames processed: {n_frames_processed}")
            print(f"   Average FPS: {n_frames_processed/of_total_time:.1f}")
            print(f"   Average time per frame: {of_total_time/n_frames_processed*1000:.1f}ms")

    # Performance comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    py_fps = successful / py_total_time
    of_fps = n_frames_processed / of_total_time if 'of_total_time' in locals() else 0

    print(f"Python FPS: {py_fps:.1f}")
    print(f"OpenFace FPS: {of_fps:.1f}")

    if of_fps > 0:
        speedup = of_fps / py_fps
        if speedup > 1:
            print(f"OpenFace is {speedup:.1f}x faster")
        else:
            print(f"Python is {1/speedup:.1f}x faster")


def benchmark_memory():
    """Benchmark memory usage."""
    print("\n" + "=" * 80)
    print("MEMORY BENCHMARK")
    print("=" * 80)

    # Test image
    test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Python pipeline memory
    print("\n1. Python Pipeline Memory:")
    print("-" * 40)

    tracemalloc.start()
    process = psutil.Process()

    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    python_pipeline = PythonAUPipeline()

    mem_after_init = process.memory_info().rss / 1024 / 1024

    # Process frames
    for _ in range(10):
        _ = python_pipeline.process_frame(test_image)

    mem_after_process = process.memory_info().rss / 1024 / 1024

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"   Memory before init: {mem_before:.1f} MB")
    print(f"   Memory after init: {mem_after_init:.1f} MB")
    print(f"   Memory after processing: {mem_after_process:.1f} MB")
    print(f"   Init overhead: {mem_after_init - mem_before:.1f} MB")
    print(f"   Processing overhead: {mem_after_process - mem_after_init:.1f} MB")
    print(f"   Peak traced: {peak / 1024 / 1024:.1f} MB")


def main():
    """Run comprehensive benchmarks."""
    print("=" * 80)
    print("AU PIPELINE BENCHMARK: Python vs C++ OpenFace")
    print("=" * 80)

    # Use specified test video
    test_file = "Patient Data/Normal Cohort/Shorty.mov"

    # Check if file exists
    if not Path(test_file).exists():
        # Fall back to other test files
        test_files = [
            "calibration_frames/patient1_frame1.jpg",
            "calibration_frames/patient2_frame1.jpg",
            "Patient Data/IMG_0441.MOV",
            "test_image.jpg"
        ]

        for file in test_files:
            if Path(file).exists():
                test_file = file
                break
        else:
            print("\n⚠️  No test file found. Creating synthetic test image...")
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite("test_image.jpg", test_image)
            test_file = "test_image.jpg"

    print(f"\nUsing test file: {test_file}")

    # Run benchmarks
    try:
        # Speed benchmark
        benchmark_speed(test_file, n_frames=50)

        # Accuracy benchmark
        benchmark_accuracy(test_file)

        # Memory benchmark
        benchmark_memory()

    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()