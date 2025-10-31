#!/usr/bin/env python3
"""
AU Pipeline Profiler - Identify Cython Optimization Opportunities

This script profiles the full OpenFace 2.2 AU extraction pipeline to identify
bottlenecks suitable for Cython optimization.

Measures:
1. Feature extraction (C++ OpenFace binary)
2. HOG parsing and loading
3. Geometric feature extraction
4. Running median updates (already optimized with Cython)
5. SVR prediction
6. Temporal smoothing
7. Overall pipeline throughput

Goal: Find components taking >10% of total time for Cython optimization
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
import cProfile
import pstats
import io
from typing import Dict, List, Tuple

# Import the AU predictor
from openface22_au_predictor import OpenFace22AUPredictor


class PipelineProfiler:
    """Profile the full AU extraction pipeline"""

    def __init__(self):
        self.timings = {}
        self.frame_count = 0

    def profile_component(self, component_name: str):
        """Context manager for timing a component"""
        class ComponentTimer:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
                self.start_time = None

            def __enter__(self):
                self.start_time = time.perf_counter()
                return self

            def __exit__(self, *args):
                elapsed = time.perf_counter() - self.start_time
                if self.name not in self.profiler.timings:
                    self.profiler.timings[self.name] = []
                self.profiler.timings[self.name].append(elapsed)

        return ComponentTimer(self, component_name)

    def generate_report(self) -> str:
        """Generate profiling report"""
        report = []
        report.append("=" * 80)
        report.append("AU PIPELINE PROFILING REPORT")
        report.append("=" * 80)
        report.append("")

        # Calculate statistics
        total_time = 0
        component_stats = {}

        for component, times in self.timings.items():
            times_array = np.array(times)
            total = times_array.sum()
            mean = times_array.mean()
            std = times_array.std()

            component_stats[component] = {
                'total': total,
                'mean': mean,
                'std': std,
                'count': len(times)
            }
            total_time += total

        # Sort by total time (descending)
        sorted_components = sorted(
            component_stats.items(),
            key=lambda x: x[1]['total'],
            reverse=True
        )

        # Print summary
        report.append(f"Total Pipeline Time: {total_time:.2f} seconds")
        report.append(f"Frames Processed: {self.frame_count}")
        report.append(f"Average Time per Frame: {total_time / max(self.frame_count, 1):.4f} seconds")
        report.append("")

        # Component breakdown
        report.append("COMPONENT BREAKDOWN:")
        report.append("-" * 80)
        report.append(f"{'Component':<40} {'Total (s)':<12} {'%':<8} {'Per Frame (ms)':<15}")
        report.append("-" * 80)

        for component, stats in sorted_components:
            percentage = (stats['total'] / total_time) * 100
            per_frame_ms = (stats['mean'] * 1000) if stats['count'] > 0 else 0

            report.append(
                f"{component:<40} {stats['total']:>10.4f}  "
                f"{percentage:>6.2f}%  {per_frame_ms:>12.4f}"
            )

        report.append("-" * 80)
        report.append("")

        # Identify optimization opportunities
        report.append("CYTHON OPTIMIZATION OPPORTUNITIES:")
        report.append("-" * 80)

        high_impact_components = [
            (name, stats) for name, stats in sorted_components
            if (stats['total'] / total_time) > 0.10  # >10% of time
        ]

        if high_impact_components:
            for component, stats in high_impact_components:
                percentage = (stats['total'] / total_time) * 100
                per_frame_ms = (stats['mean'] * 1000) if stats['count'] > 0 else 0

                report.append(f"🎯 {component}")
                report.append(f"   Time: {stats['total']:.2f}s ({percentage:.1f}% of total)")
                report.append(f"   Per Frame: {per_frame_ms:.2f}ms")
                report.append(f"   Impact: HIGH - Good candidate for Cython optimization")
                report.append("")
        else:
            report.append("✅ No components taking >10% of time")
            report.append("   Pipeline is already well-optimized!")
            report.append("")

        # Performance metrics
        report.append("PERFORMANCE METRICS:")
        report.append("-" * 80)
        fps = self.frame_count / total_time if total_time > 0 else 0
        report.append(f"Throughput: {fps:.2f} frames/second")
        report.append(f"Video Processing Speed: {fps / 30:.2f}x realtime (assuming 30 fps input)")
        report.append("")

        # Detailed per-component stats
        report.append("DETAILED COMPONENT STATISTICS:")
        report.append("-" * 80)

        for component, stats in sorted_components:
            report.append(f"\n{component}:")
            report.append(f"  Total:   {stats['total']:.4f}s")
            report.append(f"  Mean:    {stats['mean']*1000:.4f}ms")
            report.append(f"  Std Dev: {stats['std']*1000:.4f}ms")
            report.append(f"  Count:   {stats['count']}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


def profile_au_extraction(video_path: str, num_frames: int = 50):
    """
    Profile AU extraction on a video

    Args:
        video_path: Path to test video
        num_frames: Number of frames to process (default: 50 for quick profiling)
    """

    print("=" * 80)
    print("AU PIPELINE PROFILER")
    print("=" * 80)
    print(f"Video: {Path(video_path).name}")
    print(f"Target frames: {num_frames}")
    print("")

    # Configuration
    openface_binary = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
    models_dir = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors"
    pdm_file = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors/In-the-wild_aligned_PDM_68.txt"

    # Initialize profiler
    profiler = PipelineProfiler()

    # Initialize AU predictor
    print("[1/4] Initializing AU predictor...")
    with profiler.profile_component("Initialization"):
        predictor = OpenFace22AUPredictor(
            openface_binary=openface_binary,
            models_dir=models_dir,
            pdm_file=pdm_file
        )
    print("✓ Initialization complete\n")

    # We'll manually process the video to get fine-grained profiling
    import tempfile
    import shutil
    import subprocess
    import cv2
    from openface22_hog_parser import OF22HOGParser

    temp_dir = tempfile.mkdtemp(prefix="profile_au_")

    try:
        # Step 1: Extract features with OpenFace C++ binary
        print("[2/4] Extracting features (OpenFace C++)...")
        with profiler.profile_component("C++ Feature Extraction (OpenFace binary)"):
            video_path_obj = Path(video_path)
            cmd = [
                openface_binary,
                "-f", str(video_path_obj),
                "-out_dir", temp_dir,
                "-hogalign",
                "-pdmparams",
                "-2Dfp",
                "-q"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"OpenFace failed: {result.stderr}")

        video_stem = video_path_obj.stem
        hog_file = Path(temp_dir) / f"{video_stem}.hog"
        csv_file = Path(temp_dir) / f"{video_stem}.csv"
        print(f"✓ Features extracted\n")

        # Step 2: Load and parse features
        print("[3/4] Loading features...")
        with profiler.profile_component("HOG Parsing"):
            hog_parser = OF22HOGParser(str(hog_file))
            frame_indices, hog_features = hog_parser.parse()

        with profiler.profile_component("CSV Loading"):
            csv_data = pd.read_csv(csv_file)

        # Limit to target number of frames
        num_frames = min(num_frames, len(frame_indices), len(csv_data))
        hog_features = hog_features[:num_frames]
        csv_data = csv_data.iloc[:num_frames]
        profiler.frame_count = num_frames

        print(f"✓ Loaded {num_frames} frames\n")

        # Step 3: Process frame by frame with detailed profiling
        print(f"[4/4] Processing {num_frames} frames...")

        # Initialize running median tracker
        try:
            from cython_histogram_median import DualHistogramMedianTrackerCython as DualHistogramMedianTracker
            using_cython = True
        except ImportError:
            from histogram_median_tracker import DualHistogramMedianTracker
            using_cython = False

        print(f"   Running median: {'Cython (optimized)' if using_cython else 'Python'}")

        with profiler.profile_component("Running Median Initialization"):
            median_tracker = DualHistogramMedianTracker(
                hog_dim=4464,
                geom_dim=238,
                hog_bins=1000,
                hog_min=-0.005,
                hog_max=1.0,
                geom_bins=10000,
                geom_min=-60.0,
                geom_max=60.0
            )

        # Pass 1: Build running median
        for i in range(num_frames):
            with profiler.profile_component("Geometric Feature Extraction"):
                pdm_cols = [f'p_{i}' for i in range(34)]
                pdm_params = csv_data.iloc[i][pdm_cols].values
                geom_feat = predictor.pdm_parser.extract_geometric_features(pdm_params)

            hog_feat = hog_features[i].astype(np.float32)
            geom_feat = geom_feat.astype(np.float32)

            with profiler.profile_component("Running Median Update"):
                update_histogram = (i % 2 == 1)
                median_tracker.update(hog_feat, geom_feat, update_histogram=update_histogram)

        # Get final median for Pass 2
        final_median = median_tracker.get_combined_median()

        # Predict AUs
        predictions = {}

        for au_name, model in predictor.models.items():
            is_dynamic = (model['model_type'] == 'dynamic')
            au_predictions = []

            for i in range(num_frames):
                with profiler.profile_component("Feature Preparation"):
                    hog_feat = hog_features[i]

                    pdm_cols = [f'p_{i}' for i in range(34)]
                    pdm_params = csv_data.iloc[i][pdm_cols].values
                    geom_feat = predictor.pdm_parser.extract_geometric_features(pdm_params)

                    full_vector = np.concatenate([hog_feat, geom_feat])

                with profiler.profile_component("SVR Prediction"):
                    if is_dynamic:
                        centered = full_vector - model['means'].flatten() - final_median
                        pred = np.dot(centered.reshape(1, -1), model['support_vectors']) + model['bias']
                        pred = float(pred[0, 0])
                    else:
                        centered = full_vector - model['means'].flatten()
                        pred = np.dot(centered.reshape(1, -1), model['support_vectors']) + model['bias']
                        pred = float(pred[0, 0])

                    pred = np.clip(pred, 0.0, 5.0)

                au_predictions.append(pred)

            au_predictions = np.array(au_predictions)

            # Cutoff adjustment
            with profiler.profile_component("Cutoff Adjustment"):
                if is_dynamic and model.get('cutoff', -1) != -1:
                    cutoff = model['cutoff']
                    sorted_preds = np.sort(au_predictions)
                    cutoff_idx = int(len(sorted_preds) * cutoff)
                    offset = sorted_preds[cutoff_idx]
                    au_predictions = au_predictions - offset
                    au_predictions = np.clip(au_predictions, 0.0, 5.0)

            # Temporal smoothing
            with profiler.profile_component("Temporal Smoothing"):
                smoothed = au_predictions.copy()
                for i in range(1, len(au_predictions) - 1):
                    smoothed[i] = (au_predictions[i-1] + au_predictions[i] + au_predictions[i+1]) / 3

            predictions[au_name] = smoothed

        print(f"✓ Processing complete\n")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Generate report
    report = profiler.generate_report()
    print(report)

    # Save report to file
    report_file = Path("au_pipeline_profiling_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\n✓ Report saved to: {report_file}")

    return profiler


def main():
    """Main profiling script"""

    # Test video path
    video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        print("Please provide a valid video path")
        return

    # Profile the pipeline
    profiler = profile_au_extraction(video_path, num_frames=50)

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review 'au_pipeline_profiling_report.txt'")
    print("2. Identify components marked with 🎯 (>10% of time)")
    print("3. Consider Cython optimization for high-impact components")
    print("4. Expected candidates:")
    print("   - Geometric feature extraction (PDM calculations)")
    print("   - SVR prediction (matrix multiplications)")
    print("   - Feature preparation (array concatenation)")


if __name__ == "__main__":
    main()
