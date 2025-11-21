#!/usr/bin/env python3
"""
Simple benchmark script for OpenFace C++ pipeline.
Focuses on measuring FPS and accuracy on the Shorty.mov video.
"""

import subprocess
import time
import pandas as pd
import numpy as np
from pathlib import Path

def run_openface_benchmark(video_path: str):
    """Run OpenFace on video and measure performance."""

    print("="*80)
    print("OPENFACE C++ PIPELINE BENCHMARK")
    print("="*80)
    print(f"\nTest video: {video_path}")

    # Find OpenFace executable
    openface_path = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    if not Path(openface_path).exists():
        print(f"Error: OpenFace not found at {openface_path}")
        return

    output_dir = Path("openface_benchmark_output")
    output_dir.mkdir(exist_ok=True)

    # Run OpenFace with full AU extraction
    cmd = [
        openface_path,
        "-f", video_path,
        "-out_dir", str(output_dir),
        "-aus",  # Action Units
        "-2Dfp",  # 2D landmarks
        "-3Dfp",  # 3D landmarks
        "-pose",  # Head pose
        "-gaze",  # Gaze
        "-verbose"
    ]

    print(f"\nRunning command:")
    print(" ".join(cmd))
    print("\nProcessing...")

    # Measure execution time
    start_time = time.perf_counter()

    result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed_time = time.perf_counter() - start_time

    # Parse results
    video_name = Path(video_path).stem
    csv_path = output_dir / f"{video_name}.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        n_frames = len(df)

        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)

        # Performance metrics
        fps = n_frames / elapsed_time if elapsed_time > 0 else 0
        ms_per_frame = (elapsed_time * 1000) / n_frames if n_frames > 0 else 0

        print(f"\nPerformance Metrics:")
        print(f"  Total processing time: {elapsed_time:.2f} seconds")
        print(f"  Frames processed: {n_frames}")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Average time per frame: {ms_per_frame:.1f} ms")

        # AU statistics
        au_columns = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]

        if au_columns:
            print(f"\nAction Unit Statistics:")
            print(f"  Number of AUs tracked: {len(au_columns)}")
            print(f"  AU columns: {', '.join(sorted(au_columns)[:10])}...")

            # Check AU activations
            print(f"\nAU Activation Summary:")
            for au in sorted(au_columns)[:10]:  # First 10 AUs
                au_values = df[au]
                activation_rate = (au_values > 0.5).mean() * 100
                mean_intensity = au_values.mean()
                max_intensity = au_values.max()

                print(f"  {au:6s}: {activation_rate:5.1f}% activated, "
                      f"mean={mean_intensity:.2f}, max={max_intensity:.2f}")

        # Landmark tracking quality
        confidence_col = 'confidence' if 'confidence' in df.columns else 'success'
        if confidence_col in df.columns:
            confidence = df[confidence_col]
            print(f"\nTracking Quality:")
            print(f"  Average confidence: {confidence.mean():.3f}")
            print(f"  Min confidence: {confidence.min():.3f}")
            print(f"  Max confidence: {confidence.max():.3f}")
            print(f"  Frames with confidence > 0.9: {(confidence > 0.9).sum()} ({(confidence > 0.9).mean()*100:.1f}%)")

        # Save summary
        summary_path = output_dir / "benchmark_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"OpenFace Benchmark Results\n")
            f.write(f"==========================\n\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Total processing time: {elapsed_time:.2f} seconds\n")
            f.write(f"Frames processed: {n_frames}\n")
            f.write(f"Average FPS: {fps:.1f}\n")
            f.write(f"Average time per frame: {ms_per_frame:.1f} ms\n")

            # Calculate theoretical real-time capability
            video_fps = 30  # Assuming 30 fps video
            real_time_factor = fps / video_fps
            f.write(f"\nReal-time capability:\n")
            f.write(f"  Video FPS: {video_fps}\n")
            f.write(f"  Processing FPS: {fps:.1f}\n")
            f.write(f"  Real-time factor: {real_time_factor:.2f}x\n")
            if real_time_factor >= 1.0:
                f.write(f"  ✅ Can process in real-time\n")
            else:
                f.write(f"  ❌ Cannot process in real-time (needs {1/real_time_factor:.1f}x speed improvement)\n")

        print(f"\nSummary saved to: {summary_path}")

        return {
            'fps': fps,
            'ms_per_frame': ms_per_frame,
            'n_frames': n_frames,
            'elapsed_time': elapsed_time,
            'dataframe': df
        }
    else:
        print(f"\nError: OpenFace output not found at {csv_path}")
        print(f"Stderr: {result.stderr}")
        return None


def main():
    """Run benchmark on Shorty.mov"""

    video_path = "Patient Data/Normal Cohort/Shorty.mov"

    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return

    results = run_openface_benchmark(video_path)

    if results:
        print("\n" + "="*80)
        print("BENCHMARK COMPLETE")
        print("="*80)
        print(f"\n🎯 Key Result: OpenFace achieved {results['fps']:.1f} FPS on {results['n_frames']} frames")

        # Compare to target performance
        target_fps = 30  # Real-time target
        if results['fps'] >= target_fps:
            print(f"✅ Meets real-time target ({target_fps} FPS)")
        else:
            gap = target_fps - results['fps']
            print(f"⚠️  Below real-time target by {gap:.1f} FPS")
            print(f"   Need {target_fps/results['fps']:.1f}x speedup for real-time")


if __name__ == "__main__":
    main()