#!/usr/bin/env python3
"""
Accuracy Benchmark Test: Python vs C++ MTCNN
Tests 10 videos (5 Normal + 5 Paralysis) × 3 frames each = 30 total frames
Establishes accuracy baseline before performance optimizations
"""

import cv2
import numpy as np
import subprocess
from pathlib import Path
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2
import json
import time
from datetime import datetime

class AccuracyBenchmark:
    def __init__(self, patient_data_dir):
        self.patient_data_dir = Path(patient_data_dir)
        self.normal_cohort_dir = self.patient_data_dir / "Normal Cohort"
        self.paralysis_cohort_dir = self.patient_data_dir / "Paralysis Cohort"

        # Check for C++ FeatureExtraction tool
        self.cpp_tool = self._find_cpp_tool()

        # Initialize Python detector
        print("Loading Python MTCNN detector...")
        self.python_detector = PurePythonMTCNN_V2()

        self.results = {
            'metadata': {
                'date': datetime.now().isoformat(),
                'total_videos': 0,
                'total_frames': 0,
                'normal_videos': 0,
                'paralysis_videos': 0
            },
            'per_video': {},
            'per_frame': [],
            'summary': {}
        }

    def _find_cpp_tool(self):
        """Find C++ FeatureExtraction tool"""
        possible_paths = [
            "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction",
            "/Users/johnwilsoniv/repo/fea_tool/build/bin/FeatureExtraction",
            "/Users/johnwilsoniv/repo/fea_tool/FeatureExtraction",
            "./FeatureExtraction",
        ]

        for path in possible_paths:
            if Path(path).exists():
                print(f"✓ Found C++ tool: {path}")
                return path

        print("WARNING: C++ FeatureExtraction tool not found!")
        print("Will only run Python MTCNN")
        return None

    def select_videos(self, num_per_cohort=5):
        """Select videos from each cohort"""
        normal_videos = sorted(self.normal_cohort_dir.glob("*.MOV"))[:num_per_cohort]
        paralysis_videos = sorted(self.paralysis_cohort_dir.glob("*.MOV"))[:num_per_cohort]

        print(f"\nSelected {len(normal_videos)} Normal Cohort videos:")
        for v in normal_videos:
            print(f"  - {v.name}")

        print(f"\nSelected {len(paralysis_videos)} Paralysis Cohort videos:")
        for v in paralysis_videos:
            print(f"  - {v.name}")

        self.results['metadata']['normal_videos'] = len(normal_videos)
        self.results['metadata']['paralysis_videos'] = len(paralysis_videos)
        self.results['metadata']['total_videos'] = len(normal_videos) + len(paralysis_videos)

        return normal_videos + paralysis_videos

    def extract_frames(self, video_path, num_frames=3):
        """Extract evenly spaced frames from video"""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"ERROR: Could not open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Select frame indices (beginning, middle, end)
        if num_frames == 3:
            frame_indices = [
                int(total_frames * 0.2),   # 20% into video
                int(total_frames * 0.5),   # 50% (middle)
                int(total_frames * 0.8)    # 80% into video
            ]
        else:
            # Evenly spaced
            frame_indices = [int(total_frames * i / (num_frames + 1)) for i in range(1, num_frames + 1)]

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append({
                    'frame': frame,
                    'index': idx,
                    'timestamp': idx / fps if fps > 0 else 0
                })

        cap.release()

        print(f"  Extracted {len(frames)} frames from {total_frames} total (FPS: {fps:.2f})")
        return frames

    def run_cpp_mtcnn(self, frame, temp_dir):
        """Run C++ OpenFace MTCNN on a single frame"""
        if not self.cpp_tool:
            return None

        # Save frame to temporary file
        temp_frame_path = temp_dir / "temp_frame.jpg"
        cv2.imwrite(str(temp_frame_path), frame)

        # Run C++ FeatureExtraction
        output_dir = temp_dir / "cpp_output"
        output_dir.mkdir(exist_ok=True)

        cmd = [
            self.cpp_tool,
            "-f", str(temp_frame_path),
            "-out_dir", str(output_dir),
            "-no3Dfp",  # Disable 3D face points
            "-nomask",  # Disable masking
            "-noMparams",  # Disable model parameters
            "-noAppearance",  # Disable appearance
            "-noGaze"  # Disable gaze
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            # Parse CSV output for bbox
            csv_file = output_dir / "temp_frame.csv"
            if csv_file.exists():
                import pandas as pd
                df = pd.read_csv(csv_file)

                if len(df) > 0 and df.iloc[0]['success'] == 1:
                    # Extract landmarks to calculate bbox
                    x_cols = [col for col in df.columns if col.startswith('x_')]
                    y_cols = [col for col in df.columns if col.startswith('y_')]

                    xs = df.iloc[0][x_cols].values
                    ys = df.iloc[0][y_cols].values

                    x = np.min(xs)
                    y = np.min(ys)
                    w = np.max(xs) - x
                    h = np.max(ys) - y

                    return {
                        'bbox': (x, y, w, h),
                        'confidence': df.iloc[0]['confidence'],
                        'success': True
                    }

            return {'success': False, 'bbox': None}

        except Exception as e:
            print(f"    C++ MTCNN error: {e}")
            return {'success': False, 'bbox': None, 'error': str(e)}

    def run_python_mtcnn(self, frame):
        """Run Python MTCNN on a single frame"""
        start_time = time.perf_counter()
        bboxes, landmarks = self.python_detector.detect(frame, debug=False)
        latency = (time.perf_counter() - start_time) * 1000  # ms

        if len(bboxes) > 0:
            bbox = bboxes[0]
            return {
                'bbox': (bbox[0], bbox[1], bbox[2], bbox[3]),
                'confidence': bbox[4] if len(bbox) > 4 else 1.0,
                'success': True,
                'latency_ms': latency
            }

        return {
            'success': False,
            'bbox': None,
            'latency_ms': latency
        }

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes (x, y, w, h format)"""
        if box1 is None or box2 is None:
            return 0.0

        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0.0

    def run_benchmark(self):
        """Run full benchmark test"""
        print("\n" + "=" * 80)
        print("ACCURACY BENCHMARK TEST")
        print("Python MTCNN vs C++ OpenFace MTCNN")
        print("=" * 80)

        # Select videos
        videos = self.select_videos(num_per_cohort=5)

        # Create temp directory
        temp_dir = Path("./benchmark_temp")
        temp_dir.mkdir(exist_ok=True)

        total_frames_processed = 0

        # Process each video
        for video_idx, video_path in enumerate(videos):
            cohort = "Normal" if "Normal Cohort" in str(video_path) else "Paralysis"

            print(f"\n{'=' * 80}")
            print(f"Video {video_idx + 1}/{len(videos)}: {video_path.name} ({cohort})")
            print('=' * 80)

            # Extract frames
            frames = self.extract_frames(video_path, num_frames=3)

            if not frames:
                print(f"  WARNING: No frames extracted from {video_path.name}")
                continue

            video_results = {
                'video_name': video_path.name,
                'cohort': cohort,
                'frames': []
            }

            # Process each frame
            for frame_data in frames:
                frame = frame_data['frame']
                frame_idx = frame_data['index']
                timestamp = frame_data['timestamp']

                print(f"\n  Frame {frame_idx} (t={timestamp:.2f}s)")
                print(f"    Frame size: {frame.shape}")

                # Run C++ MTCNN
                print(f"    Running C++ MTCNN...")
                cpp_result = self.run_cpp_mtcnn(frame, temp_dir)

                if cpp_result and cpp_result['success']:
                    cpp_bbox = cpp_result['bbox']
                    print(f"      C++ BBox: x={cpp_bbox[0]:.1f}, y={cpp_bbox[1]:.1f}, w={cpp_bbox[2]:.1f}, h={cpp_bbox[3]:.1f}")
                else:
                    cpp_bbox = None
                    print(f"      C++: No face detected")

                # Run Python MTCNN
                print(f"    Running Python MTCNN...")
                python_result = self.run_python_mtcnn(frame)

                if python_result['success']:
                    python_bbox = python_result['bbox']
                    print(f"      Python BBox: x={python_bbox[0]:.1f}, y={python_bbox[1]:.1f}, w={python_bbox[2]:.1f}, h={python_bbox[3]:.1f}")
                    print(f"      Latency: {python_result['latency_ms']:.2f} ms")
                else:
                    python_bbox = None
                    print(f"      Python: No face detected")
                    print(f"      Latency: {python_result['latency_ms']:.2f} ms")

                # Compare results
                frame_result = {
                    'video_name': video_path.name,
                    'cohort': cohort,
                    'frame_index': frame_idx,
                    'timestamp': timestamp,
                    'cpp_detected': cpp_result['success'] if cpp_result else False,
                    'python_detected': python_result['success'],
                    'cpp_bbox': cpp_bbox,
                    'python_bbox': python_bbox,
                    'python_latency_ms': python_result['latency_ms']
                }

                if cpp_bbox and python_bbox:
                    # Calculate metrics
                    iou = self.calculate_iou(cpp_bbox, python_bbox)
                    dx = abs(cpp_bbox[0] - python_bbox[0])
                    dy = abs(cpp_bbox[1] - python_bbox[1])
                    dw = abs(cpp_bbox[2] - python_bbox[2])
                    dh = abs(cpp_bbox[3] - python_bbox[3])

                    frame_result['iou'] = iou
                    frame_result['position_diff'] = {'dx': dx, 'dy': dy}
                    frame_result['size_diff'] = {'dw': dw, 'dh': dh}

                    print(f"\n    Comparison:")
                    print(f"      IoU: {iou:.4f} ({iou*100:.1f}%)")
                    print(f"      Position diff: dx={dx:.1f}px, dy={dy:.1f}px")
                    print(f"      Size diff: dw={dw:.1f}px, dh={dh:.1f}px")

                video_results['frames'].append(frame_result)
                self.results['per_frame'].append(frame_result)
                total_frames_processed += 1

            self.results['per_video'][video_path.name] = video_results

        self.results['metadata']['total_frames'] = total_frames_processed

        # Calculate summary statistics
        self._calculate_summary()

        # Save results
        self._save_results()

        # Print summary
        self._print_summary()

    def _calculate_summary(self):
        """Calculate summary statistics"""
        frames = self.results['per_frame']

        both_detected = [f for f in frames if f['cpp_detected'] and f['python_detected']]
        only_cpp = [f for f in frames if f['cpp_detected'] and not f['python_detected']]
        only_python = [f for f in frames if f['python_detected'] and not f['cpp_detected']]
        neither = [f for f in frames if not f['cpp_detected'] and not f['python_detected']]

        self.results['summary'] = {
            'total_frames': len(frames),
            'both_detected': len(both_detected),
            'only_cpp_detected': len(only_cpp),
            'only_python_detected': len(only_python),
            'neither_detected': len(neither),
            'detection_agreement_rate': len(both_detected) / len(frames) if frames else 0
        }

        # Calculate accuracy metrics for frames where both detected
        if both_detected:
            ious = [f['iou'] for f in both_detected]
            dxs = [f['position_diff']['dx'] for f in both_detected]
            dys = [f['position_diff']['dy'] for f in both_detected]
            dws = [f['size_diff']['dw'] for f in both_detected]
            dhs = [f['size_diff']['dh'] for f in both_detected]

            self.results['summary']['accuracy_metrics'] = {
                'iou': {
                    'mean': float(np.mean(ious)),
                    'median': float(np.median(ious)),
                    'std': float(np.std(ious)),
                    'min': float(np.min(ious)),
                    'max': float(np.max(ious)),
                    'gt_99_percent': sum(1 for iou in ious if iou > 0.99),
                    'gt_95_percent': sum(1 for iou in ious if iou > 0.95),
                    'gt_90_percent': sum(1 for iou in ious if iou > 0.90)
                },
                'position_diff': {
                    'mean_dx': float(np.mean(dxs)),
                    'mean_dy': float(np.mean(dys)),
                    'median_dx': float(np.median(dxs)),
                    'median_dy': float(np.median(dys)),
                    'max_dx': float(np.max(dxs)),
                    'max_dy': float(np.max(dys))
                },
                'size_diff': {
                    'mean_dw': float(np.mean(dws)),
                    'mean_dh': float(np.mean(dhs)),
                    'median_dw': float(np.median(dws)),
                    'median_dh': float(np.median(dhs)),
                    'max_dw': float(np.max(dws)),
                    'max_dh': float(np.max(dhs))
                }
            }

        # Calculate performance metrics
        latencies = [f['python_latency_ms'] for f in frames if 'python_latency_ms' in f]
        if latencies:
            self.results['summary']['performance_metrics'] = {
                'python_mtcnn': {
                    'mean_latency_ms': float(np.mean(latencies)),
                    'median_latency_ms': float(np.median(latencies)),
                    'p95_latency_ms': float(np.percentile(latencies, 95)),
                    'p99_latency_ms': float(np.percentile(latencies, 99)),
                    'min_latency_ms': float(np.min(latencies)),
                    'max_latency_ms': float(np.max(latencies)),
                    'mean_fps': 1000 / np.mean(latencies) if np.mean(latencies) > 0 else 0
                }
            }

    def _save_results(self):
        """Save results to JSON file"""
        output_file = "accuracy_benchmark_results.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")

    def _print_summary(self):
        """Print summary statistics"""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        summary = self.results['summary']

        print(f"\nDataset:")
        print(f"  Total videos:     {self.results['metadata']['total_videos']}")
        print(f"  Normal cohort:    {self.results['metadata']['normal_videos']}")
        print(f"  Paralysis cohort: {self.results['metadata']['paralysis_videos']}")
        print(f"  Total frames:     {summary['total_frames']}")

        print(f"\nDetection Agreement:")
        print(f"  Both detected:         {summary['both_detected']:3d} ({summary['both_detected']/summary['total_frames']*100:.1f}%)")
        print(f"  Only C++ detected:     {summary['only_cpp_detected']:3d} ({summary['only_cpp_detected']/summary['total_frames']*100:.1f}%)")
        print(f"  Only Python detected:  {summary['only_python_detected']:3d} ({summary['only_python_detected']/summary['total_frames']*100:.1f}%)")
        print(f"  Neither detected:      {summary['neither_detected']:3d} ({summary['neither_detected']/summary['total_frames']*100:.1f}%)")

        if 'accuracy_metrics' in summary:
            acc = summary['accuracy_metrics']

            print(f"\nAccuracy Metrics (for {summary['both_detected']} frames where both detected):")
            print(f"  Mean IoU:       {acc['iou']['mean']:.4f} ({acc['iou']['mean']*100:.2f}%)")
            print(f"  Median IoU:     {acc['iou']['median']:.4f}")
            print(f"  IoU std:        {acc['iou']['std']:.4f}")
            print(f"  IoU range:      [{acc['iou']['min']:.4f}, {acc['iou']['max']:.4f}]")
            print(f"  IoU > 99%:      {acc['iou']['gt_99_percent']}/{summary['both_detected']} ({acc['iou']['gt_99_percent']/summary['both_detected']*100:.1f}%)")
            print(f"  IoU > 95%:      {acc['iou']['gt_95_percent']}/{summary['both_detected']} ({acc['iou']['gt_95_percent']/summary['both_detected']*100:.1f}%)")
            print(f"  IoU > 90%:      {acc['iou']['gt_90_percent']}/{summary['both_detected']} ({acc['iou']['gt_90_percent']/summary['both_detected']*100:.1f}%)")

            print(f"\nPosition Differences:")
            print(f"  Mean:   dx={acc['position_diff']['mean_dx']:.2f}px, dy={acc['position_diff']['mean_dy']:.2f}px")
            print(f"  Median: dx={acc['position_diff']['median_dx']:.2f}px, dy={acc['position_diff']['median_dy']:.2f}px")
            print(f"  Max:    dx={acc['position_diff']['max_dx']:.2f}px, dy={acc['position_diff']['max_dy']:.2f}px")

            print(f"\nSize Differences:")
            print(f"  Mean:   dw={acc['size_diff']['mean_dw']:.2f}px, dh={acc['size_diff']['mean_dh']:.2f}px")
            print(f"  Median: dw={acc['size_diff']['median_dw']:.2f}px, dh={acc['size_diff']['median_dh']:.2f}px")
            print(f"  Max:    dw={acc['size_diff']['max_dw']:.2f}px, dh={acc['size_diff']['max_dh']:.2f}px")

        if 'performance_metrics' in summary:
            perf = summary['performance_metrics']['python_mtcnn']

            print(f"\nPython MTCNN Performance (Current Baseline):")
            print(f"  Mean latency:   {perf['mean_latency_ms']:.2f} ms")
            print(f"  Median latency: {perf['median_latency_ms']:.2f} ms")
            print(f"  P95 latency:    {perf['p95_latency_ms']:.2f} ms")
            print(f"  P99 latency:    {perf['p99_latency_ms']:.2f} ms")
            print(f"  Mean FPS:       {perf['mean_fps']:.2f}")
            print(f"\n  ⚠️  TARGET: 30 FPS (33.33 ms per frame)")
            print(f"      Current: {perf['mean_fps']:.2f} FPS ({perf['mean_latency_ms']:.2f} ms)")

            if perf['mean_fps'] >= 30:
                print(f"      ✅ Already meets target!")
            else:
                speedup_needed = 30 / perf['mean_fps']
                print(f"      ❌ Need {speedup_needed:.1f}x speedup to reach 30 FPS")

        print("\n" + "=" * 80)
        print("This baseline will be used to measure optimization improvements.")
        print("=" * 80)


def main():
    patient_data_dir = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data"

    benchmark = AccuracyBenchmark(patient_data_dir)
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
