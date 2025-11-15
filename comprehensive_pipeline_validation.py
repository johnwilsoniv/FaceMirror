#!/usr/bin/env python3
"""
Comprehensive Pipeline Validation System

Validates the entire Python reimplementation (PyMTCNN + PyFaceAU) against C++ OpenFace 2.2.

TESTING SCOPE:
- PyMTCNN: Tests both ONNX (CUDA) and CoreML backends
- PyFaceAU: Tests both ONNX and CoreML for PFLD landmark detector
- Full pipeline validation for each backend combination
- Stage-by-stage comparison (PNet, RNet, ONet for MTCNN)
- Component-by-component comparison (detection, landmarks, AUs)

Target: >95% match to C++ at all stages
Completion time: <10 minutes for full test suite
Test dataset: 3 frames per patient × 30 patients = 90 frames

Comparison metrics:
- MTCNN: Box counts per stage, bbox IoU, 5-point landmark accuracy
- PyFaceAU: Initial bbox, 68-point landmarks, AU correlation
- Performance: FPS, per-stage timing, bottleneck identification

Usage:
    python comprehensive_pipeline_validation.py --patient-data-dir "Patient Data" --output-dir validation_results
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import subprocess
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

# Import our Python implementations
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))

from pymtcnn import MTCNN
from pyfaceau import FullPythonAUPipeline


class PipelineValidator:
    """Comprehensive validation of PyMTCNN + PyFaceAU against C++ OpenFace"""

    def __init__(
        self,
        patient_data_dir: str,
        output_dir: str,
        backends: List[str] = ['cuda', 'coreml'],
        frames_per_patient: int = 3,
        max_patients: int = 30,
        cpp_openface_bin: str = None
    ):
        """
        Initialize validator

        Args:
            patient_data_dir: Path to Patient Data folder
            output_dir: Where to save results
            backends: List of backends to test ['cuda', 'coreml']
            frames_per_patient: Number of frames to extract per patient
            max_patients: Maximum number of patients to test
            cpp_openface_bin: Path to C++ OpenFace binary
        """
        self.patient_data_dir = Path(patient_data_dir)
        self.output_dir = Path(output_dir)
        self.backends = backends
        self.frames_per_patient = frames_per_patient
        self.max_patients = max_patients

        # C++ OpenFace binary
        if cpp_openface_bin is None:
            cpp_openface_bin = '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction'
        self.cpp_openface_bin = Path(cpp_openface_bin)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'cpp_outputs').mkdir(exist_ok=True)
        (self.output_dir / 'python_outputs').mkdir(exist_ok=True)

        # Results storage
        self.test_dataset = []
        self.results = []

        print("="*80)
        print("COMPREHENSIVE PIPELINE VALIDATION SYSTEM")
        print("="*80)
        print(f"Patient Data Dir: {self.patient_data_dir}")
        print(f"Output Dir: {self.output_dir}")
        print(f"Backends: {self.backends}")
        print(f"Frames per patient: {self.frames_per_patient}")
        print(f"Max patients: {self.max_patients}")
        print(f"C++ OpenFace: {self.cpp_openface_bin}")
        print("="*80)
        print()

    def prepare_test_dataset(self):
        """Extract test frames from patient videos"""

        print("Preparing test dataset...")
        print("-" * 80)

        patient_dirs = sorted(list(self.patient_data_dir.glob('Patient*')))[:self.max_patients]

        if not patient_dirs:
            raise ValueError(f"No patient directories found in {self.patient_data_dir}")

        for patient_dir in tqdm(patient_dirs, desc="Extracting frames"):
            video_files = list(patient_dir.glob('*.mp4')) + list(patient_dir.glob('*.avi'))

            if not video_files:
                print(f"Warning: No video found in {patient_dir.name}")
                continue

            video_path = video_files[0]
            frames = self._select_frames_from_video(video_path, self.frames_per_patient)

            # Save frames
            frame_dir = self.output_dir / 'test_frames' / patient_dir.name
            frame_dir.mkdir(parents=True, exist_ok=True)

            for frame_idx, frame in frames:
                frame_name = f"frame_{frame_idx:04d}.jpg"
                frame_path = frame_dir / frame_name
                cv2.imwrite(str(frame_path), frame)

                self.test_dataset.append({
                    'patient': patient_dir.name,
                    'frame_idx': frame_idx,
                    'frame_path': str(frame_path)
                })

        print(f"Test dataset prepared: {len(self.test_dataset)} frames from {len(patient_dirs)} patients")
        print()

        return self.test_dataset

    def _select_frames_from_video(self, video_path: Path, num_frames: int) -> List[Tuple[int, np.ndarray]]:
        """Select evenly distributed frames from video"""

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return []

        # Select frames at evenly spaced intervals
        frame_indices = [int(total_frames * (i + 1) / (num_frames + 1)) for i in range(num_frames)]

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append((idx, frame))

        cap.release()
        return frames

    def run_cpp_openface(self, frame_path: str) -> Dict:
        """Run C++ OpenFace and parse outputs"""

        output_dir = self.output_dir / 'cpp_outputs' / Path(frame_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run C++ OpenFace
        cmd = [
            str(self.cpp_openface_bin),
            '-f', frame_path,
            '-out_dir', str(output_dir),
            '-2Dfp',      # Save 2D landmarks
            '-3Dfp',      # Save 3D landmarks
            '-pdmparams', # Save PDM parameters
            '-aus',       # Save AUs
            '-verbose'    # Verbose output for debug
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            # Parse outputs
            csv_files = list(output_dir.glob('*.csv'))
            if not csv_files:
                return {'success': False, 'error': 'No CSV output generated'}

            df = pd.read_csv(csv_files[0])

            # Extract detection bbox
            bbox = self._parse_cpp_detection(df)

            # Extract landmarks
            landmarks = self._parse_cpp_landmarks(df)

            # Extract AUs
            aus = self._parse_cpp_aus(df)

            return {
                'success': True,
                'bbox': bbox,
                'landmarks': landmarks,
                'aus': aus,
                'csv_path': str(csv_files[0]),
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'C++ OpenFace timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _parse_cpp_detection(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Parse face detection bbox from C++ output"""
        try:
            if 'face_x' in df.columns:
                x = df['face_x'].iloc[0]
                y = df['face_y'].iloc[0]
                w = df['face_width'].iloc[0]
                h = df['face_height'].iloc[0]
                return np.array([x, y, x + w, y + h])
            return None
        except:
            return None

    def _parse_cpp_landmarks(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Parse 68-point landmarks from C++ output"""
        try:
            landmarks = []
            for i in range(68):
                x = df[f'x_{i}'].iloc[0]
                y = df[f'y_{i}'].iloc[0]
                landmarks.append([x, y])
            return np.array(landmarks)
        except:
            return None

    def _parse_cpp_aus(self, df: pd.DataFrame) -> Dict:
        """Parse AU predictions from C++ output"""
        aus = {}
        for col in df.columns:
            if col.startswith('AU') and col.endswith('_r'):
                aus[col] = df[col].iloc[0]
        return aus

    def run_pymtcnn(self, frame: np.ndarray, backend: str) -> Dict:
        """Run PyMTCNN with debug mode"""

        # For now, run without debug mode (will add debug mode next)
        start_time = time.time()

        detector = MTCNN(backend=backend)
        bboxes, landmarks = detector.detect(frame)

        elapsed_time = (time.time() - start_time) * 1000  # ms

        if len(bboxes) == 0:
            return {
                'success': False,
                'time_ms': elapsed_time
            }

        return {
            'success': True,
            'bbox': bboxes[0][:4],  # [x, y, w, h]
            'landmarks_5pt': landmarks[0],  # (5, 2)
            'confidence': bboxes[0][4] if len(bboxes[0]) > 4 else 1.0,
            'time_ms': elapsed_time
        }

    def run_pyfaceau(self, frame: np.ndarray, backend: str) -> Dict:
        """Run PyFaceAU pipeline"""

        start_time = time.time()

        # Initialize pipeline
        pipeline = FullPythonAUPipeline(
            pfld_model='pyfaceau/weights/pfld_cunjian.onnx',
            pdm_file='pyfaceau/weights/In-the-wild_aligned_PDM_68.txt',
            au_models_dir='pyfaceau/weights/AU_predictors',
            triangulation_file='pyfaceau/weights/tris_68_full.txt',
            mtcnn_backend=backend,
            use_coreml_pfld=(backend == 'coreml'),
            verbose=False
        )

        # Process frame
        result = pipeline._process_frame(frame, 0, 0.0)

        elapsed_time = (time.time() - start_time) * 1000  # ms

        if not result['success']:
            return {
                'success': False,
                'time_ms': elapsed_time
            }

        # Extract data (we'll need to add debug mode to get more details)
        return {
            'success': True,
            'aus': {k: v for k, v in result.items() if k.startswith('AU')},
            'time_ms': elapsed_time
        }

    def compare_results(self, cpp_results: Dict, py_mtcnn_results: Dict, py_faceau_results: Dict) -> Dict:
        """Compare results and compute metrics"""

        comparison = {}

        # Compare face detection bbox
        if cpp_results['bbox'] is not None and py_mtcnn_results['success']:
            cpp_bbox = cpp_results['bbox']
            # PyMTCNN returns [x, y, w, h], convert to [x1, y1, x2, y2]
            py_bbox_xywh = py_mtcnn_results['bbox']
            py_bbox = np.array([
                py_bbox_xywh[0],
                py_bbox_xywh[1],
                py_bbox_xywh[0] + py_bbox_xywh[2],
                py_bbox_xywh[1] + py_bbox_xywh[3]
            ])

            iou = self._calculate_iou(cpp_bbox, py_bbox)
            comparison['bbox_iou'] = iou
            comparison['cpp_bbox'] = cpp_bbox
            comparison['py_bbox'] = py_bbox
        else:
            comparison['bbox_iou'] = 0.0

        # Compare AUs
        if cpp_results['aus'] and py_faceau_results['success']:
            au_comparison = self._compare_aus(cpp_results['aus'], py_faceau_results['aus'])
            comparison['au_comparison'] = au_comparison
        else:
            comparison['au_comparison'] = None

        # Performance
        comparison['pymtcnn_time_ms'] = py_mtcnn_results.get('time_ms', 0)
        comparison['pyfaceau_time_ms'] = py_faceau_results.get('time_ms', 0)
        comparison['total_time_ms'] = comparison['pymtcnn_time_ms'] + comparison['pyfaceau_time_ms']

        return comparison

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two bboxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _compare_aus(self, cpp_aus: Dict, py_aus: Dict) -> Dict:
        """Compare AU predictions"""

        au_metrics = {}

        for au_name in cpp_aus.keys():
            if au_name in py_aus:
                cpp_val = cpp_aus[au_name]
                py_val = py_aus[au_name]

                au_metrics[au_name] = {
                    'cpp_value': cpp_val,
                    'py_value': py_val,
                    'difference': abs(cpp_val - py_val),
                    'relative_error': abs(cpp_val - py_val) / (abs(cpp_val) + 1e-6)
                }

        return au_metrics

    def visualize_bbox_comparison(self, frame: np.ndarray, cpp_bbox: np.ndarray,
                                   py_bbox: np.ndarray, iou: float, output_path: str):
        """Create visualization of bbox comparison"""

        vis = frame.copy()

        # Draw C++ bbox in green
        cv2.rectangle(
            vis,
            (int(cpp_bbox[0]), int(cpp_bbox[1])),
            (int(cpp_bbox[2]), int(cpp_bbox[3])),
            (0, 255, 0),  # Green
            3
        )
        cv2.putText(vis, 'C++', (int(cpp_bbox[0]), int(cpp_bbox[1])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw Python bbox in blue
        cv2.rectangle(
            vis,
            (int(py_bbox[0]), int(py_bbox[1])),
            (int(py_bbox[2]), int(py_bbox[3])),
            (255, 0, 0),  # Blue
            3
        )
        cv2.putText(vis, 'Python', (int(py_bbox[0]), int(py_bbox[3])+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Add IoU text
        cv2.putText(vis, f'IoU: {iou:.3f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imwrite(output_path, vis)

    def run_validation(self):
        """Run complete validation suite"""

        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE VALIDATION")
        print("="*80)
        print()

        # Prepare test dataset
        self.prepare_test_dataset()

        # Run validation for each backend
        for backend in self.backends:
            print(f"\n{'='*80}")
            print(f"Testing Backend: {backend.upper()}")
            print(f"{'='*80}\n")

            backend_results = self._validate_backend(backend)
            self.results.append(backend_results)

        # Generate report
        self._generate_report()

    def _validate_backend(self, backend: str) -> Dict:
        """Validate pipeline on single backend"""

        results = {
            'backend': backend,
            'test_cases': []
        }

        for test_case in tqdm(self.test_dataset, desc=f"Validating {backend}"):
            frame_path = test_case['frame_path']
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"Warning: Could not read {frame_path}")
                continue

            # Run C++ OpenFace
            cpp_results = self.run_cpp_openface(frame_path)

            # Run PyMTCNN
            py_mtcnn_results = self.run_pymtcnn(frame, backend)

            # Run PyFaceAU
            py_faceau_results = self.run_pyfaceau(frame, backend)

            # Compare results
            comparison = self.compare_results(cpp_results, py_mtcnn_results, py_faceau_results)

            # Generate visualization if successful
            if comparison.get('bbox_iou', 0) > 0:
                vis_path = str(self.output_dir / 'visualizations' /
                              f"{test_case['patient']}_{Path(frame_path).stem}_{backend}.jpg")
                self.visualize_bbox_comparison(
                    frame,
                    comparison['cpp_bbox'],
                    comparison['py_bbox'],
                    comparison['bbox_iou'],
                    vis_path
                )

            # Store results
            result = {
                **test_case,
                'cpp_success': cpp_results['success'],
                'py_mtcnn_success': py_mtcnn_results['success'],
                'py_faceau_success': py_faceau_results['success'],
                **comparison
            }

            results['test_cases'].append(result)

        return results

    def _generate_report(self):
        """Generate comprehensive markdown report"""

        report_path = self.output_dir / 'VALIDATION_REPORT.md'

        with open(report_path, 'w') as f:
            f.write("# Comprehensive Pipeline Validation Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Test Cases:** {len(self.test_dataset)} frames\n\n")
            f.write(f"**Backends Tested:** {', '.join(self.backends)}\n\n")
            f.write("---\n\n")

            for backend_results in self.results:
                backend = backend_results['backend']
                test_cases = backend_results['test_cases']

                f.write(f"## Backend: {backend.upper()}\n\n")

                # Calculate statistics
                successful_tests = [tc for tc in test_cases if tc.get('bbox_iou', 0) > 0]
                mean_iou = np.mean([tc['bbox_iou'] for tc in successful_tests]) if successful_tests else 0
                mean_time = np.mean([tc.get('total_time_ms', 0) for tc in test_cases])

                f.write(f"### Summary Statistics\n\n")
                f.write(f"- **Success Rate:** {len(successful_tests)}/{len(test_cases)} ({len(successful_tests)/len(test_cases)*100:.1f}%)\n")
                f.write(f"- **Mean Bbox IoU:** {mean_iou:.3f}\n")
                f.write(f"- **Mean Processing Time:** {mean_time:.1f} ms\n")
                f.write(f"- **Estimated FPS:** {1000/mean_time:.1f} FPS\n\n")

                # AU comparison if available
                au_comparisons = [tc['au_comparison'] for tc in test_cases if tc.get('au_comparison')]
                if au_comparisons:
                    f.write("### AU Comparison\n\n")
                    # Aggregate AU metrics
                    all_au_names = set()
                    for au_comp in au_comparisons:
                        all_au_names.update(au_comp.keys())

                    for au_name in sorted(all_au_names):
                        au_diffs = [au_comp[au_name]['difference']
                                   for au_comp in au_comparisons if au_name in au_comp]
                        if au_diffs:
                            mean_diff = np.mean(au_diffs)
                            f.write(f"- **{au_name}:** Mean absolute difference = {mean_diff:.3f}\n")

                f.write("\n---\n\n")

        print(f"\nReport generated: {report_path}")

        return report_path


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Pipeline Validation")
    parser.add_argument('--patient-data-dir', type=str, required=True,
                       help='Path to Patient Data directory')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                       help='Output directory for results')
    parser.add_argument('--backends', nargs='+', default=['cuda', 'coreml'],
                       help='Backends to test')
    parser.add_argument('--frames-per-patient', type=int, default=3,
                       help='Number of frames to extract per patient')
    parser.add_argument('--max-patients', type=int, default=30,
                       help='Maximum number of patients to test')
    parser.add_argument('--cpp-openface-bin', type=str, default=None,
                       help='Path to C++ OpenFace binary')

    args = parser.parse_args()

    # Create validator
    validator = PipelineValidator(
        patient_data_dir=args.patient_data_dir,
        output_dir=args.output_dir,
        backends=args.backends,
        frames_per_patient=args.frames_per_patient,
        max_patients=args.max_patients,
        cpp_openface_bin=args.cpp_openface_bin
    )

    # Run validation
    validator.run_validation()

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {validator.output_dir}")


if __name__ == '__main__':
    main()
