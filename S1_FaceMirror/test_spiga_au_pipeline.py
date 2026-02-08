#!/usr/bin/env python3
"""
Test SPIGA-based AU pipeline vs CLNF baseline.

Compares:
1. Full CLNF pipeline (baseline) - CLNF landmarks + CLNF params_local
2. SPIGA + constrained geometric features - SPIGA landmarks + constrained params_local

Hypothesis: SPIGA alignment preserves accuracy in HOG features, and constrained
geometric features are "good enough" for valid AU predictions.
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent / 'pyfaceau'))
sys.path.insert(0, str(script_dir.parent / 'pyclnf'))

from config import apply_environment_settings
apply_environment_settings()

import numpy as np
import cv2
import argparse
import json
from typing import Optional, Dict
import time


def get_video_rotation(video_path: str) -> int:
    """Get rotation metadata from video using ffprobe."""
    import subprocess
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-select_streams', 'v:0',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)

        for stream in data.get('streams', []):
            for side_data in stream.get('side_data_list', []):
                if side_data.get('side_data_type') == 'Display Matrix':
                    if 'rotation' in side_data:
                        return int(side_data['rotation'])
            rotation = stream.get('tags', {}).get('rotate', 0)
            if rotation:
                return int(rotation)
    except Exception as e:
        print(f"Warning: Could not get video rotation: {e}")
    return 0


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate frame based on rotation metadata."""
    if rotation == 90 or rotation == -270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == -90 or rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180 or rotation == -180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


class CLNFAUPipeline:
    """Full CLNF-based AU pipeline (baseline)."""

    def __init__(self, weights_dir: Path):
        from pymtcnn.backends.onnx_backend import ONNXMTCNN
        from pyclnf import CLNF
        from pyfaceau.alignment.face_aligner import OpenFace22FaceAligner
        from pyfaceau.features.pdm import PDMParser
        import pyfhog

        self.pyfhog = pyfhog

        self.face_detector = ONNXMTCNN()
        self.landmark_detector = CLNF(
            detector=None,
            use_gpu=False,
            convergence_profile='video'
        )
        self.landmark_detector.optimizer.use_cpp_warp = False

        pdm_file = str(weights_dir / 'In-the-wild_aligned_PDM_68.txt')
        self.face_aligner = OpenFace22FaceAligner(
            pdm_file=pdm_file,
            sim_scale=0.7,
            output_size=(112, 112)
        )

        self.pdm_parser = PDMParser(pdm_file)

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Dict]:
        """Process a single frame and return features."""
        # Detect face
        bboxes, _ = self.face_detector.detect(frame_bgr)
        if bboxes is None or len(bboxes) == 0:
            return None

        x, y, w, h = bboxes[0][:4]
        bbox_xywh = (x, y, w, h)

        # Get landmarks with params
        landmarks, info = self.landmark_detector.fit(frame_bgr, bbox_xywh, return_params=True)
        if landmarks is None:
            return None

        # Extract params from CLNF optimization
        if 'params' not in info:
            return None

        clnf_params = info['params']
        params_global = clnf_params[:6]
        params_local = clnf_params[6:]

        # Extract pose for alignment
        pose_tx = params_global[4]
        pose_ty = params_global[5]
        pose_rz = params_global[3]

        # Align face
        aligned_face = self.face_aligner.align_face(
            frame_bgr, landmarks, pose_tx, pose_ty, p_rz=pose_rz
        )

        # Extract HOG features
        hog_features = self.pyfhog.extract_fhog_features(aligned_face, cell_size=8)

        # Extract geometric features
        geom_features = self.pdm_parser.extract_geometric_features(params_local)

        return {
            'landmarks': landmarks,
            'params_global': params_global,
            'params_local': params_local,
            'aligned_face': aligned_face,
            'hog_features': hog_features,
            'geom_features': geom_features
        }


class SPIGAAUPipeline:
    """SPIGA-based AU pipeline with Procrustes-corrected alignment and regularized geometric features."""

    # Same rigid indices used by face_aligner's Kabsch alignment
    RIGID_INDICES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]

    # SPIGA-to-CLNF alignment calibration constants
    # Measured from 50 frames across 5 patients (3 paralysis + 2 normal)
    # Scale: corrects SPIGA rigid points being more tightly clustered than CLNF
    # Offset: corrects SPIGA landmark centroid shift vs CLNF CalcParams center
    SPIGA_SCALE_CORRECTION = 1.0346
    SPIGA_OFFSET_X = 0.22
    SPIGA_OFFSET_Y = 2.37

    def __init__(self, weights_dir: Path, reg_factor: float = 10.0):
        from spiga_detector import SPIGALandmarkDetector
        from pyfaceau.alignment.face_aligner import OpenFace22FaceAligner
        from pyfaceau.alignment.calc_params import CalcParams
        from pyfaceau.features.pdm import PDMParser
        import pyfhog

        self.pyfhog = pyfhog

        self.spiga = SPIGALandmarkDetector(debug_mode=False)

        pdm_file = str(weights_dir / 'In-the-wild_aligned_PDM_68.txt')
        self.face_aligner = OpenFace22FaceAligner(
            pdm_file=pdm_file,
            sim_scale=0.7,
            output_size=(112, 112)
        )

        self.pdm_parser = PDMParser(pdm_file)
        self.calc_params = CalcParams(self.pdm_parser, reg_factor=reg_factor)

    def process_frame(self, frame_bgr: np.ndarray, use_zero_geom: bool = False) -> Optional[Dict]:
        """Process a single frame and return features."""
        # SPIGA detection (raw landmarks)
        landmarks, spiga_info = self.spiga.get_face_mesh(frame_bgr, detection_interval=0)
        if landmarks is None:
            return None

        # Procrustes-correct: map PDM mean shape to SPIGA face position using rigid points.
        # Gives landmarks with PDM proportions (stable Kabsch) at detected position (no drift).
        landmarks_for_alignment = self.calc_params.procrustes_correct(
            landmarks, self.RIGID_INDICES
        )

        # Apply scale calibration: expand Procrustes-corrected landmarks to match CLNF's
        # RMS spread. SPIGA rigid points are ~3.5% more tightly clustered than CLNF's,
        # causing the Kabsch alignment to over-zoom.
        centroid = np.mean(landmarks_for_alignment, axis=0)
        landmarks_for_alignment = (landmarks_for_alignment - centroid) * self.SPIGA_SCALE_CORRECTION + centroid

        # Use raw landmark centroid for alignment centering, with calibrated offset
        pose_tx = float(np.mean(landmarks[:, 0])) + self.SPIGA_OFFSET_X
        pose_ty = float(np.mean(landmarks[:, 1])) + self.SPIGA_OFFSET_Y

        # Align face using Procrustes-corrected landmarks
        aligned_face = self.face_aligner.align_face(
            frame_bgr, landmarks_for_alignment, pose_tx, pose_ty
        )

        # Run CalcParams on raw SPIGA landmarks for geometric features
        params_global, params_local = self.calc_params.calc_params(landmarks)

        # Extract HOG features
        hog_features = self.pyfhog.extract_fhog_features(aligned_face, cell_size=8)

        # Extract geometric features (or zeros if requested)
        if use_zero_geom:
            geom_features = np.zeros(238, dtype=np.float32)
        else:
            geom_features = self.pdm_parser.extract_geometric_features(params_local)

        return {
            'landmarks': landmarks,
            'params_global': params_global,
            'params_local': params_local,
            'aligned_face': aligned_face,
            'hog_features': hog_features,
            'geom_features': geom_features
        }


def predict_aus(hog_features: np.ndarray, geom_features: np.ndarray, models: Dict) -> Dict[str, float]:
    """
    Predict AU intensities using SVR models.

    Simplified version - just computes raw SVR output without median normalization.
    """
    full_features = np.concatenate([hog_features, geom_features])

    predictions = {}
    for au_name, model in models.items():
        centered = full_features - model['means'].flatten()
        pred = np.dot(centered.reshape(1, -1), model['support_vectors']) + model['bias']
        pred = float(np.clip(pred[0, 0], 0.0, 5.0))
        predictions[au_name] = pred

    return predictions


def load_au_models(models_dir: Path) -> Dict:
    """Load AU SVR models."""
    from pyfaceau.prediction.model_parser import OF22ModelParser

    parser = OF22ModelParser(str(models_dir))
    models = parser.load_all_models(use_recommended=True, use_combined=True)
    return models


def main():
    parser = argparse.ArgumentParser(description='Test SPIGA AU pipeline vs CLNF baseline')
    parser.add_argument('input', nargs='?', default='IMG_3324.MOV', help='Input video file')
    parser.add_argument('--frames', type=str, default='100,200,300', help='Comma-separated frame numbers')
    parser.add_argument('--reg-factor', type=float, default=10.0, help='CalcParams regularization factor for SPIGA')
    parser.add_argument('--zero-geom', action='store_true', help='Use zero geometric features (HOG-only)')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    # Resolve paths
    if not Path(args.input).is_absolute():
        video_path = script_dir / args.input
        if not video_path.exists():
            video_path = script_dir.parent / 'S Data' / 'Paralysis Cohort' / args.input
    else:
        video_path = Path(args.input)

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return 1

    output_dir = Path(args.output) if args.output else script_dir / 'test_output' / 'spiga_au_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_nums = [int(f.strip()) for f in args.frames.split(',')]
    weights_dir = script_dir / 'weights'

    # Find AU models
    au_models_dir = weights_dir / 'AU_predictors'
    if not au_models_dir.exists():
        print(f"Error: AU models not found at {au_models_dir}")
        return 1

    print("=" * 80)
    print("SPIGA AU Pipeline Test")
    print("=" * 80)
    print(f"Video: {video_path.name}")
    print(f"Frames: {frame_nums}")
    print(f"Regularization factor: {args.reg_factor}")
    print(f"Zero geometric features: {args.zero_geom}")

    # Initialize pipelines (SPIGA first to avoid segfault)
    print("\nInitializing SPIGA pipeline...")
    spiga_pipeline = SPIGAAUPipeline(weights_dir, reg_factor=args.reg_factor)

    print("Initializing CLNF pipeline...")
    clnf_pipeline = CLNFAUPipeline(weights_dir)

    print("Loading AU models...")
    au_models = load_au_models(au_models_dir)
    print(f"Loaded {len(au_models)} AU models: {sorted(au_models.keys())}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rotation = get_video_rotation(str(video_path))

    print(f"\nVideo info:")
    print(f"  Total frames: {total_frames}")
    print(f"  Rotation: {rotation}°")

    # Results storage
    results = []

    for frame_num in frame_nums:
        print(f"\n{'='*60}")
        print(f"Frame {frame_num}")
        print(f"{'='*60}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"  Failed to read frame")
            continue

        frame = rotate_frame(frame, rotation)

        # Reset temporal state for both detectors when seeking to non-consecutive frames
        spiga_pipeline.spiga.reset_tracking_history()
        clnf_pipeline.landmark_detector.reset_temporal_state()

        # Process with both pipelines
        t0 = time.time()
        clnf_result = clnf_pipeline.process_frame(frame)
        clnf_time = time.time() - t0

        t0 = time.time()
        spiga_result = spiga_pipeline.process_frame(frame, use_zero_geom=args.zero_geom)
        spiga_time = time.time() - t0

        if clnf_result is None:
            print(f"  CLNF: Failed to process")
            continue
        if spiga_result is None:
            print(f"  SPIGA: Failed to process")
            continue

        print(f"\n  Processing time:")
        print(f"    CLNF:  {clnf_time*1000:.1f} ms")
        print(f"    SPIGA: {spiga_time*1000:.1f} ms")

        # Alignment diagnostics: compare pose centers and Kabsch parameters
        clnf_tx = clnf_result['params_global'][4]
        clnf_ty = clnf_result['params_global'][5]
        spiga_centroid = np.mean(spiga_result['landmarks'], axis=0)
        clnf_lm_centroid = np.mean(clnf_result['landmarks'], axis=0)

        print(f"\n  Alignment diagnostics:")
        print(f"    CLNF  pose_tx/ty:   ({clnf_tx:.1f}, {clnf_ty:.1f})")
        print(f"    CLNF  lm centroid:  ({clnf_lm_centroid[0]:.1f}, {clnf_lm_centroid[1]:.1f})")
        print(f"    SPIGA lm centroid:  ({spiga_centroid[0]:.1f}, {spiga_centroid[1]:.1f})")
        print(f"    Center offset:      ({spiga_centroid[0]-clnf_tx:.1f}, {spiga_centroid[1]-clnf_ty:.1f})")

        # Compare params_local
        clnf_zscore = clnf_result['params_local'] / np.sqrt(clnf_pipeline.pdm_parser.eigen_values)
        spiga_zscore = spiga_result['params_local'] / np.sqrt(spiga_pipeline.pdm_parser.eigen_values)

        print(f"\n  params_local z-scores:")
        print(f"    CLNF:  min={clnf_zscore.min():.2f}, max={clnf_zscore.max():.2f}, mean={np.abs(clnf_zscore).mean():.2f}")
        print(f"    SPIGA: min={spiga_zscore.min():.2f}, max={spiga_zscore.max():.2f}, mean={np.abs(spiga_zscore).mean():.2f}")

        # Compare HOG features
        hog_diff = np.abs(clnf_result['hog_features'] - spiga_result['hog_features'])
        print(f"\n  HOG features (4464 values):")
        print(f"    Mean absolute diff: {hog_diff.mean():.4f}")
        print(f"    Max absolute diff:  {hog_diff.max():.4f}")
        print(f"    Correlation:        {np.corrcoef(clnf_result['hog_features'].flatten(), spiga_result['hog_features'].flatten())[0,1]:.4f}")

        # Compare geometric features
        geom_diff = np.abs(clnf_result['geom_features'] - spiga_result['geom_features'])
        print(f"\n  Geometric features (238 values):")
        print(f"    Mean absolute diff: {geom_diff.mean():.2f}")
        print(f"    Max absolute diff:  {geom_diff.max():.2f}")
        print(f"    Correlation:        {np.corrcoef(clnf_result['geom_features'], spiga_result['geom_features'])[0,1]:.4f}")

        # Predict AUs
        clnf_aus = predict_aus(clnf_result['hog_features'], clnf_result['geom_features'], au_models)
        spiga_aus = predict_aus(spiga_result['hog_features'], spiga_result['geom_features'], au_models)

        print(f"\n  AU Predictions:")
        print(f"    {'AU':<10} {'CLNF':>8} {'SPIGA':>8} {'Diff':>8}")
        print(f"    {'-'*36}")

        au_diffs = []
        for au_name in sorted(clnf_aus.keys()):
            clnf_val = clnf_aus[au_name]
            spiga_val = spiga_aus[au_name]
            diff = abs(clnf_val - spiga_val)
            au_diffs.append(diff)
            print(f"    {au_name:<10} {clnf_val:>8.2f} {spiga_val:>8.2f} {diff:>8.2f}")

        print(f"    {'-'*36}")
        print(f"    {'Mean diff':<10} {'':<8} {'':<8} {np.mean(au_diffs):>8.2f}")
        print(f"    {'Max diff':<10} {'':<8} {'':<8} {np.max(au_diffs):>8.2f}")

        # Store results
        results.append({
            'frame': frame_num,
            'clnf_aus': clnf_aus,
            'spiga_aus': spiga_aus,
            'au_mean_diff': np.mean(au_diffs),
            'hog_correlation': np.corrcoef(clnf_result['hog_features'].flatten(), spiga_result['hog_features'].flatten())[0,1],
            'geom_correlation': np.corrcoef(clnf_result['geom_features'], spiga_result['geom_features'])[0,1]
        })

        # Save aligned face comparison
        scale = 3
        clnf_scaled = cv2.resize(clnf_result['aligned_face'], (112*scale, 112*scale), interpolation=cv2.INTER_NEAREST)
        spiga_scaled = cv2.resize(spiga_result['aligned_face'], (112*scale, 112*scale), interpolation=cv2.INTER_NEAREST)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(clnf_scaled, 'CLNF', (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(spiga_scaled, 'SPIGA', (10, 30), font, 0.8, (0, 0, 255), 2)

        comparison = np.hstack([clnf_scaled, spiga_scaled])
        cv2.imwrite(str(output_dir / f'frame{frame_num:04d}_aligned.jpg'), comparison)

    cap.release()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if results:
        mean_au_diff = np.mean([r['au_mean_diff'] for r in results])
        mean_hog_corr = np.mean([r['hog_correlation'] for r in results])
        mean_geom_corr = np.mean([r['geom_correlation'] for r in results])

        print(f"\nAcross {len(results)} frames:")
        print(f"  Mean AU difference:      {mean_au_diff:.3f}")
        print(f"  Mean HOG correlation:    {mean_hog_corr:.4f}")
        print(f"  Mean Geom correlation:   {mean_geom_corr:.4f}")

        print(f"\nConclusion:")
        if mean_au_diff < 0.5 and mean_hog_corr > 0.95:
            print("  ✓ SPIGA pipeline produces comparable AU predictions")
        elif mean_au_diff < 1.0:
            print("  ~ SPIGA pipeline produces similar but not identical AU predictions")
        else:
            print("  ✗ SPIGA pipeline produces significantly different AU predictions")

    print(f"\nOutput saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
