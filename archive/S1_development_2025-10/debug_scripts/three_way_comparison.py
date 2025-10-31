#!/usr/bin/env python3
"""
Three-Way OpenFace Comparison Script

Processes ONE test video through three different pipelines:
1. OpenFace 3.0 with ONNX (current production)
2. OpenFace 3.0 with pure PyTorch (no ONNX optimization)
3. OpenFace 2.2 binary (reference baseline)

Then compares the outputs to determine root cause of AU differences.

Usage:
    python three_way_comparison.py /path/to/test_video.mp4
"""

import sys
import os
import subprocess
import shutil
import time
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import OpenFace 3.0 processor
from openface_integration import OpenFace3Processor

# Configuration
OPENFACE2_BINARY = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
OUTPUT_DIR = Path.home() / "Documents/SplitFace/S1O Processed Files/Three-Way Comparison"


def process_with_openface3_onnx(video_path, output_csv):
    """
    Process video with OpenFace 3.0 using ONNX models (current production)

    Args:
        video_path: Path to input video
        output_csv: Path for output CSV

    Returns:
        tuple: (success, processing_time, error_msg)
    """
    print("\n" + "="*80)
    print("PIPELINE 1: OpenFace 3.0 with ONNX (Current Production)")
    print("="*80)

    start_time = time.time()

    try:
        # Initialize with ONNX models (default behavior)
        processor = OpenFace3Processor(
            device='cpu',
            calculate_landmarks=True,
            num_threads=6,
            debug_mode=False,
            skip_face_detection=False  # Enable face detection for raw videos
        )

        # Check which backend was loaded
        if hasattr(processor.multitask_model, 'backend'):
            backend = processor.multitask_model.backend
            print(f"MTL Model Backend: {backend}")

        if hasattr(processor.landmark_detector, 'backend'):
            backend = processor.landmark_detector.backend
            print(f"Landmark Detector Backend: {backend}")

        # Process video
        frame_count = processor.process_video(video_path, output_csv)

        elapsed_time = time.time() - start_time

        print(f"✓ Processed {frame_count} frames in {elapsed_time:.1f}s")
        return True, elapsed_time, None

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed_time, str(e)


def process_with_openface3_pytorch(video_path, output_csv):
    """
    Process video with OpenFace 3.0 using pure PyTorch (no ONNX)

    This forces PyTorch by not providing ONNX model paths.

    Args:
        video_path: Path to input video
        output_csv: Path for output CSV

    Returns:
        tuple: (success, processing_time, error_msg)
    """
    print("\n" + "="*80)
    print("PIPELINE 2: OpenFace 3.0 with Pure PyTorch (No ONNX)")
    print("="*80)

    start_time = time.time()

    try:
        # Get weights directory
        script_dir = Path(__file__).parent
        weights_dir = script_dir / 'weights'

        # Import components directly to bypass ONNX
        from openface.landmark_detection import LandmarkDetector
        from openface.multitask_model import MultitaskPredictor
        from openface3_to_18au_adapter import OpenFace3To18AUAdapter
        from openface.Pytorch_Retinaface.models.retinaface import RetinaFace
        from openface.Pytorch_Retinaface.detect import load_model
        from openface.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
        from openface.Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
        from openface.Pytorch_Retinaface.utils.box_utils import decode
        from openface.Pytorch_Retinaface.data import cfg_mnet
        import cv2
        import csv
        import torch

        print("Initializing pure PyTorch models (this may take 30-60 seconds)...")

        # Initialize RetinaFace for face detection
        cfg = cfg_mnet
        retinaface_model = RetinaFace(cfg=cfg, phase='test')
        retinaface_model = load_model(retinaface_model, str(weights_dir / 'Alignment_RetinaFace.pth'), True)
        retinaface_model.eval()
        retinaface_model = retinaface_model.to('cpu')
        print("  ✓ RetinaFace loaded (PyTorch)")

        # Initialize landmark detector WITHOUT ONNX
        landmark_detector = LandmarkDetector(
            model_path=str(weights_dir / 'Landmark_98.pkl'),
            device='cpu'
        )
        print("  ✓ Landmark detector loaded (PyTorch)")

        # Initialize multitask model WITHOUT ONNX
        multitask_model = MultitaskPredictor(
            model_path=str(weights_dir / 'MTL_backbone.pth'),
            device='cpu'
        )
        print("  ✓ Multitask model loaded (PyTorch)")

        # Initialize adapter
        au_adapter = OpenFace3To18AUAdapter()
        print("  ✓ AU adapter initialized")

        # Process video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\nProcessing {total_frames} frames...")

        csv_rows = []
        frame_index = 0

        # Helper function for face detection
        def detect_faces_pytorch(frame, model, cfg_obj):
            """Detect faces using PyTorch RetinaFace"""
            img = frame.astype(np.float32)
            im_height, im_width, _ = img.shape

            # Preprocessing
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)

            # Run detection
            with torch.no_grad():
                loc, conf, _ = model(img)

            # Generate priors
            priorbox = PriorBox(cfg_obj, image_size=(im_height, im_width))
            priors = priorbox.forward()
            prior_data = priors.data

            # Decode detections
            boxes = decode(loc.data.squeeze(0), prior_data, cfg_obj['variance'])
            boxes = boxes * torch.tensor([im_width, im_height, im_width, im_height])
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

            # Filter by confidence
            inds = np.where(scores > 0.5)[0]
            boxes = boxes[inds]
            scores = scores[inds]

            # Keep top-K before NMS
            order = scores.argsort()[::-1]
            boxes = boxes[order]
            scores = scores[order]

            # Apply NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, 0.4)
            dets = dets[keep, :]

            return dets

        # Process frames with face detection
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_index / fps

            try:
                # Detect faces using RetinaFace
                dets = detect_faces_pytorch(frame, retinaface_model, cfg)

                if dets is None or len(dets) == 0:
                    # No face detected
                    raise ValueError("No face detected")

                # Use first detected face
                det = dets[0]
                x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                confidence = float(det[4])

                # Extract face region
                cropped_face = frame[y1:y2, x1:x2]

                # Create bbox for landmark detector
                bbox = np.array([x1, y1, x2, y2, confidence])

                # Extract 98-point landmarks
                landmarks_98_list = landmark_detector.detect_landmarks(
                    frame,
                    np.array([bbox]),
                    confidence_threshold=0.5
                )
                landmarks_98 = landmarks_98_list[0] if landmarks_98_list is not None and len(landmarks_98_list) > 0 else None

                # Extract AUs using multitask model
                emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_face)

                # Convert to CSV row
                csv_row = au_adapter.get_csv_row_dict(
                    au_vector_8d=au_output,
                    landmarks_98=landmarks_98,
                    frame_num=frame_index,
                    timestamp=timestamp,
                    confidence=confidence,
                    success=1
                )

                csv_rows.append(csv_row)

            except Exception as e:
                # Create failed frame row
                dummy_au_8d = np.zeros(8)
                csv_row = au_adapter.get_csv_row_dict(
                    au_vector_8d=dummy_au_8d,
                    landmarks_98=None,
                    frame_num=frame_index,
                    timestamp=timestamp,
                    confidence=0.0,
                    success=0
                )
                csv_rows.append(csv_row)

            frame_index += 1

            # Print progress every 100 frames
            if frame_index % 100 == 0:
                print(f"  Progress: {frame_index}/{total_frames} frames", end='\r')

        cap.release()

        # Write CSV
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, 'w', newline='') as csvfile:
            if csv_rows:
                fieldnames = list(csv_rows[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)

        elapsed_time = time.time() - start_time

        print(f"\n✓ Processed {len(csv_rows)} frames in {elapsed_time:.1f}s")
        return True, elapsed_time, None

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed_time, str(e)


def process_with_openface2(video_path, output_csv):
    """
    Process video with OpenFace 2.2 binary (reference baseline)

    Args:
        video_path: Path to input video
        output_csv: Path for output CSV

    Returns:
        tuple: (success, processing_time, error_msg)
    """
    print("\n" + "="*80)
    print("PIPELINE 3: OpenFace 2.2 Binary (Reference Baseline)")
    print("="*80)

    start_time = time.time()

    try:
        # Verify binary exists
        if not Path(OPENFACE2_BINARY).exists():
            raise FileNotFoundError(f"OpenFace 2.2 binary not found: {OPENFACE2_BINARY}")

        # Create temp directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            processed_dir = temp_dir_path / "processed"

            # Run OpenFace 2.2
            command = [
                OPENFACE2_BINARY,
                "-aus",
                "-verbose",
                "-tracked",
                "-f", str(video_path)
            ]

            print(f"Running: {' '.join(command)}")

            result = subprocess.run(
                command,
                cwd=str(temp_dir_path),
                check=True,
                capture_output=True,
                text=True,
                timeout=600
            )

            # Find generated CSV
            expected_csv_name = Path(video_path).stem + ".csv"
            source_csv = processed_dir / expected_csv_name

            # Wait for CSV
            for _ in range(50):
                if source_csv.exists():
                    break
                time.sleep(0.1)

            if not source_csv.exists():
                raise FileNotFoundError(f"OpenFace 2.2 did not create CSV: {source_csv}")

            # Copy to output
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_csv, output_csv)

            # Remove hidden flag
            subprocess.run(['chflags', 'nohidden', str(output_csv)], check=True, capture_output=True)

        elapsed_time = time.time() - start_time

        print(f"✓ Processed in {elapsed_time:.1f}s")
        return True, elapsed_time, None

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed_time, str(e)


def compare_csvs(onnx_csv, pytorch_csv, of2_csv, output_report):
    """
    Compare the three CSV outputs and generate diagnostic report

    Args:
        onnx_csv: Path to ONNX output
        pytorch_csv: Path to PyTorch output
        of2_csv: Path to OpenFace 2.2 output
        output_report: Path for text report
    """
    print("\n" + "="*80)
    print("COMPARISON ANALYSIS")
    print("="*80)

    # Load CSVs
    try:
        df_onnx = pd.read_csv(onnx_csv)
        print(f"✓ ONNX CSV loaded: {len(df_onnx)} frames")
    except Exception as e:
        print(f"✗ Failed to load ONNX CSV: {e}")
        return

    try:
        df_pytorch = pd.read_csv(pytorch_csv)
        print(f"✓ PyTorch CSV loaded: {len(df_pytorch)} frames")
    except Exception as e:
        print(f"✗ Failed to load PyTorch CSV: {e}")
        return

    try:
        df_of2 = pd.read_csv(of2_csv)
        print(f"✓ OpenFace 2.2 CSV loaded: {len(df_of2)} frames")
    except Exception as e:
        print(f"✗ Failed to load OpenFace 2.2 CSV: {e}")
        return

    # Extract AU columns
    onnx_aus = [c for c in df_onnx.columns if c.startswith('AU') and '_r' in c]
    pytorch_aus = [c for c in df_pytorch.columns if c.startswith('AU') and '_r' in c]
    of2_aus = [c for c in df_of2.columns if c.startswith('AU') and '_r' in c]

    common_aus = sorted(set(onnx_aus) & set(pytorch_aus) & set(of2_aus))

    print(f"\nComparing {len(common_aus)} common AUs...")

    # Build report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("THREE-WAY OPENFACE COMPARISON REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"Test Video: {onnx_csv.stem}")
    report_lines.append(f"Frames: {len(df_onnx)}")
    report_lines.append("")

    # Key question: Is ONNX the problem?
    report_lines.append("="*80)
    report_lines.append("KEY DIAGNOSTIC: Is ONNX Conversion the Problem?")
    report_lines.append("="*80)
    report_lines.append("")

    # Compare ONNX vs PyTorch (should be identical if ONNX is correct)
    onnx_pytorch_correlations = []
    onnx_pytorch_diffs = []

    for au in common_aus:
        onnx_vals = df_onnx[au].dropna()
        pytorch_vals = df_pytorch[au].dropna()

        if len(onnx_vals) > 0 and len(pytorch_vals) > 0:
            # Correlation
            min_len = min(len(df_onnx), len(df_pytorch))
            onnx_aligned = df_onnx[au][:min_len]
            pytorch_aligned = df_pytorch[au][:min_len]
            valid_mask = ~(onnx_aligned.isna() | pytorch_aligned.isna())

            if valid_mask.sum() >= 2:
                corr = onnx_aligned[valid_mask].corr(pytorch_aligned[valid_mask])
                onnx_pytorch_correlations.append((au, corr))

            # Mean difference
            mean_diff = abs(onnx_vals.mean() - pytorch_vals.mean())
            onnx_pytorch_diffs.append((au, mean_diff))

    # Verdict on ONNX
    avg_onnx_pytorch_corr = np.mean([c[1] for c in onnx_pytorch_correlations if not np.isnan(c[1])])

    report_lines.append(f"ONNX vs PyTorch Average Correlation: {avg_onnx_pytorch_corr:.3f}")
    report_lines.append("")

    if avg_onnx_pytorch_corr > 0.95:
        report_lines.append("✓ VERDICT: ONNX conversion is CORRECT")
        report_lines.append("  ONNX and PyTorch produce nearly identical results")
        report_lines.append("  Problem is NOT with ONNX optimization")
        report_lines.append("")
    elif avg_onnx_pytorch_corr > 0.80:
        report_lines.append("⚠ VERDICT: ONNX conversion has MINOR issues")
        report_lines.append("  ONNX and PyTorch are similar but not identical")
        report_lines.append("  Small accuracy loss from ONNX conversion")
        report_lines.append("")
    else:
        report_lines.append("❌ VERDICT: ONNX conversion is BROKEN")
        report_lines.append("  ONNX and PyTorch produce very different results")
        report_lines.append("  Problem IS with ONNX conversion - needs fixing")
        report_lines.append("")

    # Compare PyTorch vs OpenFace 2.2 (should be similar if MTL model is compatible)
    pytorch_of2_correlations = []

    for au in common_aus:
        pytorch_vals = df_pytorch[au].dropna()
        of2_vals = df_of2[au].dropna()

        if len(pytorch_vals) > 0 and len(of2_vals) > 0:
            min_len = min(len(df_pytorch), len(df_of2))
            pytorch_aligned = df_pytorch[au][:min_len]
            of2_aligned = df_of2[au][:min_len]
            valid_mask = ~(pytorch_aligned.isna() | of2_aligned.isna())

            if valid_mask.sum() >= 2:
                corr = pytorch_aligned[valid_mask].corr(of2_aligned[valid_mask])
                pytorch_of2_correlations.append((au, corr))

    avg_pytorch_of2_corr = np.mean([c[1] for c in pytorch_of2_correlations if not np.isnan(c[1])])

    report_lines.append("="*80)
    report_lines.append("KEY DIAGNOSTIC: Is MTL Model Compatible with OpenFace 2.2?")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"PyTorch vs OpenFace 2.2 Average Correlation: {avg_pytorch_of2_corr:.3f}")
    report_lines.append("")

    if avg_pytorch_of2_corr > 0.80:
        report_lines.append("✓ VERDICT: MTL model is COMPATIBLE with OpenFace 2.2")
        report_lines.append("  PyTorch OpenFace 3.0 produces similar results to OpenFace 2.2")
        report_lines.append("  Models should work after fixing ONNX (if broken)")
        report_lines.append("")
    elif avg_pytorch_of2_corr > 0.50:
        report_lines.append("⚠ VERDICT: MTL model is PARTIALLY compatible")
        report_lines.append("  Some correlation but significant differences")
        report_lines.append("  Models may need recalibration")
        report_lines.append("")
    else:
        report_lines.append("❌ VERDICT: MTL model is INCOMPATIBLE with OpenFace 2.2")
        report_lines.append("  PyTorch OpenFace 3.0 produces very different results")
        report_lines.append("  Must retrain models OR switch to OpenFace 2.2")
        report_lines.append("")

    # Overall conclusion
    report_lines.append("="*80)
    report_lines.append("OVERALL DIAGNOSIS")
    report_lines.append("="*80)
    report_lines.append("")

    if avg_onnx_pytorch_corr > 0.95 and avg_pytorch_of2_corr > 0.80:
        report_lines.append("✓✓ GOOD NEWS: Everything works!")
        report_lines.append("   - ONNX conversion is correct")
        report_lines.append("   - MTL model is compatible with OpenFace 2.2")
        report_lines.append("   - Problem must be elsewhere in the pipeline")
        report_lines.append("")
    elif avg_onnx_pytorch_corr < 0.80 and avg_pytorch_of2_corr > 0.80:
        report_lines.append("⚠ FIXABLE: ONNX conversion needs fixing")
        report_lines.append("   - PyTorch works well, ONNX is broken")
        report_lines.append("   - FIX: Debug ONNX conversion scripts")
        report_lines.append("   - After fix, models should work with OpenFace 3.0")
        report_lines.append("")
    elif avg_onnx_pytorch_corr > 0.95 and avg_pytorch_of2_corr < 0.50:
        report_lines.append("❌ FUNDAMENTAL: MTL model incompatible")
        report_lines.append("   - ONNX is fine, but MTL produces different AUs")
        report_lines.append("   - OPTIONS:")
        report_lines.append("     1. Retrain models on OpenFace 3.0 data")
        report_lines.append("     2. Switch to OpenFace 2.2 for production")
        report_lines.append("")
    else:
        report_lines.append("❌❌ DOUBLE PROBLEM: Both ONNX and MTL are broken")
        report_lines.append("   - ONNX conversion has issues")
        report_lines.append("   - MTL model is also incompatible")
        report_lines.append("   - RECOMMEND: Switch to OpenFace 2.2 while fixing")
        report_lines.append("")

    # Detailed AU comparison table
    report_lines.append("="*80)
    report_lines.append("DETAILED AU COMPARISON")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"{'AU':<12} {'ONNX Mean':<12} {'PyTorch Mean':<12} {'OF2.2 Mean':<12} {'ONNX-PT Corr':<15} {'PT-OF2 Corr':<15}")
    report_lines.append("-"*80)

    for au in common_aus:
        onnx_mean = df_onnx[au].dropna().mean() if len(df_onnx[au].dropna()) > 0 else np.nan
        pytorch_mean = df_pytorch[au].dropna().mean() if len(df_pytorch[au].dropna()) > 0 else np.nan
        of2_mean = df_of2[au].dropna().mean() if len(df_of2[au].dropna()) > 0 else np.nan

        onnx_pt_corr = next((c[1] for c in onnx_pytorch_correlations if c[0] == au), np.nan)
        pt_of2_corr = next((c[1] for c in pytorch_of2_correlations if c[0] == au), np.nan)

        report_lines.append(
            f"{au:<12} {onnx_mean:<12.3f} {pytorch_mean:<12.3f} {of2_mean:<12.3f} "
            f"{onnx_pt_corr:<15.3f} {pt_of2_corr:<15.3f}"
        )

    report_lines.append("")
    report_lines.append("="*80)

    # Write report
    report_text = "\n".join(report_lines)
    output_report.write_text(report_text)

    # Print to console
    print("\n" + report_text)

    print(f"\nReport saved to: {output_report}")


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python three_way_comparison.py /path/to/test_video.mp4")
        print("\nExample:")
        print('  python three_way_comparison.py "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/20240723_175947000_iOS_left_mirrored.mp4"')
        sys.exit(1)

    video_path = Path(sys.argv[1])

    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    print("\n" + "="*80)
    print("THREE-WAY OPENFACE COMPARISON")
    print("="*80)
    print(f"\nTest Video: {video_path.name}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("\nThis will process the video through:")
    print("  1. OpenFace 3.0 with ONNX (current production)")
    print("  2. OpenFace 3.0 with pure PyTorch (no ONNX)")
    print("  3. OpenFace 2.2 binary (reference baseline)")
    print("\nEstimated time: 5-15 minutes depending on video length")
    print("="*80)

    # Skip confirmation prompt if running non-interactively
    if sys.stdin.isatty():
        input("\nPress Enter to start, or Ctrl+C to cancel...")
    else:
        print("\nStarting automatically (non-interactive mode)...")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define output paths
    base_name = video_path.stem
    onnx_csv = OUTPUT_DIR / f"{base_name}_OF30_ONNX.csv"
    pytorch_csv = OUTPUT_DIR / f"{base_name}_OF30_PyTorch.csv"
    of2_csv = OUTPUT_DIR / f"{base_name}_OF22_Baseline.csv"
    report_path = OUTPUT_DIR / f"{base_name}_COMPARISON_REPORT.txt"

    results = {}

    # Process through all three pipelines
    results['onnx'] = process_with_openface3_onnx(video_path, onnx_csv)
    results['pytorch'] = process_with_openface3_pytorch(video_path, pytorch_csv)
    results['of2'] = process_with_openface2(video_path, of2_csv)

    # Check results
    all_success = all(r[0] for r in results.values())

    if not all_success:
        print("\n" + "="*80)
        print("ERROR: Some pipelines failed")
        print("="*80)
        for name, (success, time, error) in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {name}: {time:.1f}s" + (f" - {error}" if error else ""))
        sys.exit(1)

    # Compare outputs
    compare_csvs(onnx_csv, pytorch_csv, of2_csv, report_path)

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print("\nOutput files:")
    print(f"  ONNX CSV:      {onnx_csv}")
    print(f"  PyTorch CSV:   {pytorch_csv}")
    print(f"  OF2.2 CSV:     {of2_csv}")
    print(f"  Report:        {report_path}")
    print("\nOpen the report to see the diagnosis!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
