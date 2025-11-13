#!/usr/bin/env python3
"""
Derive empirical correction formula: raw pyMTCNN → OpenFace C++ corrected output

Tests on 9 frames (3 videos × 3 frames each) to:
1. Compute transformation coefficients for each frame
2. Average to get empirical correction formula
3. Test if this correction improves CLNF convergence
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import pandas as pd
import subprocess
import tempfile
import shutil

sys.path.insert(0, 'pyfaceau')
from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN
from pyclnf import CLNF

# Configuration
VIDEO_CONFIGS = [
    ('Patient Data/Normal Cohort/IMG_0433.MOV', [50, 150, 250]),
    ('Patient Data/Normal Cohort/IMG_0434.MOV', [50, 150, 250]),
    ('Patient Data/Normal Cohort/IMG_0435.MOV', [50, 150, 250]),
]

OPENFACE_BIN = '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction'


def extract_frame(video_path, frame_num):
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read frame {frame_num} from {video_path}")

    return frame


def get_raw_pymtcnn_bbox(detector, frame):
    """Get raw pyMTCNN bbox by running detection and inverting the correction."""
    bboxes, _ = detector.detect(frame)

    if len(bboxes) == 0:
        return None

    # The detect() method returns corrected bboxes
    # We need to INVERT the correction to get raw bbox
    det = bboxes[0]
    x_corr, y_corr, x2, y2 = det[:4]
    w_corr = x2 - x_corr
    h_corr = y2 - y_corr

    # Invert OpenFace correction (from openface_mtcnn.py line 746)
    # Original formula:
    #   x_new = x + w * -0.0075
    #   y_new = y + h * 0.2459
    #   w_new = w * 1.0323
    #   h_new = h * 0.7751

    # Invert:
    w_raw = w_corr / 1.0323
    h_raw = h_corr / 0.7751
    x_raw = x_corr - w_raw * (-0.0075)
    y_raw = y_corr - h_raw * 0.2459

    return (int(x_raw), int(y_raw), int(w_raw), int(h_raw))


def run_openface_cpp_get_bbox(video_path, frame_num):
    """Run OpenFace C++ on specific frame and extract the corrected MTCNN bbox from debug output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract single frame and save as temp video
        frame = extract_frame(video_path, frame_num)
        h, w = frame.shape[:2]

        temp_video = tmpdir / 'temp.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_video), fourcc, 1.0, (w, h))
        out.write(frame)
        out.release()

        # Run OpenFace with debug output
        cpp_output_dir = tmpdir / 'output'
        cpp_output_dir.mkdir()

        cmd = [
            OPENFACE_BIN,
            '-f', str(temp_video),
            '-out_dir', str(cpp_output_dir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse debug output to find bbox
        # Look for lines like: "DEBUG_BBOX: 293.145,702.034,418.033,404.659"
        for line in result.stdout.split('\n') + result.stderr.split('\n'):
            if 'DEBUG_BBOX:' in line or 'Face detection bbox' in line:
                # Try to extract x,y,w,h from the line
                parts = line.split(':')[-1].strip().split(',')
                if len(parts) >= 4:
                    try:
                        x, y, w, h = map(float, parts[:4])
                        return (int(x), int(y), int(w), int(h))
                    except:
                        pass

        # If no debug output, try to infer from landmarks
        csv_files = list(cpp_output_dir.glob('*.csv'))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            x_cols = [c for c in df.columns if c.startswith('x_')]
            y_cols = [c for c in df.columns if c.startswith('y_')]

            if len(x_cols) > 0:
                x_coords = [df[c].values[0] for c in x_cols]
                y_coords = [df[c].values[0] for c in y_cols]

                # Estimate bbox from landmarks with 10% margin
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                w = max_x - min_x
                h = max_y - min_y
                margin_w = w * 0.1
                margin_h = h * 0.1

                x = int(min_x - margin_w)
                y = int(min_y - margin_h)
                w = int(w + 2 * margin_w)
                h = int(h + 2 * margin_h)

                return (x, y, w, h)

        return None


def compute_correction_coefficients(raw_bbox, target_bbox):
    """Compute coefficients to transform raw_bbox → target_bbox using OpenFace formula."""
    x1, y1, w1, h1 = raw_bbox
    x2, y2, w2, h2 = target_bbox

    # Solve for coefficients in the formula:
    #   x_new = x + w * dx_coeff
    #   y_new = y + h * dy_coeff
    #   w_new = w * w_scale
    #   h_new = h * h_scale

    dx = x2 - x1
    dy = y2 - y1

    dx_coeff = dx / w1 if w1 > 0 else 0
    dy_coeff = dy / h1 if h1 > 0 else 0
    w_scale = w2 / w1 if w1 > 0 else 1.0
    h_scale = h2 / h1 if h1 > 0 else 1.0

    return {
        'dx_coeff': dx_coeff,
        'dy_coeff': dy_coeff,
        'w_scale': w_scale,
        'h_scale': h_scale
    }


def apply_correction(bbox, dx_coeff, dy_coeff, w_scale, h_scale):
    """Apply correction formula to bbox."""
    x, y, w, h = bbox

    x_new = x + w * dx_coeff
    y_new = y + h * dy_coeff
    w_new = w * w_scale
    h_new = h * h_scale

    return (int(x_new), int(y_new), int(w_new), int(h_new))


def test_convergence(gray, bbox, max_iterations=20):
    """Test CLNF convergence with given bbox."""
    clnf = CLNF(model_dir='pyclnf/models', max_iterations=max_iterations)
    landmarks, info = clnf.fit(gray, bbox, return_params=True)

    return {
        'converged': info['converged'],
        'iterations': info['iterations'],
        'final_update': info['final_update'],
        'ratio_to_target': info['final_update'] / 0.005
    }


def main():
    print("="*80)
    print("DERIVING EMPIRICAL CORRECTION: raw pyMTCNN → OpenFace C++ corrected")
    print("="*80)
    print()
    print(f"Videos: {len(VIDEO_CONFIGS)}")
    print(f"Frames per video: {len(VIDEO_CONFIGS[0][1])}")
    print(f"Total samples: {len(VIDEO_CONFIGS) * len(VIDEO_CONFIGS[0][1])}")
    print()

    # Initialize detector once
    print("Initializing pyMTCNN detector...")
    detector = OpenFaceMTCNN(min_face_size=40, thresholds=[0.6, 0.7, 0.7])
    print()

    all_coefficients = []
    all_results = []

    # Collect data from all frames
    for video_idx, (video_path, frame_nums) in enumerate(VIDEO_CONFIGS, 1):
        video_name = Path(video_path).stem
        print(f"\n{'='*80}")
        print(f"VIDEO {video_idx}/{len(VIDEO_CONFIGS)}: {video_name}")
        print(f"{'='*80}\n")

        for frame_idx, frame_num in enumerate(frame_nums, 1):
            print(f"  Frame {frame_idx}/{len(frame_nums)} (frame #{frame_num}):")

            # Extract frame
            try:
                frame = extract_frame(video_path, frame_num)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"    ✗ Failed to extract frame: {e}")
                continue

            # Get raw pyMTCNN bbox
            print(f"    Getting raw pyMTCNN bbox...")
            try:
                raw_bbox = get_raw_pymtcnn_bbox(detector, frame)
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                continue

            if raw_bbox is None:
                print(f"    ✗ pyMTCNN failed to detect face")
                continue

            print(f"    ✓ Raw pyMTCNN: {raw_bbox}")

            # Get OpenFace C++ corrected bbox
            print(f"    Running OpenFace C++...")
            try:
                cpp_bbox = run_openface_cpp_get_bbox(video_path, frame_num)
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                continue

            if cpp_bbox is None:
                print(f"    ✗ OpenFace C++ failed")
                continue

            print(f"    ✓ C++ corrected: {cpp_bbox}")

            # Compute correction coefficients
            coeffs = compute_correction_coefficients(raw_bbox, cpp_bbox)
            print(f"    Coefficients: dx={coeffs['dx_coeff']:+.4f}, dy={coeffs['dy_coeff']:+.4f}, "
                  f"w_scale={coeffs['w_scale']:.4f}, h_scale={coeffs['h_scale']:.4f}")

            all_coefficients.append(coeffs)

            # Test convergence
            print(f"    Testing convergence...")

            # Original raw bbox
            orig_conv = test_convergence(gray, raw_bbox)
            print(f"      Raw:       {orig_conv['final_update']:.6f} ({orig_conv['ratio_to_target']:.1f}x target)")

            # Corrected bbox
            corrected_bbox = apply_correction(raw_bbox, **coeffs)
            corr_conv = test_convergence(gray, corrected_bbox)
            print(f"      Corrected: {corr_conv['final_update']:.6f} ({corr_conv['ratio_to_target']:.1f}x target)")

            improvement = (orig_conv['final_update'] - corr_conv['final_update']) / orig_conv['final_update'] * 100
            print(f"      Improvement: {improvement:+.1f}%")
            print()

            all_results.append({
                'video': video_name,
                'frame': frame_num,
                'raw_bbox': raw_bbox,
                'cpp_bbox': cpp_bbox,
                'corrected_bbox': corrected_bbox,
                'raw_final_update': orig_conv['final_update'],
                'corrected_final_update': corr_conv['final_update'],
                'improvement_pct': improvement,
                **coeffs
            })

    if len(all_coefficients) == 0:
        print("No valid detections found!")
        return

    # Compute empirical formula (average of all coefficients)
    print(f"\n{'='*80}")
    print("EMPIRICAL CORRECTION FORMULA")
    print(f"{'='*80}\n")

    df = pd.DataFrame(all_coefficients)

    dx_mean = df['dx_coeff'].mean()
    dy_mean = df['dy_coeff'].mean()
    w_scale_mean = df['w_scale'].mean()
    h_scale_mean = df['h_scale'].mean()

    dx_std = df['dx_coeff'].std()
    dy_std = df['dy_coeff'].std()
    w_scale_std = df['w_scale'].std()
    h_scale_std = df['h_scale'].std()

    print("Empirical formula (raw pyMTCNN → OpenFace C++ corrected):")
    print()
    print(f"  x_new = x + width * {dx_mean:+.4f}  (std: {dx_std:.4f})")
    print(f"  y_new = y + height * {dy_mean:+.4f}  (std: {dy_std:.4f})")
    print(f"  width_new = width * {w_scale_mean:.4f}  (std: {w_scale_std:.4f})")
    print(f"  height_new = height * {h_scale_mean:.4f}  (std: {h_scale_std:.4f})")
    print()

    # Compare to OpenFace's hardcoded formula
    print("COMPARISON TO OPENFACE'S HARDCODED FORMULA:")
    print()
    print("  OpenFace hardcoded (C++ MTCNN → corrected):")
    print("    x_new = x + width * -0.0075")
    print("    y_new = y + height * +0.2459")
    print("    width_new = width * 1.0323")
    print("    height_new = height * 0.7751")
    print()
    print("  Our empirical (pyMTCNN → C++ corrected):")
    print(f"    x_new = x + width * {dx_mean:+.4f}")
    print(f"    y_new = y + height * {dy_mean:+.4f}")
    print(f"    width_new = width * {w_scale_mean:.4f}")
    print(f"    height_new = height * {h_scale_mean:.4f}")
    print()

    # Convergence analysis
    print(f"{'='*80}")
    print("CONVERGENCE PERFORMANCE")
    print(f"{'='*80}\n")

    df_results = pd.DataFrame(all_results)

    print(f"{'Frame':<20} {'Raw':<15} {'Corrected':<15} {'Improvement':<15}")
    print("-"*70)
    for _, row in df_results.iterrows():
        frame_name = f"{row['video']} f{row['frame']}"
        raw = row['raw_final_update']
        corr = row['corrected_final_update']
        imp = row['improvement_pct']
        print(f"{frame_name:<20} {raw:>14.6f} {corr:>14.6f} {imp:>13.1f}%")

    print()
    print("SUMMARY STATISTICS:")
    mean_improvement = df_results['improvement_pct'].mean()
    median_improvement = df_results['improvement_pct'].median()
    std_improvement = df_results['improvement_pct'].std()

    print(f"  Mean improvement: {mean_improvement:+.1f}%")
    print(f"  Median improvement: {median_improvement:+.1f}%")
    print(f"  Std improvement: {std_improvement:.1f}%")
    print()

    num_improved = (df_results['improvement_pct'] > 0).sum()
    num_regressed = (df_results['improvement_pct'] < 0).sum()

    print(f"  Improved: {num_improved}/{len(df_results)} ({num_improved/len(df_results)*100:.0f}%)")
    print(f"  Regressed: {num_regressed}/{len(df_results)} ({num_regressed/len(df_results)*100:.0f}%)")
    print()

    # Recommendations
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()

    if mean_improvement > 10:
        print(f"✓ Empirical correction significantly improves convergence (+{mean_improvement:.1f}%)")
        print(f"  Implement this correction in production pipeline")
    elif mean_improvement > 5:
        print(f"~ Empirical correction provides moderate improvement (+{mean_improvement:.1f}%)")
        print(f"  Worth considering for production")
    elif mean_improvement > 0:
        print(f"~ Empirical correction provides minor improvement (+{mean_improvement:.1f}%)")
        print(f"  May not justify added complexity")
    else:
        print(f"✗ Empirical correction degrades convergence ({mean_improvement:.1f}%)")
        print(f"  Do NOT implement")

    print()
    print("="*80)


if __name__ == "__main__":
    main()
