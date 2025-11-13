#!/usr/bin/env python3
"""
Derive empirical correction formula for pyMTCNN to match OpenFace C++ output.

Approach:
1. Run pyMTCNN on frames → get raw pyMTCNN bbox
2. Run OpenFace C++ on same frames → get landmarks → estimate effective bbox
3. Compute transformation: pyMTCNN → C++ effective bbox
4. Average across frames to get empirical correction formula
5. Test if this correction improves convergence

This accounts for differences between pyMTCNN and C++ MTCNN implementations.
"""

import cv2
import numpy as np
import sys
import subprocess
from pathlib import Path
import pandas as pd

sys.path.insert(0, 'pyfaceau')
from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNNDetector
from pyclnf import CLNF

# Configuration
VIDEO_CONFIGS = [
    ('Patient Data/Normal Cohort/IMG_0433.MOV', [50, 150, 250]),
    ('Patient Data/Normal Cohort/IMG_0434.MOV', [50, 150, 250]),
    ('Patient Data/Normal Cohort/IMG_0435.MOV', [50, 150, 250]),
]

OPENFACE_BIN = '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction'
OUTPUT_DIR = Path('/tmp/mtcnn_correction_analysis')
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_frame(video_path, frame_num):
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read frame {frame_num} from {video_path}")

    return frame


def run_pymtcnn(frame):
    """Run pyMTCNN detector on frame."""
    detector = OpenFaceMTCNNDetector(
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        scale_factor=0.709
    )

    bboxes, _ = detector.detect_faces(frame)

    if len(bboxes) == 0:
        return None

    det = bboxes[0]
    x1, y1, x2, y2 = det[:4]
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def run_openface_cpp(frame_path, output_dir):
    """Run OpenFace C++ and extract effective bbox from landmarks."""
    # OpenFace needs a video file
    temp_video = output_dir / 'temp_video.mp4'
    frame = cv2.imread(str(frame_path))
    h, w = frame.shape[:2]

    # Create single-frame video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_video), fourcc, 1.0, (w, h))
    out.write(frame)
    out.release()

    # Run OpenFace
    cpp_output_dir = output_dir / 'openface_output'
    cpp_output_dir.mkdir(exist_ok=True)

    cmd = [
        OPENFACE_BIN,
        '-f', str(temp_video),
        '-out_dir', str(cpp_output_dir)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse CSV to get landmarks
    csv_files = list(cpp_output_dir.glob('*.csv'))
    if not csv_files:
        return None

    df = pd.read_csv(csv_files[0])

    # Extract landmark positions
    x_cols = [c for c in df.columns if c.startswith('x_')]
    y_cols = [c for c in df.columns if c.startswith('y_')]

    if len(x_cols) == 0:
        return None

    x_coords = [df[c].values[0] for c in x_cols]
    y_coords = [df[c].values[0] for c in y_cols]

    # Estimate effective bbox from landmarks
    # Use tight bounds around landmarks (OpenFace's correction makes bbox tighter)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Add small margin (5% on each side)
    w = max_x - min_x
    h = max_y - min_y
    margin_w = w * 0.05
    margin_h = h * 0.05

    x = int(min_x - margin_w)
    y = int(min_y - margin_h)
    w = int(w + 2 * margin_w)
    h = int(h + 2 * margin_h)

    return (x, y, w, h)


def compute_correction_coefficients(pymtcnn_bbox, cpp_bbox):
    """Compute what correction transforms pyMTCNN to C++ bbox.

    Returns coefficients for OpenFace-style formula:
        x_new = x + w * dx_coeff
        y_new = y + h * dy_coeff
        w_new = w * w_scale
        h_new = h * h_scale
    """
    x1, y1, w1, h1 = pymtcnn_bbox
    x2, y2, w2, h2 = cpp_bbox

    # Position corrections (normalized by bbox size)
    dx = x2 - x1
    dy = y2 - y1
    dx_coeff = dx / w1 if w1 > 0 else 0
    dy_coeff = dy / h1 if h1 > 0 else 0

    # Scale corrections
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
    print("EMPIRICAL CORRECTION FORMULA DERIVATION")
    print("="*80)
    print()
    print("Deriving correction to transform pyMTCNN → OpenFace C++ effective bbox")
    print(f"Videos: {len(VIDEO_CONFIGS)}, Frames per video: {len(VIDEO_CONFIGS[0][1])}")
    print()

    all_coefficients = []
    all_comparisons = []

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

            # Save frame for OpenFace
            frame_path = OUTPUT_DIR / f'{video_name}_frame_{frame_num}.jpg'
            cv2.imwrite(str(frame_path), frame)

            # Run pyMTCNN
            print(f"    Running pyMTCNN...")
            pymtcnn_bbox = run_pymtcnn(frame)

            if pymtcnn_bbox is None:
                print(f"    ✗ pyMTCNN failed to detect face")
                continue

            print(f"    ✓ pyMTCNN bbox: {pymtcnn_bbox}")

            # Run OpenFace C++
            print(f"    Running OpenFace C++...")
            frame_output_dir = OUTPUT_DIR / f'{video_name}_frame_{frame_num}_cpp'
            cpp_bbox = run_openface_cpp(frame_path, frame_output_dir)

            if cpp_bbox is None:
                print(f"    ✗ OpenFace C++ failed")
                continue

            print(f"    ✓ C++ effective bbox: {cpp_bbox}")

            # Compute correction coefficients
            coeffs = compute_correction_coefficients(pymtcnn_bbox, cpp_bbox)
            print(f"    Correction: dx={coeffs['dx_coeff']:+.4f}, dy={coeffs['dy_coeff']:+.4f}, "
                  f"w_scale={coeffs['w_scale']:.4f}, h_scale={coeffs['h_scale']:.4f}")

            all_coefficients.append(coeffs)
            all_comparisons.append({
                'video': video_name,
                'frame': frame_num,
                'pymtcnn_bbox': pymtcnn_bbox,
                'cpp_bbox': cpp_bbox,
                **coeffs
            })

            print()

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

    print("Derived formula to transform pyMTCNN → OpenFace C++ output:")
    print()
    print(f"  x_new = x + width * {dx_mean:+.4f}  (std: {dx_std:.4f})")
    print(f"  y_new = y + height * {dy_mean:+.4f}  (std: {dy_std:.4f})")
    print(f"  width_new = width * {w_scale_mean:.4f}  (std: {w_scale_std:.4f})")
    print(f"  height_new = height * {h_scale_mean:.4f}  (std: {h_scale_std:.4f})")
    print()

    # Compare to OpenFace's hardcoded formula
    print("COMPARISON TO OPENFACE C++ HARDCODED FORMULA:")
    print("(From FaceDetectorMTCNN.cpp:919-924)")
    print()
    print("  OpenFace hardcoded (C++ MTCNN → corrected):")
    print("    x_new = x + width * -0.0075")
    print("    y_new = y + height * +0.2459")
    print("    width_new = width * 1.0323")
    print("    height_new = height * 0.7751")
    print()
    print("  Our empirical (pyMTCNN → C++ final output):")
    print(f"    x_new = x + width * {dx_mean:+.4f}")
    print(f"    y_new = y + height * {dy_mean:+.4f}")
    print(f"    width_new = width * {w_scale_mean:.4f}")
    print(f"    height_new = height * {h_scale_mean:.4f}")
    print()

    if abs(dx_mean - (-0.0075)) < 0.01 and abs(dy_mean - 0.2459) < 0.05:
        print("→ Very similar to OpenFace's formula! pyMTCNN ≈ C++ MTCNN")
    else:
        print("→ Different from OpenFace's formula. pyMTCNN ≠ C++ MTCNN")
        print("   This confirms we need a custom correction for pyMTCNN")

    print()
    print(f"{'='*80}")
    print("CONVERGENCE TESTING")
    print(f"{'='*80}\n")

    # Now test if this correction actually helps convergence
    print("Testing convergence with original vs corrected pyMTCNN bboxes...")
    print()

    convergence_results = []

    for comp in all_comparisons[:3]:  # Test on first 3 frames
        video_name = comp['video']
        frame_num = comp['frame']
        pymtcnn_bbox = comp['pymtcnn_bbox']

        print(f"{video_name} frame {frame_num}:")

        # Load frame
        video_path = next(v for v, _ in VIDEO_CONFIGS if Path(v).stem == video_name)
        frame = extract_frame(video_path, frame_num)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Test with original bbox
        print(f"  Original pyMTCNN bbox...")
        orig_conv = test_convergence(gray, pymtcnn_bbox)
        print(f"    Final update: {orig_conv['final_update']:.6f} ({orig_conv['ratio_to_target']:.1f}x target)")

        # Test with corrected bbox
        corrected_bbox = apply_correction(pymtcnn_bbox, dx_mean, dy_mean, w_scale_mean, h_scale_mean)
        print(f"  Corrected pyMTCNN bbox...")
        corr_conv = test_convergence(gray, corrected_bbox)
        print(f"    Final update: {corr_conv['final_update']:.6f} ({corr_conv['ratio_to_target']:.1f}x target)")

        improvement = (orig_conv['final_update'] - corr_conv['final_update']) / orig_conv['final_update'] * 100
        print(f"  Improvement: {improvement:+.1f}%")
        print()

        convergence_results.append({
            'video': video_name,
            'frame': frame_num,
            'original_ratio': orig_conv['ratio_to_target'],
            'corrected_ratio': corr_conv['ratio_to_target'],
            'improvement_pct': improvement
        })

    # Summary
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    df_conv = pd.DataFrame(convergence_results)
    mean_improvement = df_conv['improvement_pct'].mean()

    print(f"Average convergence improvement: {mean_improvement:+.1f}%")
    print()

    if mean_improvement > 10:
        print("✓ Empirical correction significantly improves convergence!")
        print("  Recommend implementing this correction in production")
    elif mean_improvement > 5:
        print("~ Empirical correction provides moderate improvement")
        print("  Worth considering for production")
    elif mean_improvement > 0:
        print("~ Empirical correction provides minor improvement")
        print("  May not justify added complexity")
    else:
        print("✗ Empirical correction degrades convergence")
        print("  Do NOT implement")

    print()
    print("RECOMMENDATION:")
    print()

    if abs(dx_mean - (-0.0075)) < 0.01 and abs(dy_mean - 0.2459) < 0.05:
        print("Since pyMTCNN output is very similar to C++ MTCNN,")
        print("you can use OpenFace's existing correction formula.")
    else:
        print("pyMTCNN differs from C++ MTCNN. Use the empirical formula derived above.")

    print()
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
