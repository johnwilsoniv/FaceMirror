#!/usr/bin/env python3
"""
Derive correction coefficients: raw pyMTCNN → OpenFace C++ corrected bbox.

This script ONLY computes the transformation coefficients, without testing convergence.
We'll test convergence impact separately.
"""

import cv2
import numpy as np
import sys
import subprocess
from pathlib import Path
import pandas as pd

sys.path.insert(0, 'pyfaceau')
from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN

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
    """Get raw pyMTCNN bbox by inverting the built-in correction."""
    bboxes, _ = detector.detect(frame)

    if len(bboxes) == 0:
        return None

    # The detect() method returns corrected bboxes
    # Invert the correction to get raw bbox
    det = bboxes[0]
    x_corr, y_corr, x2, y2 = det[:4]
    w_corr = x2 - x_corr
    h_corr = y2 - y_corr

    # Invert OpenFace correction
    w_raw = w_corr / 1.0323
    h_raw = h_corr / 0.7751
    x_raw = x_corr - w_raw * (-0.0075)
    y_raw = y_corr - h_raw * 0.2459

    return (int(x_raw), int(y_raw), int(w_raw), int(h_raw))


def get_cpp_corrected_bbox_from_csv(csv_path):
    """Estimate OpenFace C++ effective bbox from landmarks in CSV."""
    df = pd.read_csv(csv_path)

    # Extract landmark positions
    x_cols = [c for c in df.columns if c.startswith('x_')]
    y_cols = [c for c in df.columns if c.startswith('y_')]

    if len(x_cols) == 0:
        return None

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


def run_openface_cpp_minimal(frame, output_dir):
    """Run OpenFace C++ with minimal processing to get landmarks."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save frame as temp image
        temp_img = tmpdir / 'frame.jpg'
        cv2.imwrite(str(temp_img), frame)

        # Create single-frame video
        h, w = frame.shape[:2]
        temp_video = tmpdir / 'temp.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_video), fourcc, 1.0, (w, h))
        out.write(frame)
        out.release()

        # Run OpenFace (just landmark detection, no AU extraction)
        cpp_output_dir = output_dir / 'cpp_output'
        cpp_output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            OPENFACE_BIN,
            '-f', str(temp_video),
            '-out_dir', str(cpp_output_dir),
            '-aus'  # Skip AU extraction to speed up
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Find CSV file
        csv_files = list(cpp_output_dir.glob('*.csv'))
        if not csv_files:
            return None

        return get_cpp_corrected_bbox_from_csv(csv_files[0])


def compute_correction_coefficients(raw_bbox, target_bbox):
    """Compute coefficients to transform raw_bbox → target_bbox."""
    x1, y1, w1, h1 = raw_bbox
    x2, y2, w2, h2 = target_bbox

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


def main():
    print("="*80)
    print("DERIVING CORRECTION COEFFICIENTS: raw pyMTCNN → OpenFace C++ corrected")
    print("="*80)
    print()
    print(f"Processing {len(VIDEO_CONFIGS)} videos × {len(VIDEO_CONFIGS[0][1])} frames = {len(VIDEO_CONFIGS) * len(VIDEO_CONFIGS[0][1])} samples")
    print()
    print("NOTE: This will take ~5-10 minutes as we need to run OpenFace C++ 9 times")
    print()

    # Initialize detector once
    print("Initializing pyMTCNN detector...")
    detector = OpenFaceMTCNN(min_face_size=40, thresholds=[0.6, 0.7, 0.7])
    print()

    all_coefficients = []
    output_dir = Path('/tmp/mtcnn_correction_derivation')
    output_dir.mkdir(exist_ok=True)

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
            print(f"    Running OpenFace C++ (this may take 30-60 seconds)...")
            frame_output_dir = output_dir / f'{video_name}_frame_{frame_num}'
            try:
                cpp_bbox = run_openface_cpp_minimal(frame, frame_output_dir)
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

            all_coefficients.append({
                'video': video_name,
                'frame': frame_num,
                'raw_bbox': raw_bbox,
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

    # Show per-frame breakdown
    print("PER-FRAME COEFFICIENTS:")
    print()
    print(f"{'Video':<15} {'Frame':<8} {'dx_coeff':<12} {'dy_coeff':<12} {'w_scale':<12} {'h_scale':<12}")
    print("-"*80)
    for _, row in df.iterrows():
        print(f"{row['video']:<15} {row['frame']:<8} {row['dx_coeff']:>11.4f} {row['dy_coeff']:>11.4f} "
              f"{row['w_scale']:>11.4f} {row['h_scale']:>11.4f}")

    print()
    print("="*80)
    print("NEXT STEP:")
    print("="*80)
    print()
    print("Now test convergence improvement with a separate script that uses these coefficients.")
    print()


if __name__ == "__main__":
    main()
