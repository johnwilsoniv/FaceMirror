#!/usr/bin/env python3
"""
Validate Corrected RetinaFace Pipeline

Tests complete pipeline with corrected RetinaFace:
1. RetinaFace detection
2. Apply correction transform
3. pyCLNF fitting
4. Compare to C++ OpenFace

Goal: Verify that 2.2% init error leads to good final convergence.
"""

import subprocess
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add pyfaceau to path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

# Correction parameters V2 (derived from calibration - bbox alignment objective)
RETINAFACE_CORRECTION = {
    'alpha': -0.01642482,  # horizontal shift
    'beta':  0.23601291,   # vertical shift
    'gamma': 0.99941800,   # width scale
    'delta': 0.76624999,   # height scale
}

def apply_retinaface_correction(bbox, params=None):
    """
    Apply correction transform to RetinaFace bbox.

    Args:
        bbox: (x, y, w, h) from RetinaFace
        params: Correction parameters (default: calibrated values)

    Returns:
        corrected_bbox: (x, y, w, h) transformed to match C++ MTCNN
    """
    if params is None:
        params = RETINAFACE_CORRECTION

    rx, ry, rw, rh = bbox
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    delta = params['delta']

    cx = rx + alpha * rw
    cy = ry + beta * rh
    cw = rw * gamma
    ch = rh * delta

    return (cx, cy, cw, ch)

def run_cpp_pipeline(image_path, output_dir):
    """Run C++ OpenFace complete pipeline."""
    binary_path = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    cmd = [
        str(binary_path),
        "-f", str(image_path),
        "-out_dir", str(output_dir),
        "-verbose"
    ]

    print("="*80)
    print("C++ OPENFACE PIPELINE")
    print("="*80)
    print("Detector: C++ MTCNN")
    print("Fitting: C++ CLNF")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Parse debug output
    bbox_match = re.search(r'DEBUG_BBOX: ([\d.]+),([\d.]+),([\d.]+),([\d.]+)', output)
    init_params_match = re.search(r'DEBUG_INIT_PARAMS: scale=([\d.]+)', output)
    final_params_match = re.search(r'DEBUG_PARAMS: scale=([\d.]+)', output)

    data = {'detector': 'C++ MTCNN'}

    if bbox_match:
        data['bbox'] = tuple(float(bbox_match.group(i)) for i in range(1, 5))
        print(f"✓ Detection: {data['bbox']}")

    if init_params_match:
        data['init_scale'] = float(init_params_match.group(1))
        print(f"✓ Init scale: {data['init_scale']:.6f}")

    if final_params_match:
        data['final_scale'] = float(final_params_match.group(1))
        print(f"✓ Final scale: {data['final_scale']:.6f}")

    # Read final landmarks from CSV
    csv_path = output_dir / f"{image_path.stem}.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)

        x_cols = [col for col in df.columns if col.startswith('x_')]
        y_cols = [col for col in df.columns if col.startswith('y_')]

        if x_cols and y_cols:
            x_coords = df[x_cols].values[0]
            y_coords = df[y_cols].values[0]
            data['landmarks'] = np.column_stack([x_coords, y_coords])
            print(f"✓ Final landmarks: mean=({data['landmarks'][:, 0].mean():.2f}, {data['landmarks'][:, 1].mean():.2f})")

    return data

def run_corrected_retinaface_pipeline(image_path):
    """Run Python pipeline with CORRECTED RetinaFace."""
    from pyfaceau.detectors.retinaface import ONNXRetinaFaceDetector
    from pyclnf import CLNF

    print("\n" + "="*80)
    print("PYTHON PYCLNF PIPELINE - CORRECTED RETINAFACE")
    print("="*80)
    print("Detector: RetinaFace + Correction")
    print("Fitting: Python CLNF")
    print()

    # Load image
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect with RetinaFace
    retinaface_model = Path("pyfaceau/weights/retinaface_mobilenet025_coreml.onnx")
    detector = ONNXRetinaFaceDetector(
        str(retinaface_model),
        use_coreml=False,
        confidence_threshold=0.5,
        nms_threshold=0.4
    )

    detections, _ = detector.detect_faces(img, resize=1.0)

    if len(detections) == 0:
        print("✗ ERROR: RetinaFace found no faces!")
        return None

    # Get first detection
    detection = detections[0]
    x1, y1, x2, y2 = detection[:4]
    raw_bbox = (x1, y1, x2 - x1, y2 - y1)

    print(f"✓ Raw RetinaFace bbox: ({raw_bbox[0]:.1f}, {raw_bbox[1]:.1f}, {raw_bbox[2]:.1f}, {raw_bbox[3]:.1f})")

    # Apply correction
    corrected_bbox = apply_retinaface_correction(raw_bbox)
    print(f"✓ Corrected bbox: ({corrected_bbox[0]:.1f}, {corrected_bbox[1]:.1f}, {corrected_bbox[2]:.1f}, {corrected_bbox[3]:.1f})")

    # Initialize CLNF
    model_dir = Path("pyclnf/models")
    clnf = CLNF(model_dir=str(model_dir))

    # Get init scale for comparison
    init_params = clnf.pdm.init_params(corrected_bbox)
    init_scale = init_params[0]
    print(f"✓ Init scale: {init_scale:.6f}")

    # Run fitting
    print("Running fitting...")
    final_landmarks, info = clnf.fit(gray, corrected_bbox, return_params=True)

    # Get final params
    final_params = info.get('params', None)
    if final_params is not None:
        final_scale = final_params[0]
    else:
        final_scale = init_scale

    print(f"✓ Final scale: {final_scale:.6f}")
    print(f"✓ Final landmarks: mean=({final_landmarks[:, 0].mean():.2f}, {final_landmarks[:, 1].mean():.2f})")

    data = {
        'detector': 'RetinaFace (Corrected)',
        'raw_bbox': raw_bbox,
        'corrected_bbox': corrected_bbox,
        'init_scale': init_scale,
        'final_scale': final_scale,
        'landmarks': final_landmarks
    }

    return data

def visualize_validation(image_path, cpp_data, retina_data, output_path):
    """Create comprehensive validation visualization."""
    img = cv2.imread(str(image_path))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Row 1: BBox comparison
    # C++ Detection
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb_img)
    if 'bbox' in cpp_data:
        x, y, w, h = cpp_data['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=3, label='C++ MTCNN')
        ax.add_patch(rect)
    ax.set_title(f"C++ MTCNN Detection\nBBox: {w:.0f}×{h:.0f}\nInit Scale: {cpp_data.get('init_scale', 0):.4f}",
                 fontsize=12, fontweight='bold', color='darkred')
    ax.axis('off')

    # Raw RetinaFace
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(rgb_img)
    if retina_data and 'raw_bbox' in retina_data:
        x, y, w, h = retina_data['raw_bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='orange', linewidth=3, linestyle='--', label='Raw')
        ax.add_patch(rect)
    if retina_data and 'corrected_bbox' in retina_data:
        x, y, w, h = retina_data['corrected_bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='blue', linewidth=3, label='Corrected')
        ax.add_patch(rect)
    ax.set_title(f"RetinaFace: Raw vs Corrected\nCorrected: {w:.0f}×{h:.0f}\nInit Scale: {retina_data.get('init_scale', 0):.4f}",
                 fontsize=12, fontweight='bold', color='darkblue')
    ax.legend(loc='upper right')
    ax.axis('off')

    # BBox Overlay
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(rgb_img)
    if 'bbox' in cpp_data:
        x, y, w, h = cpp_data['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2, label='C++ MTCNN')
        ax.add_patch(rect)
    if retina_data and 'corrected_bbox' in retina_data:
        x, y, w, h = retina_data['corrected_bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='blue', linewidth=2, label='Corrected RetinaFace')
        ax.add_patch(rect)

    if retina_data and 'init_scale' in retina_data and 'init_scale' in cpp_data:
        init_error = abs(retina_data['init_scale'] - cpp_data['init_scale'])
        ax.set_title(f"BBox Comparison\nInit Scale Error: {init_error:.4f} ({(init_error/cpp_data['init_scale'])*100:.2f}%)",
                     fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.axis('off')

    # Row 2: Final landmarks
    # C++ Final
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(rgb_img)
    if 'landmarks' in cpp_data:
        lm = cpp_data['landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='red', s=20, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.set_title(f"C++ OpenFace Final\nScale: {cpp_data.get('final_scale', 0):.4f}",
                 fontsize=12, fontweight='bold', color='darkred')
    ax.axis('off')

    # Python Final
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(rgb_img)
    if retina_data and 'landmarks' in retina_data:
        lm = retina_data['landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='blue', s=20, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.set_title(f"Python pyCLNF Final (Corrected RetinaFace)\nScale: {retina_data.get('final_scale', 0):.4f}",
                 fontsize=12, fontweight='bold', color='darkblue')
    ax.axis('off')

    # Overlay comparison
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(rgb_img)
    if 'landmarks' in cpp_data:
        lm = cpp_data['landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='red', s=30, alpha=0.6, label='C++ OpenFace', marker='o')
    if retina_data and 'landmarks' in retina_data:
        lm = retina_data['landmarks']
        ax.scatter(lm[:, 0], lm[:, 1], c='blue', s=30, alpha=0.6, label='Python pyCLNF', marker='x')

    if retina_data and 'landmarks' in retina_data and 'landmarks' in cpp_data:
        diff = np.linalg.norm(cpp_data['landmarks'] - retina_data['landmarks'], axis=1).mean()
        ax.set_title(f"Final Landmark Comparison\nMean Difference: {diff:.2f}px",
                     fontsize=12, fontweight='bold')

    ax.legend(loc='upper right')
    ax.axis('off')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Visualization saved: {output_path}")
    print(f"{'='*80}")
    plt.close()

def main():
    """Main validation function."""
    # Test on a calibration frame
    image_path = Path("calibration_frames/patient1_frame1.jpg")
    output_dir = Path("test_output/corrected_retinaface_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CORRECTED RETINAFACE VALIDATION")
    print("="*80)
    print(f"Image: {image_path}")
    print()
    print("Testing pipeline with calibration-derived correction factor")
    print()

    # Run both pipelines
    cpp_data = run_cpp_pipeline(image_path, output_dir)
    retina_data = run_corrected_retinaface_pipeline(image_path)

    if not cpp_data or not retina_data:
        print("ERROR: One or both pipelines failed!")
        return

    # Create visualization
    output_path = output_dir / f"{image_path.stem}_corrected_validation.jpg"
    visualize_validation(image_path, cpp_data, retina_data, output_path)

    # Final summary
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    # Initialization comparison
    init_error = abs(retina_data['init_scale'] - cpp_data['init_scale'])
    init_percent = (init_error / cpp_data['init_scale']) * 100

    print(f"\nInitialization:")
    print(f"  C++ MTCNN scale:         {cpp_data['init_scale']:.6f}")
    print(f"  Corrected RetinaFace:    {retina_data['init_scale']:.6f}")
    print(f"  Error: {init_error:.6f} ({init_percent:.2f}%)")

    if init_percent < 1.0:
        print(f"  ✓ PERFECT - Sub-1% error!")
    elif init_percent < 3.0:
        print(f"  ✓ EXCELLENT - Within calibration target")
    elif init_percent < 5.0:
        print(f"  ✓ GOOD")
    else:
        print(f"  ⚠ Needs improvement")

    # Final convergence comparison
    if 'landmarks' in retina_data and 'landmarks' in cpp_data:
        landmark_diff = np.linalg.norm(cpp_data['landmarks'] - retina_data['landmarks'], axis=1).mean()

        print(f"\nFinal Convergence:")
        print(f"  Mean landmark difference: {landmark_diff:.2f}px")

        if landmark_diff < 5.0:
            print(f"  ✓ EXCELLENT - Matches C++ OpenFace!")
        elif landmark_diff < 15.0:
            print(f"  ✓ GOOD - Acceptable accuracy")
        elif landmark_diff < 30.0:
            print(f"  ⚠ MODERATE")
        else:
            print(f"  ✗ POOR")

    print(f"\n{'='*80}")
    print("CONCLUSION:")
    print("="*80)

    if init_percent < 3.0 and landmark_diff < 15.0:
        print("✓ SUCCESS! Corrected RetinaFace achieves excellent accuracy.")
        print("  Ready for integration into pyCLNF pipeline.")
    elif init_percent < 5.0 and landmark_diff < 30.0:
        print("✓ GOOD! Corrected RetinaFace provides acceptable accuracy.")
        print("  Can be integrated with confidence.")
    else:
        print("⚠ Correction may need refinement for production use.")

    print(f"\nVisualization: {output_path}")
    print()

if __name__ == "__main__":
    main()
