#!/usr/bin/env python3
"""
Derive RetinaFace Correction Factor V2

IMPROVED VERSION: Optimizes for bbox coordinate alignment, not just init scale.

The problem with V1 was that init scale can match even with displaced bboxes.
V2 ensures the corrected bbox spatially aligns with C++ MTCNN bbox.

Objective: Minimize bbox coordinate differences:
    - Center position (x + w/2, y + h/2)
    - Width
    - Height
"""

import subprocess
import re
import cv2
import numpy as np
from pathlib import Path
import sys
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Add pyfaceau to path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

def run_cpp_mtcnn(image_path, output_dir):
    """Run C++ MTCNN and get bbox."""
    binary_path = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    cmd = [
        str(binary_path),
        "-f", str(image_path),
        "-out_dir", str(output_dir),
        "-verbose"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    bbox_match = re.search(r'DEBUG_BBOX: ([\d.]+),([\d.]+),([\d.]+),([\d.]+)', output)

    if not bbox_match:
        return None

    return {
        'bbox': tuple(float(bbox_match.group(i)) for i in range(1, 5))
    }

def run_retinaface(image_path):
    """Run RetinaFace and get bbox."""
    from pyfaceau.detectors.retinaface import ONNXRetinaFaceDetector

    img = cv2.imread(str(image_path))

    retinaface_model = Path("pyfaceau/weights/retinaface_mobilenet025_coreml.onnx")
    detector = ONNXRetinaFaceDetector(
        str(retinaface_model),
        use_coreml=False,
        confidence_threshold=0.5,
        nms_threshold=0.4
    )

    detections, _ = detector.detect_faces(img, resize=1.0)

    if len(detections) == 0:
        return None

    detection = detections[0]
    x1, y1, x2, y2 = detection[:4]

    return {
        'bbox': (x1, y1, x2 - x1, y2 - y1)
    }

def apply_correction(retina_bbox, params):
    """Apply correction transform."""
    rx, ry, rw, rh = retina_bbox
    alpha, beta, gamma, delta = params

    cx = rx + alpha * rw
    cy = ry + beta * rh
    cw = rw * gamma
    ch = rh * delta

    return (cx, cy, cw, ch)

def bbox_distance(bbox1, bbox2):
    """
    Compute distance between two bboxes.

    Uses weighted combination of:
    - Center distance
    - Width difference
    - Height difference
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Center points
    cx1, cy1 = x1 + w1/2, y1 + h1/2
    cx2, cy2 = x2 + w2/2, y2 + h2/2

    # Center distance
    center_dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    # Size differences
    width_diff = abs(w1 - w2)
    height_diff = abs(h1 - h2)

    # Weighted combination
    # Center is most important (weight=2), then size (weight=1)
    total_distance = 2.0 * center_dist + width_diff + height_diff

    return total_distance

def objective_function_v2(params, calibration_data):
    """
    Objective function V2: Minimize bbox coordinate differences.

    Args:
        params: [alpha, beta, gamma, delta] to optimize
        calibration_data: List of dicts with 'cpp' and 'retina' data

    Returns:
        total_error: Sum of bbox distances
    """
    total_error = 0.0

    for data in calibration_data:
        # Apply correction to RetinaFace bbox
        corrected_bbox = apply_correction(data['retina']['bbox'], params)

        # Compute distance to target C++ bbox
        cpp_bbox = data['cpp']['bbox']
        distance = bbox_distance(corrected_bbox, cpp_bbox)

        total_error += distance**2  # Square for optimization

    return total_error

def collect_calibration_data(frame_paths, output_dir):
    """Collect bbox data from both detectors."""
    calibration_data = []

    print("="*80)
    print("COLLECTING CALIBRATION DATA")
    print("="*80)

    for i, frame_path in enumerate(frame_paths, 1):
        print(f"\nFrame {i}/{len(frame_paths)}: {frame_path.name}")
        print("-" * 80)

        # Run C++ MTCNN
        cpp_data = run_cpp_mtcnn(frame_path, output_dir)
        if cpp_data is None:
            print("  ✗ C++ MTCNN failed")
            continue

        x, y, w, h = cpp_data['bbox']
        print(f"  C++ MTCNN: ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")

        # Run RetinaFace
        retina_data = run_retinaface(frame_path)
        if retina_data is None:
            print("  ✗ RetinaFace failed")
            continue

        x, y, w, h = retina_data['bbox']
        print(f"  RetinaFace: ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")

        calibration_data.append({
            'frame': frame_path.name,
            'cpp': cpp_data,
            'retina': retina_data
        })

        print(f"  ✓ Data collected")

    print(f"\n{'='*80}")
    print(f"Collected {len(calibration_data)} samples")
    print(f"{'='*80}\n")

    return calibration_data

def optimize_correction_params_v2(calibration_data):
    """Optimize correction parameters to minimize bbox differences."""
    print("="*80)
    print("OPTIMIZING CORRECTION PARAMETERS V2")
    print("="*80)
    print("Objective: Minimize bbox coordinate differences")
    print()

    # Initial guess from manual analysis
    initial_params = [0.0, 0.29, 1.0, 0.76]

    print(f"Initial guess: alpha={initial_params[0]:.4f}, beta={initial_params[1]:.4f}, " +
          f"gamma={initial_params[2]:.4f}, delta={initial_params[3]:.4f}")

    # Bounds
    bounds = [
        (-0.3, 0.3),   # alpha: horizontal shift
        (0.0, 0.5),    # beta: vertical shift
        (0.7, 1.3),    # gamma: width scale (expanded range)
        (0.5, 1.0),    # delta: height scale
    ]

    print("\nOptimizing for bbox alignment...")
    result = minimize(
        objective_function_v2,
        initial_params,
        args=(calibration_data,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )

    if result.success:
        print("✓ Optimization converged!")
    else:
        print("⚠ Optimization stopped but found best parameters")

    optimal_params = result.x
    print(f"\nOptimal parameters:")
    print(f"  alpha = {optimal_params[0]:.6f}  (horizontal shift)")
    print(f"  beta  = {optimal_params[1]:.6f}  (vertical shift)")
    print(f"  gamma = {optimal_params[2]:.6f}  (width scale)")
    print(f"  delta = {optimal_params[3]:.6f}  (height scale)")
    print(f"\nFinal error: {result.fun:.8f}")
    print(f"{'='*80}\n")

    return optimal_params

def validate_correction_v2(calibration_data, correction_params):
    """Validate bbox alignment and init scale."""
    from pyclnf.core import PDM

    model_dir = Path("pyclnf/models/exported_pdm")
    pdm = PDM(str(model_dir))

    print("="*80)
    print("VALIDATING CORRECTION V2")
    print("="*80)

    bbox_errors = []
    scale_errors = []

    for i, data in enumerate(calibration_data, 1):
        # Apply correction
        corrected_bbox = apply_correction(data['retina']['bbox'], correction_params)
        cpp_bbox = data['cpp']['bbox']

        # BBox alignment
        cx_corr, cy_corr = corrected_bbox[0] + corrected_bbox[2]/2, corrected_bbox[1] + corrected_bbox[3]/2
        cx_cpp, cy_cpp = cpp_bbox[0] + cpp_bbox[2]/2, cpp_bbox[1] + cpp_bbox[3]/2

        center_error = np.sqrt((cx_corr - cx_cpp)**2 + (cy_corr - cy_cpp)**2)
        width_error = abs(corrected_bbox[2] - cpp_bbox[2])
        height_error = abs(corrected_bbox[3] - cpp_bbox[3])

        bbox_error = center_error + width_error + height_error
        bbox_errors.append(bbox_error)

        # Init scale comparison
        corrected_scale = pdm.init_params(corrected_bbox)[0]
        cpp_scale = pdm.init_params(cpp_bbox)[0]
        scale_error = abs(corrected_scale - cpp_scale)
        scale_errors.append(scale_error)

        print(f"\nFrame {i}: {data['frame']}")
        print(f"  BBox alignment:")
        print(f"    Center error:  {center_error:.2f}px")
        print(f"    Width error:   {width_error:.2f}px")
        print(f"    Height error:  {height_error:.2f}px")
        print(f"    Total: {bbox_error:.2f}px")

        print(f"  Init scale:")
        print(f"    C++ MTCNN:     {cpp_scale:.6f}")
        print(f"    Corrected:     {corrected_scale:.6f}")
        print(f"    Error: {scale_error:.6f} ({(scale_error/cpp_scale)*100:.2f}%)")

        if center_error < 10 and width_error < 20 and height_error < 20:
            print(f"  ✓ EXCELLENT bbox alignment")
        elif center_error < 20 and width_error < 40 and height_error < 40:
            print(f"  ✓ GOOD bbox alignment")
        else:
            print(f"  ⚠ Moderate bbox alignment")

    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY V2")
    print(f"{'='*80}")
    print(f"\nBBox Alignment:")
    print(f"  Mean total error:  {np.mean(bbox_errors):.2f}px")
    print(f"  Max total error:   {np.max(bbox_errors):.2f}px")

    print(f"\nInit Scale:")
    print(f"  Mean error:   {np.mean(scale_errors):.6f}")
    print(f"  Mean % error: {(np.mean(scale_errors) / np.mean([pdm.init_params(d['cpp']['bbox'])[0] for d in calibration_data])) * 100:.3f}%")

    if np.mean(bbox_errors) < 30:
        print(f"\n✓ EXCELLENT! BBox alignment achieved")
    elif np.mean(bbox_errors) < 60:
        print(f"\n✓ GOOD! BBox alignment is acceptable")
    else:
        print(f"\n⚠ BBox alignment needs improvement")

    print(f"{'='*80}\n")

    return bbox_errors, scale_errors

def create_visualization_v2(calibration_data, correction_params, output_path):
    """Create visualization of V2 results."""
    from pyclnf.core import PDM

    model_dir = Path("pyclnf/models/exported_pdm")
    pdm = PDM(str(model_dir))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: BBox center alignment
    ax = axes[0, 0]
    frame_indices = np.arange(len(calibration_data))

    cpp_centers_x = [(d['cpp']['bbox'][0] + d['cpp']['bbox'][2]/2) for d in calibration_data]
    cpp_centers_y = [(d['cpp']['bbox'][1] + d['cpp']['bbox'][3]/2) for d in calibration_data]

    corr_centers_x = []
    corr_centers_y = []
    for d in calibration_data:
        corr = apply_correction(d['retina']['bbox'], correction_params)
        corr_centers_x.append(corr[0] + corr[2]/2)
        corr_centers_y.append(corr[1] + corr[3]/2)

    ax.scatter(cpp_centers_x, cpp_centers_y, c='red', s=100, marker='o', label='C++ MTCNN', edgecolors='black', linewidth=2)
    ax.scatter(corr_centers_x, corr_centers_y, c='blue', s=100, marker='x', label='Corrected RetinaFace', linewidth=3)

    # Draw connection lines
    for i in range(len(calibration_data)):
        ax.plot([cpp_centers_x[i], corr_centers_x[i]],
               [cpp_centers_y[i], corr_centers_y[i]],
               'k--', alpha=0.3, linewidth=1)

    ax.set_xlabel('X Center (px)', fontsize=12)
    ax.set_ylabel('Y Center (px)', fontsize=12)
    ax.set_title('BBox Center Alignment', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Match image coordinates

    # Plot 2: BBox size comparison
    ax = axes[0, 1]

    cpp_widths = [d['cpp']['bbox'][2] for d in calibration_data]
    cpp_heights = [d['cpp']['bbox'][3] for d in calibration_data]

    corr_widths = []
    corr_heights = []
    for d in calibration_data:
        corr = apply_correction(d['retina']['bbox'], correction_params)
        corr_widths.append(corr[2])
        corr_heights.append(corr[3])

    x_pos = np.arange(len(calibration_data))
    width = 0.35

    ax.bar(x_pos - width/2, cpp_widths, width, label='C++ Width', color='lightcoral', alpha=0.7)
    ax.bar(x_pos + width/2, corr_widths, width, label='Corrected Width', color='lightblue', alpha=0.7)

    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Width (px)', fontsize=12)
    ax.set_title('BBox Width Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Parameters and stats
    ax = axes[1, 0]
    ax.axis('off')

    text = "CORRECTION PARAMETERS V2\n"
    text += "="*50 + "\n\n"
    text += f"alpha = {correction_params[0]:+.6f}\n"
    text += f"beta  = {correction_params[1]:+.6f}\n"
    text += f"gamma = {correction_params[2]:+.6f}\n"
    text += f"delta = {correction_params[3]:+.6f}\n\n"
    text += "OPTIMIZATION TARGET:\n"
    text += "  Minimize bbox coordinate differences\n"
    text += "  (center position + width + height)\n\n"
    text += "="*50 + "\n\n"

    # Calculate errors
    bbox_errors = []
    scale_errors = []
    for d in calibration_data:
        corr = apply_correction(d['retina']['bbox'], correction_params)
        cpp = d['cpp']['bbox']

        center_err = np.sqrt(((corr[0]+corr[2]/2) - (cpp[0]+cpp[2]/2))**2 +
                            ((corr[1]+corr[3]/2) - (cpp[1]+cpp[3]/2))**2)
        width_err = abs(corr[2] - cpp[2])
        height_err = abs(corr[3] - cpp[3])
        bbox_errors.append(center_err + width_err + height_err)

        scale_err = abs(pdm.init_params(corr)[0] - pdm.init_params(cpp)[0])
        scale_errors.append(scale_err)

    text += "VALIDATION RESULTS:\n"
    text += "-"*50 + "\n"
    text += f"Mean bbox error:    {np.mean(bbox_errors):.2f}px\n"
    text += f"Mean scale error:   {np.mean(scale_errors):.6f}\n"
    text += f"Mean scale % error: {(np.mean(scale_errors)/2.8)*100:.2f}%\n\n"

    if np.mean(bbox_errors) < 30:
        text += "✓ EXCELLENT bbox alignment!\n"

    ax.text(0.05, 0.95, text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    # Plot 4: Error distribution
    ax = axes[1, 1]

    center_errors = []
    for d in calibration_data:
        corr = apply_correction(d['retina']['bbox'], correction_params)
        cpp = d['cpp']['bbox']
        center_err = np.sqrt(((corr[0]+corr[2]/2) - (cpp[0]+cpp[2]/2))**2 +
                            ((corr[1]+corr[3]/2) - (cpp[1]+cpp[3]/2))**2)
        center_errors.append(center_err)

    ax.bar(frame_indices, center_errors, color='steelblue', alpha=0.7)
    ax.axhline(y=10, color='green', linestyle='--', linewidth=2, label='10px threshold')
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Center Error (px)', fontsize=12)
    ax.set_title('Center Position Error per Frame', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {output_path}\n")
    plt.close()

def main():
    """Main V2 calibration."""
    calibration_dir = Path("calibration_frames")
    output_dir = Path("test_output/retinaface_correction_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(calibration_dir.glob("*.jpg"))

    print("\n" + "="*80)
    print("RETINAFACE CORRECTION FACTOR DERIVATION V2")
    print("="*80)
    print(f"Calibration frames: {len(frame_paths)}")
    print(f"Goal: Minimize bbox coordinate differences (not just init scale)")
    print()

    # Collect data
    calibration_data = collect_calibration_data(frame_paths, output_dir)

    if len(calibration_data) < 3:
        print("ERROR: Need at least 3 samples")
        return

    # Optimize V2
    correction_params = optimize_correction_params_v2(calibration_data)

    # Validate V2
    bbox_errors, scale_errors = validate_correction_v2(calibration_data, correction_params)

    # Visualize V2
    viz_path = output_dir / "correction_v2_results.jpg"
    create_visualization_v2(calibration_data, correction_params, viz_path)

    # Save parameters
    params_file = output_dir / "retinaface_correction_v2_params.txt"
    with open(params_file, 'w') as f:
        f.write("# RetinaFace Correction Parameters V2\n")
        f.write("# Optimized for bbox coordinate alignment\n\n")
        f.write(f"alpha = {correction_params[0]:.8f}\n")
        f.write(f"beta  = {correction_params[1]:.8f}\n")
        f.write(f"gamma = {correction_params[2]:.8f}\n")
        f.write(f"delta = {correction_params[3]:.8f}\n\n")
        f.write(f"# Validation: Mean bbox error = {np.mean(bbox_errors):.2f}px\n")
        f.write(f"# Validation: Mean scale error = {np.mean(scale_errors):.6f}\n")

    print(f"{'='*80}")
    print(f"V2 PARAMETERS SAVED")
    print(f"{'='*80}")
    print(f"File: {params_file}")
    print(f"Visualization: {viz_path}")
    print()

if __name__ == "__main__":
    main()
