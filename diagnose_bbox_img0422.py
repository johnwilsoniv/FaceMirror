#!/usr/bin/env python3
"""
Diagnose bbox and initialization differences on IMG_0422 (difficult frame).
Compare C++ MTCNN vs Python RetinaFace V2 correction.
"""

import subprocess
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

def run_cpp_openface_with_debug(image_path, output_dir):
    """Run C++ OpenFace and extract bbox/init params from debug output."""
    binary_path = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    cmd = [
        str(binary_path),
        "-f", str(image_path),
        "-out_dir", str(output_dir),
        "-wild",  # Video mode
        "-verbose"
    ]

    print("="*80)
    print("C++ OPENFACE DETECTION & INITIALIZATION")
    print("="*80)

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Extract bbox from debug output
    bbox_match = re.search(r'DEBUG_BBOX: ([\d.]+),([\d.]+),([\d.]+),([\d.]+)', output)
    init_params_match = re.search(r'DEBUG_INIT_PARAMS: scale=([\d.]+)', output)
    final_params_match = re.search(r'DEBUG_PARAMS: scale=([\d.]+)', output)

    data = {}

    if bbox_match:
        bbox = tuple(float(bbox_match.group(i)) for i in range(1, 5))
        data['bbox'] = bbox
        print(f"✓ C++ MTCNN bbox: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")
        print(f"  Center: ({bbox[0] + bbox[2]/2:.1f}, {bbox[1] + bbox[3]/2:.1f})")
        print(f"  Size: {bbox[2]:.1f} × {bbox[3]:.1f}")
    else:
        print("⚠ Could not extract C++ bbox from debug output")

    if init_params_match:
        init_scale = float(init_params_match.group(1))
        data['init_scale'] = init_scale
        print(f"✓ C++ init scale: {init_scale:.6f}")

    if final_params_match:
        final_scale = float(final_params_match.group(1))
        data['final_scale'] = final_scale
        print(f"✓ C++ final scale: {final_scale:.6f}")

    # Read final landmarks
    csv_path = output_dir / f"{image_path.stem}.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        x_cols = [col for col in df.columns if col.startswith('x_')]
        y_cols = [col for col in df.columns if col.startswith('y_')]

        if x_cols and y_cols:
            x_coords = df[x_cols].values[0]
            y_coords = df[y_cols].values[0]
            landmarks = np.column_stack([x_coords, y_coords])
            data['landmarks'] = landmarks
            print(f"✓ C++ landmarks: {len(landmarks)} points")

    return data


def run_python_retinaface_corrected(image_path):
    """Run Python RetinaFace with V2 correction."""
    from pyclnf import CLNF
    from pyclnf.utils.retinaface_correction import RetinaFaceCorrectedDetector

    print("\n" + "="*80)
    print("PYTHON RETINAFACE V2 DETECTION & INITIALIZATION")
    print("="*80)

    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # RetinaFace detection + correction
    retinaface_model = Path("S1 Face Mirror/weights/retinaface_mobilenet025_coreml.onnx")
    detector = RetinaFaceCorrectedDetector(
        str(retinaface_model),
        use_coreml=False,
        confidence_threshold=0.5,
        nms_threshold=0.4
    )

    # Get raw and corrected bboxes
    from pyfaceau.detectors.retinaface import ONNXRetinaFaceDetector
    raw_detector = ONNXRetinaFaceDetector(
        str(retinaface_model),
        use_coreml=False,
        confidence_threshold=0.5,
        nms_threshold=0.4
    )

    raw_detections, _ = raw_detector.detect_faces(img, resize=1.0)
    if len(raw_detections) == 0:
        print("✗ ERROR: No faces detected!")
        return None

    # Raw bbox
    x1, y1, x2, y2 = raw_detections[0][:4]
    raw_bbox = (x1, y1, x2 - x1, y2 - y1)
    print(f"✓ RetinaFace RAW: ({raw_bbox[0]:.1f}, {raw_bbox[1]:.1f}, {raw_bbox[2]:.1f}, {raw_bbox[3]:.1f})")
    print(f"  Center: ({raw_bbox[0] + raw_bbox[2]/2:.1f}, {raw_bbox[1] + raw_bbox[3]/2:.1f})")
    print(f"  Size: {raw_bbox[2]:.1f} × {raw_bbox[3]:.1f}")

    # Corrected bbox
    corrected_bboxes = detector.detect_and_correct(img)
    corrected_bbox = corrected_bboxes[0]
    print(f"✓ RetinaFace V2 CORRECTED: ({corrected_bbox[0]:.1f}, {corrected_bbox[1]:.1f}, {corrected_bbox[2]:.1f}, {corrected_bbox[3]:.1f})")
    print(f"  Center: ({corrected_bbox[0] + corrected_bbox[2]/2:.1f}, {corrected_bbox[1] + corrected_bbox[3]/2:.1f})")
    print(f"  Size: {corrected_bbox[2]:.1f} × {corrected_bbox[3]:.1f}")

    # Initialize pyCLNF and get init params
    clnf = CLNF(detector=None)
    init_params = clnf.pdm.init_params(corrected_bbox)
    init_scale = init_params[0]
    print(f"✓ Python init scale: {init_scale:.6f}")

    # Fit to get final results
    print("Running CLNF fitting...")
    landmarks, info = clnf.fit(gray, corrected_bbox, return_params=True)
    final_params = info.get('params')
    final_scale = final_params[0] if final_params is not None else init_scale
    print(f"✓ Python final scale: {final_scale:.6f}")
    print(f"✓ Python landmarks: {len(landmarks)} points ({info['iterations']} iterations)")

    data = {
        'raw_bbox': raw_bbox,
        'corrected_bbox': corrected_bbox,
        'init_scale': init_scale,
        'final_scale': final_scale,
        'landmarks': landmarks
    }

    return data


def visualize_bbox_comparison(image_path, cpp_data, py_data, output_path):
    """Create detailed bbox comparison visualization."""
    img = cv2.imread(str(image_path))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'BBOX INITIALIZATION DIAGNOSIS: {image_path.name}',
                 fontsize=16, fontweight='bold')

    # Row 1: BBox comparisons
    # C++ MTCNN
    ax = axes[0, 0]
    ax.imshow(rgb_img)
    if 'bbox' in cpp_data:
        x, y, w, h = cpp_data['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=3)
        ax.add_patch(rect)
        # Center point
        cx, cy = x + w/2, y + h/2
        ax.plot(cx, cy, 'r+', markersize=15, markeredgewidth=3)
    ax.set_title(f"C++ MTCNN\nScale: {cpp_data.get('init_scale', 0):.4f}",
                fontweight='bold', color='darkred')
    ax.axis('off')

    # Python Raw RetinaFace
    ax = axes[0, 1]
    ax.imshow(rgb_img)
    if py_data and 'raw_bbox' in py_data:
        x, y, w, h = py_data['raw_bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='orange', linewidth=3, linestyle='--')
        ax.add_patch(rect)
        cx, cy = x + w/2, y + h/2
        ax.plot(cx, cy, 'orange', marker='+', markersize=15, markeredgewidth=3)
    ax.set_title("RetinaFace RAW (Uncorrected)", fontweight='bold', color='darkorange')
    ax.axis('off')

    # Python Corrected RetinaFace
    ax = axes[0, 2]
    ax.imshow(rgb_img)
    if py_data and 'corrected_bbox' in py_data:
        x, y, w, h = py_data['corrected_bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='blue', linewidth=3)
        ax.add_patch(rect)
        cx, cy = x + w/2, y + h/2
        ax.plot(cx, cy, 'b+', markersize=15, markeredgewidth=3)
    ax.set_title(f"RetinaFace V2 CORRECTED\nScale: {py_data.get('init_scale', 0):.4f}",
                fontweight='bold', color='darkblue')
    ax.axis('off')

    # Row 2: Overlays and analysis
    # All bboxes overlay
    ax = axes[1, 0]
    ax.imshow(rgb_img)
    if 'bbox' in cpp_data:
        x, y, w, h = cpp_data['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2, label='C++ MTCNN')
        ax.add_patch(rect)
    if py_data and 'raw_bbox' in py_data:
        x, y, w, h = py_data['raw_bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='orange', linewidth=2,
                             linestyle='--', label='Raw RetinaFace', alpha=0.7)
        ax.add_patch(rect)
    if py_data and 'corrected_bbox' in py_data:
        x, y, w, h = py_data['corrected_bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='blue', linewidth=2, label='Corrected RetinaFace')
        ax.add_patch(rect)
    ax.set_title("All BBoxes Overlay", fontweight='bold')
    ax.legend(loc='upper right')
    ax.axis('off')

    # C++ vs Corrected Python
    ax = axes[1, 1]
    ax.imshow(rgb_img)
    if 'bbox' in cpp_data:
        x, y, w, h = cpp_data['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=3, label='C++ MTCNN')
        ax.add_patch(rect)
    if py_data and 'corrected_bbox' in py_data:
        x, y, w, h = py_data['corrected_bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='blue', linewidth=3, label='Python V2')
        ax.add_patch(rect)

    if py_data and 'corrected_bbox' in py_data and 'bbox' in cpp_data:
        # Draw line between centers
        cpp_bbox = cpp_data['bbox']
        py_bbox = py_data['corrected_bbox']
        cpp_center = (cpp_bbox[0] + cpp_bbox[2]/2, cpp_bbox[1] + cpp_bbox[3]/2)
        py_center = (py_bbox[0] + py_bbox[2]/2, py_bbox[1] + py_bbox[3]/2)
        ax.plot([cpp_center[0], py_center[0]], [cpp_center[1], py_center[1]],
               'g--', linewidth=2, alpha=0.7)

        center_offset = np.sqrt((cpp_center[0] - py_center[0])**2 + (cpp_center[1] - py_center[1])**2)
        ax.set_title(f"C++ vs Python Corrected\nCenter offset: {center_offset:.1f}px", fontweight='bold')

    ax.legend(loc='upper right')
    ax.axis('off')

    # Statistics panel
    ax = axes[1, 2]
    ax.axis('off')

    text = "BBOX COMPARISON STATISTICS\n"
    text += "="*40 + "\n\n"

    if 'bbox' in cpp_data and py_data and 'corrected_bbox' in py_data:
        cpp_bbox = cpp_data['bbox']
        py_bbox = py_data['corrected_bbox']

        # Center offset
        cpp_center = (cpp_bbox[0] + cpp_bbox[2]/2, cpp_bbox[1] + cpp_bbox[3]/2)
        py_center = (py_bbox[0] + py_bbox[2]/2, py_bbox[1] + py_bbox[3]/2)
        center_offset = np.sqrt((cpp_center[0] - py_center[0])**2 + (cpp_center[1] - py_center[1])**2)

        text += "Center Alignment:\n"
        text += f"  C++: ({cpp_center[0]:.1f}, {cpp_center[1]:.1f})\n"
        text += f"  Py:  ({py_center[0]:.1f}, {py_center[1]:.1f})\n"
        text += f"  Offset: {center_offset:.1f}px\n"
        if center_offset < 10:
            text += "  ✓ EXCELLENT\n\n"
        elif center_offset < 30:
            text += "  ✓ GOOD\n\n"
        else:
            text += "  ⚠ LARGE OFFSET!\n\n"

        # Size differences
        width_diff = abs(cpp_bbox[2] - py_bbox[2])
        height_diff = abs(cpp_bbox[3] - py_bbox[3])

        text += "Size Differences:\n"
        text += f"  Width:  {cpp_bbox[2]:.1f} vs {py_bbox[2]:.1f}\n"
        text += f"          Δ = {width_diff:.1f}px\n"
        text += f"  Height: {cpp_bbox[3]:.1f} vs {py_bbox[3]:.1f}\n"
        text += f"          Δ = {height_diff:.1f}px\n\n"

        # Init scale
        if 'init_scale' in cpp_data and 'init_scale' in py_data:
            init_error = abs(py_data['init_scale'] - cpp_data['init_scale'])
            init_percent = (init_error / cpp_data['init_scale']) * 100

            text += "Initialization Scale:\n"
            text += f"  C++: {cpp_data['init_scale']:.6f}\n"
            text += f"  Py:  {py_data['init_scale']:.6f}\n"
            text += f"  Error: {init_percent:.2f}%\n"
            if init_percent < 1.0:
                text += "  ✓ SUB-1% (PERFECT)\n\n"
            elif init_percent < 3.0:
                text += "  ✓ EXCELLENT\n\n"
            elif init_percent < 10.0:
                text += "  ⚠ ACCEPTABLE\n\n"
            else:
                text += "  ❌ POOR INIT!\n\n"

        # Landmark accuracy (if available)
        if 'landmarks' in cpp_data and 'landmarks' in py_data:
            landmark_error = np.linalg.norm(cpp_data['landmarks'] - py_data['landmarks'], axis=1).mean()
            text += "Final Landmark Error:\n"
            text += f"  Mean: {landmark_error:.2f}px\n"
            if landmark_error < 10:
                text += "  ✓ EXCELLENT\n"
            elif landmark_error < 20:
                text += "  ⚠ MODERATE\n"
            else:
                text += "  ❌ POOR\n"

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Visualization saved: {output_path}")
    print(f"{'='*80}")
    plt.close()


def main():
    """Diagnose IMG_0422 bbox and initialization."""
    # Use IMG_0422 frame (the difficult one)
    image_path = Path("test_output/baseline_fast/frames/IMG_0422_frame_0000.jpg")

    if not image_path.exists():
        print(f"ERROR: Frame not found: {image_path}")
        print("Extracting frame...")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-i", "Patient Data/Normal Cohort/IMG_0422.MOV",
            "-vframes", "1", "-y", str(image_path)
        ], capture_output=True)

    output_dir = Path("test_output/bbox_diagnosis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("BBOX INITIALIZATION DIAGNOSIS: IMG_0422 (Difficult Frame)")
    print("="*80)
    print(f"Image: {image_path}")
    print()

    # Run both pipelines
    cpp_output_dir = output_dir / "cpp_output"
    cpp_output_dir.mkdir(parents=True, exist_ok=True)

    cpp_data = run_cpp_openface_with_debug(image_path, cpp_output_dir)
    py_data = run_python_retinaface_corrected(image_path)

    if not cpp_data or not py_data:
        print("ERROR: One or both pipelines failed!")
        return

    # Create visualization
    viz_path = output_dir / "bbox_comparison_IMG_0422.jpg"
    visualize_bbox_comparison(image_path, cpp_data, py_data, viz_path)

    # Print summary
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)

    if 'bbox' in cpp_data and 'corrected_bbox' in py_data:
        cpp_bbox = cpp_data['bbox']
        py_bbox = py_data['corrected_bbox']

        cpp_center = (cpp_bbox[0] + cpp_bbox[2]/2, cpp_bbox[1] + cpp_bbox[3]/2)
        py_center = (py_bbox[0] + py_bbox[2]/2, py_bbox[1] + py_bbox[3]/2)
        center_offset = np.sqrt((cpp_center[0] - py_center[0])**2 + (cpp_center[1] - py_center[1])**2)

        print(f"\nBBox Center Offset: {center_offset:.1f}px")
        print(f"Width Difference: {abs(cpp_bbox[2] - py_bbox[2]):.1f}px")
        print(f"Height Difference: {abs(cpp_bbox[3] - py_bbox[3]):.1f}px")

    if 'init_scale' in cpp_data and 'init_scale' in py_data:
        init_error = abs(py_data['init_scale'] - cpp_data['init_scale'])
        init_percent = (init_error / cpp_data['init_scale']) * 100
        print(f"\nInit Scale Error: {init_percent:.2f}%")

    if 'landmarks' in cpp_data and 'landmarks' in py_data:
        landmark_error = np.linalg.norm(cpp_data['landmarks'] - py_data['landmarks'], axis=1).mean()
        print(f"Final Landmark Error: {landmark_error:.2f}px")

    print(f"\nVisualization: {viz_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
